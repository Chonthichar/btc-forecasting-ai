"""Persistent, forward-only evaluation of hourly BTC direction forecasts.

All metric rates are fractions in [0, 1]. Candle timestamps denote the OPEN
of a one-hour candle. No historical forecast is created by settlement.
"""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
HOUR = timedelta(hours=1)


def _utc(value=None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware datetimes or ISO strings")
    return value.astimezone(UTC)


def _iso(value) -> str:
    return _utc(value).isoformat(timespec="microseconds")


def _json(value) -> str:
    def default(item):
        if isinstance(item, datetime):
            return _iso(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    return json.dumps(value, default=default, allow_nan=False, separators=(",", ":"))


def _number(value, name, *, positive=False) -> float:
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def _horizon(value) -> int:
    number = int(value)
    if str(number) != str(value) or number < 1:
        raise ValueError("horizon must be a positive integer")
    return number


class MonitoringStore:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY,
                    asset TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    horizon INTEGER NOT NULL CHECK(horizon > 0),
                    origin_open TEXT NOT NULL,
                    origin_close TEXT NOT NULL,
                    target_open TEXT NOT NULL,
                    target_close TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    current_price REAL NOT NULL CHECK(current_price > 0),
                    probability_up REAL NOT NULL CHECK(probability_up BETWEEN 0 AND 1),
                    predicted_up INTEGER NOT NULL CHECK(predicted_up IN (0, 1)),
                    model TEXT,
                    feature_set TEXT,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'scored', 'excluded_late')),
                    target_price REAL,
                    actual_up INTEGER,
                    correct INTEGER,
                    settled_at TEXT,
                    UNIQUE(asset, model_version, horizon, origin_open)
                );
                CREATE INDEX IF NOT EXISTS forecasts_window
                    ON forecasts(horizon, model_version, issued_at);
                CREATE INDEX IF NOT EXISTS forecasts_due ON forecasts(status, target_close);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY, trigger TEXT NOT NULL, started_at TEXT NOT NULL,
                    finished_at TEXT, status TEXT NOT NULL, detail TEXT, duration_ms REAL
                );
                CREATE TABLE IF NOT EXISTS worker_lock (
                    id INTEGER PRIMARY KEY CHECK(id = 1), owner TEXT NOT NULL, expires_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def set_state(self, key, value):
        encoded = _json(value)
        with self._connection() as db:
            db.execute("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) "
                       "DO UPDATE SET value=excluded.value", (str(key), encoded))

    def get_state(self, key, default=None):
        with self._connection() as db:
            row = db.execute("SELECT value FROM state WHERE key=?", (str(key),)).fetchone()
        return json.loads(row[0]) if row else default

    def save_snapshot(self, snapshot):
        self.set_state("latest_snapshot", snapshot)

    def latest_snapshot(self):
        return self.get_state("latest_snapshot")

    def record_forecasts(self, forecasts, issued_at=None):
        issued = _utc(issued_at)
        rows = []
        for key, forecast in forecasts.items():
            horizon = _horizon(key)
            origin = _utc(forecast["latest_candle"])
            if origin.minute or origin.second or origin.microsecond:
                raise ValueError("latest_candle must be an hourly candle OPEN timestamp")
            origin_close = origin + HOUR
            target_open = origin + horizon * HOUR
            target_close = target_open + HOUR
            if issued < origin_close:
                raise ValueError("Cannot issue a forecast from a candle that has not closed")
            probability = _number(forecast["probability_up"], "probability_up")
            if not 0 <= probability <= 1:
                raise ValueError("probability_up must lie between 0 and 1")
            price = _number(forecast["current_price"], "current_price", positive=True)
            version = str(forecast["model_version"] or "").strip()
            if not version:
                raise ValueError("model_version must not be empty")
            asset = str(forecast.get("asset", "BTC")).strip().upper()
            if not asset:
                raise ValueError("asset must not be empty")
            status = "pending" if issued < target_close else "excluded_late"
            rows.append((asset, version, horizon, _iso(origin), _iso(origin_close),
                         _iso(target_open), _iso(target_close), _iso(issued), price,
                         probability, int(probability >= .5), forecast.get("model"),
                         _json(forecast.get("feature_set")), status))
        result = {"inserted": 0, "duplicates": 0, "pending": 0, "excluded_late": 0}
        with self._connection() as db:
            for row in rows:
                cursor = db.execute("""INSERT INTO forecasts
                    (asset,model_version,horizon,origin_open,origin_close,target_open,
                     target_close,issued_at,current_price,probability_up,predicted_up,
                     model,feature_set,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(asset,model_version,horizon,origin_open) DO NOTHING""", row)
                if cursor.rowcount:
                    result["inserted"] += 1
                    result[row[-1]] += 1
                else:
                    result["duplicates"] += 1
        return result

    def settle(self, candles, now=None):
        current = _utc(now)
        final_candles = {}
        for candle in candles:
            opened = _utc(candle["timestamp"])
            if opened.minute or opened.second or opened.microsecond:
                raise ValueError("Candle timestamps must be hourly OPEN timestamps")
            closed = _utc(candle.get("close_timestamp", opened + HOUR))
            if closed != opened + HOUR:
                raise ValueError("close_timestamp must equal hourly candle open + one hour")
            price = _number(candle["close"], "candle close", positive=True)
            if closed <= current:
                stamp = (str(candle.get("asset", "BTC")).strip().upper(), _iso(opened))
                if stamp in final_candles and final_candles[stamp] != price:
                    raise ValueError("Conflicting close prices for the same candle")
                final_candles[stamp] = price
        scored = 0
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM forecasts WHERE status='pending' AND "
                              "target_close<=? AND issued_at<target_close", (_iso(current),)).fetchall()
            for row in rows:
                candle_key = (row["asset"], row["target_open"])
                if candle_key not in final_candles:
                    continue
                target_price = final_candles[candle_key]
                actual_up = int(target_price > row["current_price"])
                db.execute("UPDATE forecasts SET status='scored', target_price=?,actual_up=?,"
                           "correct=?,settled_at=? WHERE id=? AND status='pending'",
                           (target_price, actual_up, int(actual_up == row["predicted_up"]),
                            _iso(current), row["id"]))
                scored += 1
            awaiting = db.execute("SELECT COUNT(*) FROM forecasts WHERE status='pending' "
                                  "AND target_close<=?", (_iso(current),)).fetchone()[0]
        return {"scored": scored, "awaiting_market_data": awaiting}

    def outstanding_targets(self, limit=1000, now=None):
        """Return oldest unresolved, matured target OPEN timestamps for backfill."""
        limit = int(limit)
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be 1..10000")
        with self._connection() as db:
            rows = db.execute("SELECT DISTINCT target_open FROM forecasts WHERE status='pending' "
                              "AND target_close<=? ORDER BY target_open LIMIT ?",
                              (_iso(now), limit)).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _display(row, now):
        item = dict(row)
        if item["status"] == "pending" and item["target_close"] <= _iso(now):
            item["status"] = "awaiting_market_data"
        if item.get("feature_set") is not None:
            item["feature_set"] = json.loads(item["feature_set"])
        item["latest_candle"] = item["origin_open"]
        return item

    def versions(self, horizon=None):
        clauses, params = [], []
        if horizon is not None:
            clauses.append("horizon=?")
            params.append(_horizon(horizon))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connection() as db:
            rows = db.execute("SELECT model_version,MAX(issued_at) AS latest_issued_at,"
                              "COUNT(*) AS total FROM forecasts" + where +
                              " GROUP BY model_version ORDER BY latest_issued_at DESC,model_version",
                              params).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _days(days):
        days = int(days)
        if not 1 <= days <= 365:
            raise ValueError("days must be between 1 and 365")
        return days

    def history(self, horizon=None, days=7, model_version=None, status=None,
                limit=50, offset=0, now=None):
        current = _utc(now)
        days = self._days(days)
        limit, offset = int(limit), int(offset)
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit must be 1..1000 and offset must be non-negative")
        clauses = ["issued_at>=?", "issued_at<=?"]
        params = [_iso(current - timedelta(days=days)), _iso(current)]
        if horizon is not None:
            clauses.append("horizon=?")
            params.append(_horizon(horizon))
        if model_version is not None:
            clauses.append("model_version=?")
            params.append(str(model_version))
        if status is not None:
            if status in ("pending", "awaiting_market_data"):
                clauses.append("status='pending' AND target_close" + (">?" if status == "pending" else "<=?"))
                params.append(_iso(current))
            elif status in ("scored", "excluded_late"):
                clauses.append("status=?")
                params.append(status)
            else:
                raise ValueError("Unknown forecast status")
        where = " WHERE " + " AND ".join(clauses)
        with self._connection() as db:
            total = db.execute("SELECT COUNT(*) FROM forecasts" + where, params).fetchone()[0]
            rows = db.execute("SELECT * FROM forecasts" + where +
                              " ORDER BY issued_at DESC,id DESC LIMIT ? OFFSET ?",
                              params + [limit, offset]).fetchall()
        return {"items": [self._display(row, current) for row in rows], "total": total,
                "limit": limit, "offset": offset}

    @staticmethod
    def _metrics(rows, now):
        counts = {"correct": 0, "scored": 0, "pending": 0, "awaiting_market_data": 0,
                  "excluded": 0, "total": len(rows)}
        brier, baseline = 0., 0
        timestamp = _iso(now)
        for row in rows:
            if row["status"] == "excluded_late":
                counts["excluded"] += 1
            elif row["status"] == "scored" and row["settled_at"] <= timestamp:
                counts["scored"] += 1
                counts["correct"] += row["correct"]
                baseline += row["actual_up"]
                brier += (row["probability_up"] - row["actual_up"]) ** 2
            elif row["target_close"] <= timestamp:
                counts["awaiting_market_data"] += 1
            else:
                counts["pending"] += 1
        n = counts["scored"]
        return {**counts, "accuracy": counts["correct"] / n if n else None,
                "brier_score": brier / n if n else None,
                "baseline_accuracy": baseline / n if n else None}

    def summary(self, horizon=1, days=7, model_version=None, now=None):
        current, horizon, days = _utc(now), _horizon(horizon), self._days(days)
        cutoff = current - timedelta(days=days)
        with self._connection() as db:
            if model_version is None:
                latest = db.execute("SELECT model_version FROM forecasts WHERE horizon=? "
                                    "AND issued_at<=? ORDER BY issued_at DESC,id DESC LIMIT 1",
                                    (horizon, _iso(current))).fetchone()
                model_version = latest[0] if latest else None
            rows = db.execute("SELECT * FROM forecasts WHERE horizon=? AND model_version=? "
                              "AND issued_at>=? AND issued_at<=? ORDER BY issued_at",
                              (horizon, model_version, _iso(cutoff - timedelta(days=days)),
                               _iso(current))).fetchall()
        in_window = [row for row in rows if row["issued_at"] >= _iso(cutoff)]
        series = []
        start_day = current.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        for index in range(days):
            day = start_day + timedelta(days=index)
            end = min(day + timedelta(days=1), current)
            start = end - timedelta(days=days)
            subset = [row for row in rows if _iso(start) <= row["issued_at"] <= _iso(end)]
            series.append({"date": day.date().isoformat(), "window_end": _iso(end),
                           **self._metrics(subset, end)})
        return {"horizon": horizon, "days": days, "model_version": model_version,
                "as_of": _iso(current), "window_start": _iso(cutoff),
                **self._metrics(in_window, current), "series": series}

    def latest_forecasts(self):
        snapshot = self.latest_snapshot()
        if isinstance(snapshot, dict) and isinstance(snapshot.get("forecasts"), dict):
            return snapshot["forecasts"]
        with self._connection() as db:
            rows = db.execute("SELECT * FROM forecasts ORDER BY issued_at DESC,id DESC").fetchall()
        result = {}
        for row in rows:
            result.setdefault(str(row["horizon"]), self._display(row, _utc()))
        return result

    def start_run(self, trigger, now=None):
        with self._connection() as db:
            cursor = db.execute("INSERT INTO runs(trigger,started_at,status) VALUES(?,?,'running')",
                                (str(trigger), _iso(now)))
            return cursor.lastrowid

    def finish_run(self, run_id, status, detail=None, duration_ms=None, now=None):
        if status not in ("success", "error", "skipped", "failed", "partial"):
            raise ValueError("Invalid terminal run status")
        finished = _utc(now)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown run")
            if finished < _utc(row["started_at"]):
                raise ValueError("A run cannot finish before it starts")
            if row["status"] != "running":
                return False
            duration = (_number(duration_ms, "duration_ms") if duration_ms is not None else
                        (finished - _utc(row["started_at"])).total_seconds() * 1000)
            if duration < 0:
                raise ValueError("duration_ms must be non-negative")
            db.execute("UPDATE runs SET finished_at=?,status=?,detail=?,duration_ms=? "
                       "WHERE id=? AND status='running'", (_iso(finished), status,
                        _json(detail) if detail is not None else None, duration, run_id))
        return True

    def recent_runs(self, limit=20):
        limit = int(limit)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        with self._connection() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item["detail"]) if item["detail"] is not None else None
            result.append(item)
        return result

    def try_acquire_lock(self, owner, ttl_seconds=1800, now=None):
        owner = str(owner).strip()
        ttl = _number(ttl_seconds, "ttl_seconds", positive=True)
        if not owner:
            raise ValueError("owner must not be empty")
        current = _utc(now)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner,expires_at FROM worker_lock WHERE id=1").fetchone()
            if row and row["owner"] != owner and row["expires_at"] > _iso(current):
                return False
            db.execute("INSERT INTO worker_lock(id,owner,expires_at) VALUES(1,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at",
                       (owner, _iso(current + timedelta(seconds=ttl))))
        return True

    def release_lock(self, owner):
        with self._connection() as db:
            cursor = db.execute("DELETE FROM worker_lock WHERE id=1 AND owner=?", (str(owner),))
        return bool(cursor.rowcount)
