"""Additive SQLite storage for scans, research references, caches and leases."""
from __future__ import annotations
import json
import uuid
from datetime import timedelta
from app.monitoring_store import _utc, _iso, _json


class EventStore:
    def __init__(self, store):
        self.store = store
        with store._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS monitoring_scans (
                    id INTEGER PRIMARY KEY, run_id INTEGER UNIQUE NOT NULL,
                    scanned_at TEXT NOT NULL, market_timestamp TEXT, payload TEXT NOT NULL,
                    trigger_result TEXT NOT NULL, research_event_id TEXT
                );
                CREATE TABLE IF NOT EXISTS research_events (
                    id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, market_timestamp TEXT,
                    created_at TEXT NOT NULL, status TEXT NOT NULL, reasons TEXT NOT NULL,
                    result TEXT, summary TEXT, llm_status TEXT, completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS research_fingerprint ON research_events(fingerprint,created_at);
                CREATE TABLE IF NOT EXISTS research_query_cache (
                    key TEXT PRIMARY KEY, expires_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operation_leases (
                    name TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at TEXT NOT NULL
                );
            """)

    def acquire(self, name, owner, ttl=120, now=None):
        now = _utc(now)
        with self.store._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM operation_leases WHERE name=?", (name,)).fetchone()
            if row and row["owner"] != owner and row["expires_at"] > _iso(now):
                return False
            db.execute("INSERT INTO operation_leases VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,expires_at=excluded.expires_at",
                       (name, owner, _iso(now + timedelta(seconds=ttl))))
        return True

    def owns(self, name, owner, now=None):
        with self.store._connection() as db:
            row = db.execute("SELECT 1 FROM operation_leases WHERE name=? AND owner=? AND expires_at>?", (name, owner, _iso(now))).fetchone()
        return bool(row)

    def release(self, name, owner):
        with self.store._connection() as db:
            db.execute("DELETE FROM operation_leases WHERE name=? AND owner=?", (name, owner))

    def scan(self, run_id, payload, trigger, now=None):
        with self.store._connection() as db:
            db.execute("INSERT INTO monitoring_scans(run_id,scanned_at,market_timestamp,payload,trigger_result) VALUES(?,?,?,?,?) ON CONFLICT(run_id) DO NOTHING",
                       (run_id, _iso(now), payload.get("market_timestamp"), _json(payload), _json(trigger)))
            return db.execute("SELECT id FROM monitoring_scans WHERE run_id=?", (run_id,)).fetchone()[0]

    def scans(self, limit=50):
        with self.store._connection() as db:
            rows = db.execute("SELECT * FROM monitoring_scans ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 200)),)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"]), "trigger_result": json.loads(row["trigger_result"])} for row in rows]

    def link(self, scan_id, event_id):
        with self.store._connection() as db:
            db.execute("UPDATE monitoring_scans SET research_event_id=? WHERE id=?", (event_id, scan_id))

    @staticmethod
    def event(row):
        if not row:
            return None
        result = dict(row)
        for field in ("reasons", "result"):
            result[field] = json.loads(result[field]) if result[field] else None
        return result

    def claim(self, fingerprint, market_timestamp, reasons, cooldown_minutes, now=None):
        now = _utc(now)
        with self.store._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM research_events WHERE fingerprint=? ORDER BY created_at DESC LIMIT 1", (fingerprint,)).fetchone()
            if row and (row["market_timestamp"] == market_timestamp or row["created_at"] > _iso(now - timedelta(minutes=cooldown_minutes))):
                return self.event(row), False
            event_id = uuid.uuid4().hex
            db.execute("INSERT INTO research_events(id,fingerprint,market_timestamp,created_at,status,reasons) VALUES(?,?,?,?,?,?)",
                       (event_id, fingerprint, market_timestamp, _iso(now), "running", _json(reasons)))
            row = db.execute("SELECT * FROM research_events WHERE id=?", (event_id,)).fetchone()
        return self.event(row), True

    def finish(self, event_id, status, result=None, summary=None, llm_status=None, now=None):
        with self.store._connection() as db:
            db.execute("UPDATE research_events SET status=?,result=?,summary=?,llm_status=?,completed_at=? WHERE id=?",
                       (status, _json(result) if result is not None else None, summary, llm_status, _iso(now), event_id))

    def latest_event(self, with_evidence=False):
        with self.store._connection() as db:
            row = db.execute("SELECT * FROM research_events WHERE " + ("result IS NOT NULL AND " if with_evidence else "") +
                "COALESCE(json_extract(result,'$.workflow.mode'),'live') != 'historical' ORDER BY created_at DESC LIMIT 1").fetchone()
        return self.event(row)

    def cache_get(self, key, now=None):
        with self.store._connection() as db:
            row = db.execute("SELECT payload FROM research_query_cache WHERE key=? AND expires_at>?", (key, _iso(now))).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key, payload, seconds, now=None):
        now = _utc(now)
        with self.store._connection() as db:
            db.execute("DELETE FROM research_query_cache WHERE expires_at<=?", (_iso(now),))
            db.execute("INSERT INTO research_query_cache VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET expires_at=excluded.expires_at,payload=excluded.payload",
                       (key, _iso(now + timedelta(seconds=seconds)), _json(payload)))
            db.execute("DELETE FROM research_query_cache WHERE key NOT IN (SELECT key FROM research_query_cache ORDER BY expires_at DESC LIMIT 256)")
