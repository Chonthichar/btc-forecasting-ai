#!/usr/bin/env python3
"""
Three-agent BTC sentiment data pipeline.

Agent 1: Collector
    Collects BTC news/sentiment evidence from three sources only:
    primary Bitcoin news dataset, Alpha Vantage NEWS_SENTIMENT, and GDELT.
Agent 2: Validator
    Checks date range, BTC relevance, duplicates, source quality, URL validity,
    and re-scores sentiment with FinBERT when available.
Agent 3: Manager
    Resolves conflicting sentiment signals and creates final article-level,
    hourly, and daily training datasets.

Default target window:
    one year ending on the run date.
For this project:
    2025-09-04 through 2026-09-04.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests


PRIMARY_TSV_URL = (
    "https://github.com/mouadja02/bitcoin-news-data/"
    "raw/refs/heads/main/bitcoin-news.tsv"
)
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
ALPHA_ENDPOINT = "https://www.alphavantage.co/query"

BTC_PATTERN = re.compile(
    r"(?i)(\bbitcoin\b|\bBTC\b|\bBTC[-_/ ]?USD\b|\bXBT\b|satoshi|bitcoin ETF)"
)

POS_WORDS = {
    "gain","gains","gained","rise","rises","rising","surge","surges","surged",
    "rally","rallies","bull","bullish","breakout","record","high","approve",
    "approved","approval","adoption","adopt","buy","buying","inflow","inflows",
    "profit","profits","optimism","positive","growth","recover","recovery",
    "strong","strength","upgrade","upside","soar","soars","soared"
}
NEG_WORDS = {
    "fall","falls","fell","drop","drops","dropped","decline","declines",
    "bear","bearish","crash","selloff","sell-off","outflow","outflows",
    "ban","banned","hack","hacked","fraud","scam","lawsuit","risk","risks",
    "negative","loss","losses","plunge","plunges","plunged","liquidation",
    "liquidations","fear","weak","weakness","downgrade","downside"
}


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def parse_date(value: str, end_of_day: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if len(value) <= 10:
        ts = ts.normalize()
        if end_of_day:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def safe_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def norm_title(s: str) -> str:
    s = safe_text(s).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_hash(s: str) -> str:
    return hashlib.sha1(norm_title(s).encode("utf-8")).hexdigest()


def valid_http_url(url: str) -> bool:
    try:
        p = urlparse(safe_text(url))
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def sentiment_label(score: float, neutral_band: float = 0.15) -> str:
    if pd.isna(score):
        return "unknown"
    if score > neutral_band:
        return "positive"
    if score < -neutral_band:
        return "negative"
    return "neutral"


def lexicon_sentiment(text: str) -> Tuple[float, float, str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", safe_text(text).lower())
    if not tokens:
        return 0.0, 0.0, "neutral"
    pos = sum(t in POS_WORDS for t in tokens)
    neg = sum(t in NEG_WORDS for t in tokens)
    score = (pos - neg) / max(pos + neg, 1)
    confidence = min(0.80, 0.35 + 0.08 * (pos + neg)) if (pos + neg) else 0.25
    return float(score), float(confidence), sentiment_label(score)


def standard_frame(df: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    cols = [
        "timestamp","title","summary","source","url","api_source",
        "source_dataset","upstream_sentiment_score","upstream_sentiment_label"
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(index=df.index)
    lookup = {str(c).lower(): c for c in df.columns}

    def pick(*names: str, default: Any = "") -> pd.Series:
        for n in names:
            if n.lower() in lookup:
                return df[lookup[n.lower()]]
        return pd.Series([default] * len(df), index=df.index)

    out["timestamp"] = pick(
        "timestamp","datetime","date_time","published_at","time_published",
        "published","date","seendate","publishedat"
    )
    out["title"] = pick("title","headline","name")
    out["summary"] = pick("summary","description","excerpt","content","body")
    out["source"] = pick("source","domain","publisher","source_name")
    out["url"] = pick("url","link","article_url")
    out["api_source"] = pick("api_source","provider","source_type", default=source_dataset)
    out["source_dataset"] = source_dataset
    out["upstream_sentiment_score"] = pd.to_numeric(
        pick(
            "ticker_sentiment_score","sentiment_score","finbert_score",
            "score","overall_sentiment_score", default=np.nan
        ),
        errors="coerce"
    )
    out["upstream_sentiment_label"] = pick(
        "ticker_sentiment_label","sentiment_label","finbert_label",
        "label","overall_sentiment_label", default=""
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for c in ["title","summary","source","url","api_source","source_dataset","upstream_sentiment_label"]:
        out[c] = out[c].map(safe_text)
    return out[cols]


@dataclass
class CollectorAgent:
    start: pd.Timestamp
    end: pd.Timestamp
    session: requests.Session
    alpha_key: Optional[str] = None
    gdelt_max_days: Optional[int] = None

    def _headers(self) -> Dict[str, str]:
        h = {"User-Agent": "btc-sentiment-research-pipeline/1.0"}
        return h

    def fetch_primary_bitcoin_news(self) -> pd.DataFrame:
        print("[Agent 1] Fetching primary continuously-updated Bitcoin news dataset...")
        r = self.session.get(PRIMARY_TSV_URL, headers=self._headers(), timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), sep="\t", low_memory=False)
        out = standard_frame(df, "mouadja02/bitcoin-news-data")
        out = out[out["timestamp"].between(self.start, self.end, inclusive="both")]
        print(f"  primary rows in date window: {len(out):,}")
        return out

    def fetch_alpha_vantage(self) -> pd.DataFrame:
        if not self.alpha_key:
            print("[Agent 1] Alpha Vantage skipped (no ALPHA_VANTAGE_API_KEY).")
            return standard_frame(pd.DataFrame(), "alphavantage")

        print("[Agent 1] Fetching Alpha Vantage NEWS_SENTIMENT with adaptive 7-day chunks...")
        frames = []
        min_chunk = pd.Timedelta(hours=6)
        initial_chunk = pd.Timedelta(days=7)
        near_limit = 950

        def fetch_window(window_start: pd.Timestamp, window_end: pd.Timestamp, depth: int = 0):
            """Fetch one window; recursively split if the API result is close to its 1000-row cap."""
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": "CRYPTO:BTC",
                "time_from": window_start.strftime("%Y%m%dT%H%M"),
                "time_to": window_end.strftime("%Y%m%dT%H%M"),
                "sort": "EARLIEST",
                "limit": 1000,
                "apikey": self.alpha_key,
            }

            r = self.session.get(ALPHA_ENDPOINT, params=params, timeout=60)
            data = r.json()

            if isinstance(data, dict) and not data.get("feed"):
                msg = (
                    data.get("Information")
                    or data.get("Note")
                    or data.get("Error Message")
                    or ""
                )
                if msg:
                    raise RuntimeError(f"Alpha Vantage returned no feed: {msg}")

            feed = data.get("feed", []) if isinstance(data, dict) else []

            # If close to the provider cap, do not accept a potentially truncated window.
            if len(feed) >= near_limit and (window_end - window_start) > min_chunk:
                midpoint = window_start + (window_end - window_start) / 2
                midpoint = midpoint.floor("min")
                left_end = midpoint
                right_start = midpoint + pd.Timedelta(minutes=1)
                print(
                    f"  split {window_start} -> {window_end}: "
                    f"{len(feed):,} rows near API cap"
                )
                fetch_window(window_start, left_end, depth + 1)
                fetch_window(right_start, window_end, depth + 1)
                return

            rows = []
            for a in feed:
                btc_info = None
                for t in a.get("ticker_sentiment", []) or []:
                    if t.get("ticker") == "CRYPTO:BTC":
                        btc_info = t
                        break
                rows.append({
                    "time_published": a.get("time_published"),
                    "title": a.get("title"),
                    "summary": a.get("summary"),
                    "source": a.get("source"),
                    "url": a.get("url"),
                    "api_source": "alphavantage",
                    "ticker_sentiment_score": (
                        btc_info.get("ticker_sentiment_score") if btc_info else None
                    ),
                    "ticker_sentiment_label": (
                        btc_info.get("ticker_sentiment_label") if btc_info else None
                    ),
                })

            frames.append(standard_frame(pd.DataFrame(rows), "alphavantage"))
            indent = "  " + "  " * depth
            print(
                f"{indent}{window_start.strftime('%Y-%m-%d %H:%M')} -> "
                f"{window_end.strftime('%Y-%m-%d %H:%M')}: {len(rows):,}"
            )
            time.sleep(0.9)

        cursor = self.start.normalize()
        while cursor <= self.end:
            next_cursor = min(
                cursor + initial_chunk - pd.Timedelta(minutes=1),
                self.end,
            )
            try:
                fetch_window(cursor, next_cursor)
            except Exception as e:
                print(
                    f"  Alpha Vantage window failed "
                    f"{cursor} -> {next_cursor}: {e}",
                    file=sys.stderr,
                )
            cursor = next_cursor + pd.Timedelta(minutes=1)

        if not frames:
            return standard_frame(pd.DataFrame(), "alphavantage")

        out = pd.concat(frames, ignore_index=True)
        out = out[out["timestamp"].between(self.start, self.end, inclusive="both")]
        out = out.sort_values("timestamp").drop_duplicates(
            subset=["url", "title", "timestamp"], keep="first"
        )
        print(f"  Alpha Vantage rows in date window: {len(out):,}")
        return out.reset_index(drop=True)


    def fetch_gdelt(self) -> pd.DataFrame:
        print("[Agent 1] Fetching GDELT BTC news as additional evidence...")
        # GDELT ArticleList caps each query at 250. Daily windows reduce truncation.
        frames = []
        day = self.start.normalize()
        days_done = 0
        while day <= self.end.normalize():
            if self.gdelt_max_days and days_done >= self.gdelt_max_days:
                break
            nxt = min(day + pd.Timedelta(days=1), self.end)
            params = {
                "query": '(bitcoin OR "BTC") sourcelang:english',
                "mode": "artlist",
                "maxrecords": 250,
                "format": "json",
                "startdatetime": day.strftime("%Y%m%d%H%M%S"),
                "enddatetime": nxt.strftime("%Y%m%d%H%M%S"),
                "sort": "DateAsc",
            }
            try:
                r = self.session.get(GDELT_ENDPOINT, params=params, timeout=60)
                content_type = r.headers.get("content-type", "")
                if r.status_code == 200 and "json" in content_type.lower():
                    data = r.json()
                    rows = []
                    for a in data.get("articles", []) or []:
                        rows.append({
                            "seendate": a.get("seendate"),
                            "title": a.get("title"),
                            "summary": "",
                            "domain": a.get("domain"),
                            "url": a.get("url"),
                            "api_source": "gdelt",
                        })
                    if rows:
                        frames.append(standard_frame(pd.DataFrame(rows), "gdelt"))
                elif days_done < 3:
                    print(
                        "  note: GDELT may restrict older full-text windows; "
                        "the primary dataset remains the main historical source."
                    )
            except Exception as e:
                if days_done < 3:
                    print(f"  GDELT request failed: {e}", file=sys.stderr)
            days_done += 1
            day += pd.Timedelta(days=1)
            time.sleep(0.22)

        if not frames:
            return standard_frame(pd.DataFrame(), "gdelt")
        out = pd.concat(frames, ignore_index=True)
        out = out[out["timestamp"].between(self.start, self.end, inclusive="both")]
        print(f"  GDELT rows in date window: {len(out):,}")
        return out

    def collect_all(self, use_gdelt: bool = True) -> pd.DataFrame:
        frames = []
        for fn in [
            self.fetch_primary_bitcoin_news,
            self.fetch_alpha_vantage,
        ]:
            try:
                x = fn()
                if not x.empty:
                    frames.append(x)
            except Exception as e:
                print(f"[Agent 1] Source failed but pipeline continues: {e}", file=sys.stderr)
        if use_gdelt:
            try:
                x = self.fetch_gdelt()
                if not x.empty:
                    frames.append(x)
            except Exception as e:
                print(f"[Agent 1] GDELT failed but pipeline continues: {e}", file=sys.stderr)

        if not frames:
            raise RuntimeError("Agent 1 could not collect any rows from any source.")
        raw = pd.concat(frames, ignore_index=True)
        raw["collected_at"] = utc_now()
        return raw


@dataclass
class ValidatorAgent:
    start: pd.Timestamp
    end: pd.Timestamp
    finbert_model: str = "ProsusAI/finbert"

    def _score_with_finbert(self, texts: List[str]) -> Optional[pd.DataFrame]:
        try:
            from transformers import pipeline
            import torch
        except Exception:
            return None

        try:
            device = 0 if torch.cuda.is_available() else -1
            clf = pipeline(
                "text-classification",
                model=self.finbert_model,
                tokenizer=self.finbert_model,
                device=device,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            results = clf(texts, batch_size=32)
            rows = []
            for result in results:
                probs = {safe_text(x["label"]).lower(): float(x["score"]) for x in result}
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                neu = probs.get("neutral", 0.0)
                score = pos - neg
                label = max(probs, key=probs.get) if probs else "neutral"
                rows.append({
                    "finbert_score": score,
                    "finbert_confidence": max(pos, neg, neu),
                    "finbert_label": label,
                    "finbert_pos_prob": pos,
                    "finbert_neg_prob": neg,
                    "finbert_neu_prob": neu,
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[Agent 2] FinBERT unavailable; using lexical fallback: {e}", file=sys.stderr)
            return None

    def validate(self, raw: pd.DataFrame) -> pd.DataFrame:
        print("[Agent 2] Validating, de-duplicating and independently scoring sentiment...")
        df = raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["in_date_range"] = df["timestamp"].between(self.start, self.end, inclusive="both")
        combined = (
            df["title"].fillna("").astype(str) + " " +
            df["summary"].fillna("").astype(str)
        )
        df["btc_relevant"] = combined.str.contains(BTC_PATTERN, regex=True, na=False)
        df["url_valid"] = df["url"].map(valid_http_url)
        df["title_length"] = df["title"].map(lambda x: len(safe_text(x)))
        df["title_hash"] = df["title"].map(title_hash)
        df["source_present"] = df["source"].map(lambda x: bool(safe_text(x)))

        # Deduplicate first by URL, then by normalized headline within the same UTC day.
        date_key = df["timestamp"].dt.strftime("%Y-%m-%d").fillna("unknown")
        url_key = df["url"].where(df["url_valid"], "")
        df["dup_url"] = url_key.ne("") & url_key.duplicated(keep="first")
        pair_key = df["title_hash"] + "|" + date_key
        df["dup_title_day"] = pair_key.duplicated(keep="first")
        df["is_duplicate"] = df["dup_url"] | df["dup_title_day"]

        score_texts = []
        for t, s in zip(df["title"], df["summary"]):
            txt = safe_text(t)
            if safe_text(s):
                txt = f"{txt}. {safe_text(s)}"
            score_texts.append(txt[:4000])

        fb = self._score_with_finbert(score_texts)
        if fb is not None and len(fb) == len(df):
            for c in fb.columns:
                df[c] = fb[c].values
            df["sentiment_method"] = "finbert"
        else:
            scored = [lexicon_sentiment(t) for t in score_texts]
            df["finbert_score"] = [x[0] for x in scored]
            df["finbert_confidence"] = [x[1] for x in scored]
            df["finbert_label"] = [x[2] for x in scored]
            df["finbert_pos_prob"] = np.nan
            df["finbert_neg_prob"] = np.nan
            df["finbert_neu_prob"] = np.nan
            df["sentiment_method"] = "lexicon_fallback"

        # Simple auditable quality score.
        df["quality_score"] = (
            0.28 * df["btc_relevant"].astype(float)
            + 0.18 * df["in_date_range"].astype(float)
            + 0.12 * (~df["is_duplicate"]).astype(float)
            + 0.10 * df["url_valid"].astype(float)
            + 0.08 * df["source_present"].astype(float)
            + 0.10 * df["title_length"].between(15, 260).astype(float)
            + 0.14 * df["finbert_confidence"].fillna(0.0).clip(0, 1)
        ).clip(0, 1)

        df["validation_status"] = np.where(
            df["in_date_range"]
            & df["btc_relevant"]
            & (~df["is_duplicate"])
            & (df["title_length"] >= 8)
            & (df["quality_score"] >= 0.58),
            "accepted",
            "rejected",
        )

        def reason(r: pd.Series) -> str:
            reasons = []
            if not r["in_date_range"]:
                reasons.append("outside_date_range")
            if not r["btc_relevant"]:
                reasons.append("weak_btc_relevance")
            if r["is_duplicate"]:
                reasons.append("duplicate")
            if r["title_length"] < 8:
                reasons.append("title_too_short")
            if r["quality_score"] < 0.58:
                reasons.append("quality_below_threshold")
            return "|".join(reasons) or "passed"

        df["validation_reason"] = df.apply(reason, axis=1)
        return df


@dataclass
class ManagerAgent:
    neutral_band: float = 0.15

    def decide(self, audited: pd.DataFrame) -> pd.DataFrame:
        print("[Agent 3] Resolving sentiment conflicts and producing final training rows...")
        df = audited[audited["validation_status"] == "accepted"].copy()

        up = pd.to_numeric(df["upstream_sentiment_score"], errors="coerce").clip(-1, 1)
        fb = pd.to_numeric(df["finbert_score"], errors="coerce").fillna(0).clip(-1, 1)

        has_up = up.notna()
        # FinBERT is the primary signal; upstream sentiment acts as corroboration.
        final = np.where(has_up, 0.72 * fb + 0.28 * up.fillna(0), fb)
        df["manager_sentiment_score"] = pd.Series(final, index=df.index).clip(-1, 1)
        df["manager_sentiment_label"] = df["manager_sentiment_score"].map(
            lambda x: sentiment_label(float(x), self.neutral_band)
        )

        up_sign = np.sign(up.fillna(0))
        fb_sign = np.sign(fb)
        df["sentiment_conflict"] = (
            has_up
            & (np.abs(up.fillna(0)) >= 0.20)
            & (np.abs(fb) >= 0.20)
            & (up_sign != fb_sign)
        )

        df["manager_confidence"] = (
            0.58 * df["finbert_confidence"].fillna(0.25).clip(0, 1)
            + 0.42 * df["quality_score"].fillna(0).clip(0, 1)
        )
        df.loc[df["sentiment_conflict"], "manager_confidence"] *= 0.70
        df["manager_confidence"] = df["manager_confidence"].clip(0, 1)

        df["manager_decision"] = np.where(
            df["sentiment_conflict"], "accepted_with_conflict", "accepted"
        )
        df["manager_reason"] = np.where(
            df["sentiment_conflict"],
            "FinBERT and upstream sentiment disagree; FinBERT weighted more heavily.",
            "Validation passed; sentiment signals are compatible or no upstream score exists.",
        )
        df["training_weight"] = (
            df["quality_score"] * df["manager_confidence"]
        ).clip(0, 1)

        keep = [
            "timestamp","title","summary","source","url","api_source","source_dataset",
            "upstream_sentiment_score","upstream_sentiment_label",
            "finbert_score","finbert_label","finbert_confidence",
            "manager_sentiment_score","manager_sentiment_label","manager_confidence",
            "training_weight","quality_score","sentiment_conflict","manager_decision",
            "manager_reason","sentiment_method","collected_at"
        ]
        return df[keep].sort_values("timestamp").reset_index(drop=True)

    def aggregate(self, final: pd.DataFrame, freq: str) -> pd.DataFrame:
        if final.empty:
            return pd.DataFrame()

        x = final.copy()
        x["bucket"] = x["timestamp"].dt.floor(freq)
        x["w"] = x["training_weight"].clip(lower=0.05)
        x["weighted_score"] = x["manager_sentiment_score"] * x["w"]
        x["is_positive"] = (x["manager_sentiment_label"] == "positive").astype(int)
        x["is_negative"] = (x["manager_sentiment_label"] == "negative").astype(int)
        x["is_neutral"] = (x["manager_sentiment_label"] == "neutral").astype(int)

        agg = x.groupby("bucket", as_index=False).agg(
            article_count=("title","size"),
            weight_sum=("w","sum"),
            weighted_score_sum=("weighted_score","sum"),
            mean_confidence=("manager_confidence","mean"),
            mean_quality=("quality_score","mean"),
            positive_count=("is_positive","sum"),
            negative_count=("is_negative","sum"),
            neutral_count=("is_neutral","sum"),
            unique_sources=("source","nunique"),
        )
        agg["sentiment_score"] = agg["weighted_score_sum"] / agg["weight_sum"].replace(0, np.nan)
        agg["sentiment_label"] = agg["sentiment_score"].map(
            lambda z: sentiment_label(float(z), self.neutral_band) if pd.notna(z) else "unknown"
        )
        agg["positive_share"] = agg["positive_count"] / agg["article_count"]
        agg["negative_share"] = agg["negative_count"] / agg["article_count"]
        agg["neutral_share"] = agg["neutral_count"] / agg["article_count"]
        agg = agg.drop(columns=["weighted_score_sum"]).rename(columns={"bucket": "timestamp"})
        return agg.sort_values("timestamp").reset_index(drop=True)


def save_outputs(
    outdir: Path,
    raw: pd.DataFrame,
    audit: pd.DataFrame,
    final: pd.DataFrame,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    manifest: Dict[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "01_raw_collected": raw,
        "02_validation_audit": audit,
        "03_final_btc_sentiment_articles": final,
        "04_btc_sentiment_hourly": hourly,
        "05_btc_sentiment_daily": daily,
    }

    for name, df in datasets.items():
        csv_path = outdir / f"{name}.csv"
        pq_path = outdir / f"{name}.parquet"
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(pq_path, index=False)
        except Exception as e:
            print(f"Parquet write skipped for {name}: {e}", file=sys.stderr)

    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved outputs to: {outdir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2025-09-04")
    parser.add_argument("--end-date", default="2026-09-04")
    parser.add_argument("--output-dir", default="./btc_sentiment_output")
    parser.add_argument("--skip-gdelt", action="store_true")
    parser.add_argument(
        "--gdelt-max-days",
        type=int,
        default=None,
        help="Optional safety cap while testing. Omit for full date range.",
    )
    parser.add_argument("--neutral-band", type=float, default=0.15)
    args = parser.parse_args()

    start = parse_date(args.start_date, end_of_day=False)
    end = parse_date(args.end_date, end_of_day=True)

    print("BTC sentiment collection window:")
    print(f"  start: {start}")
    print(f"  end:   {end}")

    s = requests.Session()
    collector = CollectorAgent(
        start=start,
        end=end,
        session=s,
        alpha_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        gdelt_max_days=args.gdelt_max_days,
    )
    validator = ValidatorAgent(start=start, end=end)
    manager = ManagerAgent(neutral_band=args.neutral_band)

    raw = collector.collect_all(use_gdelt=not args.skip_gdelt)
    audit = validator.validate(raw)
    final = manager.decide(audit)
    hourly = manager.aggregate(final, "h")
    daily = manager.aggregate(final, "D")

    manifest = {
        "created_at_utc": str(utc_now()),
        "start_utc": str(start),
        "end_utc": str(end),
        "asset": "BTC",
        "agents": {
            "agent_1": "collector",
            "agent_2": "validator_and_independent_sentiment_scorer",
            "agent_3": "manager_and_final_arbiter",
        },
        "rows": {
            "raw_collected": int(len(raw)),
            "accepted_final": int(len(final)),
            "rejected": int((audit["validation_status"] == "rejected").sum()),
            "hourly_rows": int(len(hourly)),
            "daily_rows": int(len(daily)),
        },
        "sources_attempted": [
            "mouadja02/bitcoin-news-data (Guardian + Finnhub + AlphaVantage, continuously updated)",
            "Alpha Vantage NEWS_SENTIMENT (optional API key)",
            "GDELT DOC API (additional evidence; historical availability may vary)",
        ],
        "sentiment_model": "ProsusAI/finbert if available; auditable lexicon fallback otherwise",
        "notes": [
            "Keep article-level data for auditability.",
            "For hourly BTC forecasting, prefer 04_btc_sentiment_hourly.parquet.",
            "Avoid look-ahead leakage: join sentiment bucket t only to price targets strictly after t.",
        ],
    }

    save_outputs(Path(args.output_dir), raw, audit, final, hourly, daily, manifest)


if __name__ == "__main__":
    main()
