"""UTC cadence, aligned two minutes after closed hourly candle boundaries."""
from datetime import datetime, timedelta, timezone
import math


def next_cycle(now, interval_minutes=60):
    anchor = datetime(1970, 1, 1, 0, 2, tzinfo=timezone.utc)
    seconds = float(interval_minutes) * 60
    slot = math.floor((now - anchor).total_seconds() / seconds) + 1
    return anchor + timedelta(seconds=slot * seconds)
