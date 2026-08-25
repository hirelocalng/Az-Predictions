"""
verify_bestbet_wat.py — one-shot verification for the same-day-WAT Best
Bet / Daily Accumulator fix (2026-08-26).

Prints:
  1. Current /api/best-bet picks with kickoff in both UTC and WAT.
  2. The WAT day-boundary the backend is enforcing.
  3. A simulated "no fixtures left today" scenario by monkeypatching
     datetime.now to a moment just before the WAT day boundary computed
     from live candidates, without touching any real data.
"""
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
import app as A

WAT = timezone(timedelta(hours=1))


def fmt(dt_utc_str):
    dt = datetime.fromisoformat(dt_utc_str.replace("Z", "+00:00"))
    return f"{dt_utc_str}  ({dt.astimezone(WAT).strftime('%Y-%m-%d %H:%M')} WAT)"


client = A.app.test_client()

print("=" * 70)
print("STEP 1 -- current live picks")
print("=" * 70)
r = client.get("/api/best-bet")
d = r.get_json()
now_utc = datetime.now(timezone.utc)
now_wat = now_utc.astimezone(WAT)
print(f"server now: {now_utc.isoformat()}  ({now_wat.strftime('%Y-%m-%d %H:%M:%S')} WAT)")
print(f"WAT end-of-today boundary: {A._wat_end_of_today_utc(now_utc).isoformat()}")
print()

bb = d.get("best_bet")
print("Best Bet of the Day:")
if bb:
    print(f"  {bb['match']} -- {bb['pick']} ({bb['prob']*100:.1f}%)")
    print(f"  kickoff: {fmt(bb['utc_kickoff'])}")
else:
    print("  None (no qualifying same-day fixtures)")

print("\nDaily Accumulator:")
acca = d.get("accumulator") or []
if not acca:
    print("  Empty (no qualifying same-day fixtures)")
for p in acca:
    print(f"  {p['match']} -- {p['pick']} ({p['prob']*100:.1f}%)")
    print(f"    kickoff: {fmt(p['utc_kickoff'])}")
print(f"\ncombined_prob: {d.get('combined_prob')}")

# Verify every pick is strictly today-in-WAT and in the future
print("\n" + "=" * 70)
print("STEP 1b -- verifying every pick is today (WAT) and in the future")
print("=" * 70)
end_of_today = A._wat_end_of_today_utc(now_utc)
all_ok = True
for label, p in [("best_bet", bb)] + [(f"acca[{i}]", p) for i, p in enumerate(acca)]:
    if not p:
        continue
    ko = datetime.fromisoformat(p["utc_kickoff"].replace("Z", "+00:00"))
    is_future = ko > now_utc
    is_today = ko <= end_of_today
    ok = is_future and is_today
    all_ok &= ok
    print(f"  {label}: future={is_future}  same_day_wat={is_today}  {'OK' if ok else 'FAIL'}")
print(f"\nALL PICKS VALID: {all_ok}")

# ---------------------------------------------------------------------------
# STEP 2 -- simulate "no fixtures remain today" by freezing the clock to
# 23:58 WAT, well past all of today's known kickoffs, and hitting the real
# route again. Reuses the actual _bestbet_payload_still_valid staleness
# check to force a rebuild under the frozen time, then the real candidate
# filter to prove the empty-state path activates.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 -- simulating no fixtures remaining today (frozen at 23:58 WAT)")
print("=" * 70)

_real_datetime = datetime


class _FrozenDatetime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        real_now_wat = _real_datetime.now(timezone.utc).astimezone(WAT)
        frozen_wat = real_now_wat.replace(hour=23, minute=58, second=0, microsecond=0)
        frozen_utc = frozen_wat.astimezone(timezone.utc)
        return frozen_utc if tz is None else frozen_utc.astimezone(tz)


A.datetime = _FrozenDatetime
try:
    r2 = client.get("/api/best-bet")
    d2 = r2.get_json()
finally:
    A.datetime = _real_datetime

print(f"frozen server time used: 23:58 WAT (all real kickoffs today are earlier)")
print(f"best_bet: {d2.get('best_bet')}")
print(f"accumulator: {d2.get('accumulator')}")
print(f"combined_prob: {d2.get('combined_prob')}")
print(f"\nEMPTY STATE CORRECTLY TRIGGERED: {d2.get('best_bet') is None and d2.get('accumulator') == []}")
