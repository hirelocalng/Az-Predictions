"""
grant_premium.py — manually grant Premium to a user by email.

Emergency remedy for customers who were debited by Korapay but never got
marked premium (e.g. their browser never made it back to /subscribe, or a
webhook was missed before the fix in app.py deployed). Extends from the
later of "now" or the user's current premium_until, so it's safe to run
even if the user already has some active time left. Writes an audit row
into `payments` (source='manual') so the grant shows up in the same trail
as real charges.

Usage:
    set DATABASE_URL=postgresql://...
    python grant_premium.py customer@example.com monthly
    python grant_premium.py customer@example.com 3month --reference azpred_123_abc --note "debited, webhook missed"
    python grant_premium.py customer@example.com --days 14 --note "goodwill credit"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

DATABASE_URL = os.environ.get('DATABASE_URL', '')
if not DATABASE_URL:
    print('ERROR: set DATABASE_URL first, e.g.:')
    print('  set DATABASE_URL=postgresql://user:pass@host:5432/dbname')
    print('  python grant_premium.py customer@example.com monthly')
    sys.exit(1)

import psycopg  # noqa: E402

_PLAN_DAYS = {'1week': 7, 'monthly': 30, '3month': 90}


def main():
    ap = argparse.ArgumentParser(description='Manually grant Premium access by email.')
    ap.add_argument('email')
    ap.add_argument('plan', nargs='?', choices=sorted(_PLAN_DAYS),
                     help='1week | monthly | 3month (omit and use --days instead)')
    ap.add_argument('--days', type=int, help='Grant an explicit number of days instead of a plan')
    ap.add_argument('--reference', default=None,
                     help='Korapay reference this grant corresponds to, if known')
    ap.add_argument('--note', default='', help='Free-text reason, stored for audit')
    args = ap.parse_args()

    if not args.plan and not args.days:
        ap.error('pass a plan (1week|monthly|3month) or --days N')
    days = args.days if args.days else _PLAN_DAYS[args.plan]
    plan_label = args.plan or f'{days}days-manual'

    url = DATABASE_URL
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    if 'sslmode' not in url:
        url += ('&' if '?' in url else '?') + 'sslmode=require'

    conn = psycopg.connect(url, autocommit=False)
    cur = conn.cursor()
    try:
        cur.execute(
            'SELECT id, name, is_premium, premium_until FROM users WHERE email=%s',
            (args.email,)
        )
        row = cur.fetchone()
        if row is None:
            print(f'ERROR: no user found with email {args.email!r}')
            sys.exit(1)
        user_id, name, was_premium, current_until = row

        now_u   = datetime.now(timezone.utc)
        base    = current_until if (current_until and current_until > now_u) else now_u
        expires = base + timedelta(days=days)

        cur.execute(
            'UPDATE users SET is_premium=TRUE, premium_until=%s WHERE id=%s',
            (expires, user_id)
        )

        reference = args.reference or f"manual_{user_id}_{int(now_u.timestamp())}"
        cur.execute(
            "INSERT INTO payments "
            "(reference, user_id, plan, amount, currency, status, days_granted, "
            " premium_until, source, raw_payload, processed_at) "
            "VALUES (%s,%s,%s,NULL,'NGN','success',%s,%s,'manual',%s::jsonb,NOW()) "
            "ON CONFLICT (reference) DO UPDATE SET "
            "  status='success', days_granted=EXCLUDED.days_granted, "
            "  premium_until=EXCLUDED.premium_until, source='manual', "
            "  raw_payload=EXCLUDED.raw_payload, processed_at=NOW()",
            (reference, user_id, plan_label, days, expires,
             json.dumps({'note': args.note} if args.note else {}))
        )
        conn.commit()

        print(f'Granted: {args.email} (id={user_id}, {name})')
        print(f'  was_premium={was_premium}  previous_until={current_until}')
        print(f'  now_premium=True  new_until={expires.isoformat()}  (+{days}d)')
        print(f'  audit reference: {reference}')
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    main()
