"""
End-to-end test for the Korapay webhook path (app.py `/api/payment/webhook`).

Covers the bug fixed in this change: premium used to only be granted when the
customer's browser round-tripped back through /subscribe and called
/api/payment/verify. A charge could succeed at Korapay with nobody ever
telling this app. The webhook is now the source of truth, and both it and
/verify share one idempotent grant function (_process_successful_charge) so
neither can double-extend a subscription.

No real Postgres needed: `app._db` is monkeypatched to a tiny in-memory fake
that understands exactly the SQL statements _process_successful_charge and
payment_initialize issue (matched by statement prefix), backed by plain
dicts for `users` and `payments`.
"""
import hashlib
import hmac
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

import app as appmod


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._result = None

    def execute(self, sql, params=()):
        sql = sql.strip()
        users, payments = self.store['users'], self.store['payments']

        if sql.startswith('INSERT INTO payments'):
            reference, user_id, plan, amount = params
            payments.setdefault(reference, {
                'reference': reference, 'user_id': user_id, 'plan': plan,
                'amount': amount, 'status': 'initialized',
            })
            self._result = None

        elif sql.startswith('SELECT user_id, status, plan FROM payments'):
            (reference,) = params
            row = payments.get(reference)
            self._result = (row['user_id'], row['status'], row['plan']) if row else None

        elif sql.startswith('SELECT id,name,email,is_premium,premium_until FROM users'):
            (user_id,) = params
            u = users.get(user_id)
            self._result = (u['id'], u['name'], u['email'], u['is_premium'], u['premium_until']) if u else None

        elif sql.startswith('SELECT premium_until FROM users'):
            (user_id,) = params
            u = users.get(user_id)
            self._result = (u['premium_until'],) if u else None

        elif sql.startswith('UPDATE users SET is_premium=TRUE'):
            expires, user_id = params
            users[user_id]['is_premium'] = True
            users[user_id]['premium_until'] = expires
            self._result = None

        elif sql.startswith("UPDATE payments SET status='success'"):
            plan, amount, days, expires, source, raw_payload, reference = params
            payments[reference].update(
                status='success', plan=plan, amount=amount, days_granted=days,
                premium_until=expires, source=source, raw_payload=raw_payload,
            )
            self._result = None

        else:
            raise AssertionError(f'FakeCursor got an unexpected statement: {sql!r}')

    def fetchone(self):
        return self._result

    def close(self):
        pass


class FakeConn:
    def __init__(self, store):
        self.store = store

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def store():
    return {
        'users': {
            1: {'id': 1, 'name': 'Test Customer', 'email': 'customer@example.com',
                'is_premium': False, 'premium_until': None},
        },
        'payments': {
            'azpred_1_deadbeef': {
                'reference': 'azpred_1_deadbeef', 'user_id': 1, 'plan': 'monthly',
                'amount': 5000, 'status': 'initialized',
            },
        },
    }


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setattr(appmod, '_KORAPAY_SECRET', 'test_secret_key')

    @contextmanager
    def fake_db():
        yield FakeConn(store), FakeCursor(store)

    monkeypatch.setattr(appmod, '_db', fake_db)
    appmod.app.config['TESTING'] = True
    with appmod.app.test_client() as c:
        yield c


def _sign(data: dict, secret: str) -> tuple[bytes, str]:
    """Build the raw webhook body + matching x-korapay-signature header."""
    raw = json.dumps({'event': 'charge.success', 'data': data}).encode('utf-8')
    encoded = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    sig = hmac.new(secret.encode('utf-8'), encoded, hashlib.sha256).hexdigest()
    return raw, sig


def _charge_data(**overrides):
    data = {
        'reference': 'azpred_1_deadbeef',
        'status': 'success',
        'amount': 5000,
        'currency': 'NGN',
        'metadata': {'plan': 'monthly', 'user_id': 1},
    }
    data.update(overrides)
    return data


def test_webhook_grants_premium_for_valid_monthly_charge(client, store):
    raw, sig = _sign(_charge_data(), 'test_secret_key')

    resp = client.post('/api/payment/webhook', data=raw,
                        headers={'x-korapay-signature': sig,
                                 'Content-Type': 'application/json'})

    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'ok'

    user = store['users'][1]
    assert user['is_premium'] is True
    assert user['premium_until'] is not None
    days_left = (user['premium_until'] - datetime.now(timezone.utc)).days
    assert 28 <= days_left <= 30

    assert store['payments']['azpred_1_deadbeef']['status'] == 'success'
    assert store['payments']['azpred_1_deadbeef']['days_granted'] == 30


def test_webhook_rejects_invalid_signature(client, store):
    raw, _real_sig = _sign(_charge_data(), 'test_secret_key')

    resp = client.post('/api/payment/webhook', data=raw,
                        headers={'x-korapay-signature': 'not-the-real-signature',
                                 'Content-Type': 'application/json'})

    assert resp.status_code == 401
    assert store['users'][1]['is_premium'] is False


def test_webhook_is_idempotent_on_replay(client, store):
    raw, sig = _sign(_charge_data(), 'test_secret_key')
    headers = {'x-korapay-signature': sig, 'Content-Type': 'application/json'}

    first = client.post('/api/payment/webhook', data=raw, headers=headers)
    assert first.status_code == 200
    until_after_first = store['users'][1]['premium_until']

    # Korapay (or a network blip) replays the exact same webhook.
    second = client.post('/api/payment/webhook', data=raw, headers=headers)
    assert second.status_code == 200

    # Must NOT extend a second time.
    assert store['users'][1]['premium_until'] == until_after_first
