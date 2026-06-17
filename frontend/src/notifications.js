// OneSignal Web SDK v16 — loaded via CDN script in index.html.
// We queue operations through OneSignalDeferred so they run after the SDK
// finishes initialising, regardless of when React calls these functions.

function _push(cb) {
  if (window.OneSignalDeferred) {
    window.OneSignalDeferred.push(cb)
  } else if (window.OneSignal) {
    cb(window.OneSignal)
  }
}

// No-op: init is handled by the <script> block in index.html.
export function initOneSignal() {}

export function oneSignalLogin(userId, token) {
  _push(async (OneSignal) => {
    try {
      // Link this browser to the logged-in user
      await OneSignal.login(String(userId))
      // Show the permission slidedown prompt
      try { await OneSignal.Slidedown.promptPush() } catch (_) {}
      // Save the subscription ID to our backend
      await _saveSubscriptionId(token)
      // Re-save whenever subscription state changes (permission granted later, etc.)
      OneSignal.User.PushSubscription.addEventListener('change', () =>
        _saveSubscriptionId(token)
      )
    } catch (e) {
      console.warn('[OneSignal] login:', e)
    }
  })
}

export function oneSignalLogout() {
  _push(async (OneSignal) => {
    try { await OneSignal.logout() } catch (e) { console.warn('[OneSignal] logout:', e) }
  })
}

async function _saveSubscriptionId(token) {
  if (!token) return
  try {
    const id      = window.OneSignal?.User?.PushSubscription?.id
    const optedIn = window.OneSignal?.User?.PushSubscription?.optedIn
    if (!id || !optedIn) return
    await fetch('/api/user/notification-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ player_id: id }),
    })
  } catch { /* silent */ }
}
