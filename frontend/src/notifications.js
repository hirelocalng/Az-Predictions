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
      // Register change listener BEFORE login() so we catch the subscription
      // update that login() triggers (race condition fix).
      try {
        OneSignal.User.PushSubscription.addEventListener('change', () =>
          _saveSubscriptionId(token)
        )
      } catch (_) {}

      // Link this browser's subscription to the logged-in user
      await OneSignal.login(String(userId))

      // Try to save immediately — ID may already be populated
      await _saveSubscriptionId(token)

      // Retry after 800ms: PushSubscription.id is populated async after login()
      // and may be null on the first read above.
      setTimeout(() => _saveSubscriptionId(token), 800)
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
    // Primary: live subscription ID from SDK
    let id = window.OneSignal?.User?.PushSubscription?.id
    // Fallback: ID cached to localStorage by the init block (covers anonymous
    // subscribers who clicked Allow before logging in)
    if (!id) id = localStorage.getItem('os_sub_id')
    if (!id) return

    const optedIn = window.OneSignal?.User?.PushSubscription?.optedIn
    // If we have a cached ID but the live optedIn is explicitly false, skip
    if (optedIn === false) return

    await fetch('/api/user/notification-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ player_id: id }),
    })
  } catch { /* silent */ }
}
