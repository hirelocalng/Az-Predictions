import OneSignal from 'react-onesignal'

let _ready = false

export async function initOneSignal(appId) {
  if (_ready || !appId) return
  try {
    await OneSignal.init({
      appId,
      allowLocalhostAsSecureOrigin: true,
      serviceWorkerPath: '/OneSignalSDKWorker.js',
      notifyButton: { enable: false },
    })
    _ready = true
  } catch (e) {
    // "already initialized" is fine — treat as ready
    if (String(e).toLowerCase().includes('already')) {
      _ready = true
    } else {
      console.warn('[OneSignal] init:', e)
    }
  }
}

export async function oneSignalLogin(userId, token) {
  if (!_ready) return
  try {
    await OneSignal.login(String(userId))
    // Request permission; on grant capture subscription ID and save to backend
    await OneSignal.Notifications.requestPermission()
    await _saveSubscriptionId(token)
    // Also listen for future subscription changes (e.g. user grants later)
    OneSignal.User.PushSubscription.addEventListener('change', async () => {
      await _saveSubscriptionId(token)
    })
  } catch (e) {
    console.warn('[OneSignal] login/subscribe:', e)
  }
}

export async function oneSignalLogout() {
  if (!_ready) return
  try { await OneSignal.logout() } catch (e) { console.warn('[OneSignal] logout:', e) }
}

async function _saveSubscriptionId(token) {
  if (!token) return
  const id      = OneSignal.User.PushSubscription.id
  const optedIn = OneSignal.User.PushSubscription.optedIn
  if (!id || !optedIn) return
  try {
    await fetch('/api/user/notification-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ player_id: id }),
    })
  } catch { /* silent */ }
}
