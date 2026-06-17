/* AZ Prediction — Service Worker v4 */

// OneSignal must be imported first so its push/notificationclick handlers
// are registered before any other SW code runs.
importScripts('https://cdn.onesignal.com/sdks/web/v16/OneSignalSDK.sw.js');

const CACHE = 'az-predict-v4';

// Pre-cache the app shell (hashed Vite assets are cached at runtime below)
const PRECACHE = ['/', '/manifest.json', '/icon-192.png', '/icon-512.png'];

// ── Install ───────────────────────────────────────────────────────────────────

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE))
  );
  self.skipWaiting();
});

// ── Activate: clear old cache versions ───────────────────────────────────────

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Fetch ─────────────────────────────────────────────────────────────────────

self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle same-origin GET requests
  if (url.origin !== location.origin || request.method !== 'GET') return;

  // Admin routes — always network, no caching (requires auth)
  if (url.pathname.startsWith('/admin')) return;

  // API routes — network-first, cache as fallback so data stays fresh
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          if (response.ok) {
            // Clone SYNCHRONOUSLY before returning the original — calling clone()
            // inside caches.open().then() would be too late (body already consumed).
            const copy = response.clone();
            caches.open(CACHE).then(c => c.put(request, copy));
          }
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Vite-hashed static assets (/assets/index-*.js etc) — cache-first
  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(
      caches.match(request).then(cached => {
        if (cached) return cached;
        return fetch(request).then(response => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(CACHE).then(c => c.put(request, copy));
          }
          return response;
        });
      })
    );
    return;
  }

  // Everything else (SPA HTML routes) — network-first, fall back to cached /
  event.respondWith(
    fetch(request)
      .then(response => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then(c => c.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request) || caches.match('/'))
  );
});
