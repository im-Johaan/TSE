// Server address and passcode shared by the TSE pages.
//
// The pages work in two ways:
// * served by the TSE server itself (scripts/serve.py): API calls go to the same origin;
// * served by a static host (e.g. Vercel): API calls go to TSE_CONFIG.api from config.js, or
//   to the address given once with ?api=https://... (remembered in this browser).
// A passcode, if the server requires one, is asked for once and remembered (or ?code=...).
(function () {
  "use strict";
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
    set(k, v) { try { if (v === null || v === "") localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (_) { /* storage blocked */ } },
  };
  const clean = a => (a || "").trim().replace(/\/+$/, "");
  const q = new URLSearchParams(location.search);
  if (q.has("api")) store.set("tse_api", clean(q.get("api")));
  if (q.has("code")) store.set("tse_code", q.get("code"));
  const cfg = window.TSE_CONFIG || {};
  const api = () => clean(store.get("tse_api") || cfg.api || "");

  window.TSE = {
    api,                                   // "" = same origin as the page
    code: () => store.get("tse_code") || "",
    setCode: c => store.set("tse_code", c),
    setApi: a => store.set("tse_api", clean(a)),
    serverLabel: () => api() || location.origin,
    wsUrl(path) {
      const base = api() || location.origin;
      return base.replace(/^http/, "ws") + path + "?code=" + encodeURIComponent(store.get("tse_code") || "");
    },
    askCode() {
      const c = prompt("This demo server needs a passcode:");
      if (c !== null) store.set("tse_code", c.trim());
      return c !== null;
    },
    async fetch(path, opts = {}) {
      const go = () => fetch(api() + path, { ...opts, headers: { ...(opts.headers || {}), "X-Passcode": store.get("tse_code") || "" } });
      let res = await go();
      if (res.status === 401 && this.askCode()) res = await go();
      return res;
    },
  };
})();
