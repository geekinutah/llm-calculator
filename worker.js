/**
 * Cloudflare Worker — LLM Calculator
 *
 * Routes:
 *   /api/hf-proxy/*  — server-side proxy to api.huggingface.co / huggingface.co
 *                       (avoids CORS restrictions in the browser)
 *   everything else  — served from ./static via the ASSETS binding
 */

const HF_ORIGIN = 'https://huggingface.co';

// Headers the browser is allowed to send to the proxy
const ALLOWED_REQUEST_HEADERS = ['authorization', 'range'];

// Headers the browser is allowed to read from the proxy response
const EXPOSED_RESPONSE_HEADERS = 'Content-Range, Content-Length, Accept-Ranges';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // ── CORS pre-flight ──────────────────────────────────────
    if (request.method === 'OPTIONS' && url.pathname.startsWith('/api/hf-proxy/')) {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(),
      });
    }

    // ── HuggingFace proxy ────────────────────────────────────
    if (url.pathname.startsWith('/api/hf-proxy/')) {
      const hfPath = url.pathname.slice('/api/hf-proxy'.length); // keep leading /
      const hfUrl  = `${HF_ORIGIN}${hfPath}${url.search}`;

      // Forward a safe subset of request headers
      const outbound = new Headers();
      for (const h of ALLOWED_REQUEST_HEADERS) {
        const v = request.headers.get(h);
        if (v) outbound.set(h, v);
      }

      let upstream;
      try {
        upstream = await fetch(hfUrl, {
          method:  request.method,
          headers: outbound,
          // body only for methods that carry one (unlikely here, but correct)
          body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
          redirect: 'follow',
        });
      } catch (err) {
        return new Response(`Proxy error: ${err.message}`, { status: 502, headers: corsHeaders() });
      }

      // Rebuild response with CORS headers added
      const respHeaders = new Headers(upstream.headers);
      for (const [k, v] of Object.entries(corsHeaders())) {
        respHeaders.set(k, v);
      }

      return new Response(upstream.body, {
        status:  upstream.status,
        headers: respHeaders,
      });
    }

    // ── Static assets (everything else) ─────────────────────
    return env.ASSETS.fetch(request);
  },
};

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin':   '*',
    'Access-Control-Allow-Methods':  'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers':  'Authorization, Range',
    'Access-Control-Expose-Headers': EXPOSED_RESPONSE_HEADERS,
  };
}
