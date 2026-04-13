#!/usr/bin/env python3
"""
LLM GPU Calculator - Zero-dependency static file server.
Serves static files from the ./static directory and proxies
/api/hf-proxy/* to huggingface.co (mirrors the Cloudflare Worker behaviour).
"""

import http.server
import socketserver
import os
import sys
import urllib.request
import urllib.error

PORT = int(os.environ.get("PORT", 8080))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

HF_ORIGIN = "https://huggingface.co"
PROXY_PREFIX = "/api/hf-proxy"

# Headers forwarded from the browser to HuggingFace
FORWARD_REQ_HEADERS = {"authorization", "range"}

# Headers forwarded back to the browser (everything except hop-by-hop)
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    # ── CORS pre-flight ────────────────────────────────────────
    def do_OPTIONS(self):
        if self.path.startswith(PROXY_PREFIX + "/"):
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_error(405)

    # ── GET / HEAD — proxy or static ──────────────────────────
    def do_GET(self):
        if self.path.startswith(PROXY_PREFIX + "/"):
            self._proxy()
        else:
            super().do_GET()

    def do_HEAD(self):
        if self.path.startswith(PROXY_PREFIX + "/"):
            self._proxy(head=True)
        else:
            super().do_HEAD()

    # ── Proxy implementation ───────────────────────────────────
    def _proxy(self, head=False):
        hf_path = self.path[len(PROXY_PREFIX):]   # keep leading /
        hf_url  = f"{HF_ORIGIN}{hf_path}"

        req = urllib.request.Request(hf_url, method="HEAD" if head else "GET")
        for h in FORWARD_REQ_HEADERS:
            v = self.headers.get(h)
            if v:
                req.add_header(h.capitalize(), v)

        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in HOP_BY_HOP:
                        self.send_header(key, val)
                self._send_cors_headers()
                self.end_headers()
                if not head:
                    self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            if not head:
                self.wfile.write(str(e).encode())
        except Exception as e:
            self.send_response(502)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            if not head:
                self.wfile.write(f"Proxy error: {e}".encode())

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",   "*")
        self.send_header("Access-Control-Allow-Methods",  "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",  "Authorization, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")

    def end_headers(self):
        # Security headers (static files only — proxy responses skip this path)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()


def main():
    os.chdir(STATIC_DIR)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"LLM GPU Calculator running on http://0.0.0.0:{PORT}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.", flush=True)
            sys.exit(0)


if __name__ == "__main__":
    main()
