#!/usr/bin/env python3
"""
LLM GPU Calculator - Zero-dependency static file server.
Serves static files from the ./static directory.
"""

import http.server
import socketserver
import os
import sys

PORT = int(os.environ.get("PORT", 8080))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def end_headers(self):
        # Security headers
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
