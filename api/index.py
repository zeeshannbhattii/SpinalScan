"""
Vercel Serverless Function Entrypoint for SpinalScan.
Directs all incoming HTTP requests to the SpinalScan Flask application.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import Flask instance
from server import app


class VercelPathFixMiddleware:
    """
    WSGI Middleware to normalize PATH_INFO when Vercel rewrites routes to /api/index.py.
    Extracts the true original path from HTTP_X_MATCHED_PATH or PATH_INFO.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched = environ.get("HTTP_X_MATCHED_PATH")
        if matched:
            clean = matched.split("?")[0]
            if clean in ("/", "/index.html", "/api/index", "/api/index.py", "/api"):
                environ["PATH_INFO"] = "/"
            else:
                environ["PATH_INFO"] = clean
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index.py", "/api"):
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


# Wrap Flask with Vercel path normalization
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

# Expose app and handler for Vercel
handler = app
