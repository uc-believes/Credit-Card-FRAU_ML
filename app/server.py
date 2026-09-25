"""
Fraud Shield — Production Dashboard Web Server
===============================================
Flask entrypoint serving the Professional Fraud Analytics Dashboard.
"""

from __future__ import annotations

import pathlib
import sys

# Ensure project root in sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, send_from_directory

from app.routes.api import api_bp
from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("app.server")


def create_app() -> Flask:
    """Application factory for Fraud Shield."""
    root = get_project_root()
    cfg = get_config()

    app = Flask(
        __name__,
        static_folder=str(root / "app" / "static"),
        template_folder=str(root / "app" / "templates"),
    )
    import os
    env_secret = os.environ.get("SECRET_KEY")
    app.config["SECRET_KEY"] = env_secret or cfg.get("app", {}).get(
        "secret_key", "fraud-shield-institutional-secret"
    )

    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Serve static plots from artifacts/plots/
    plots_dir = root / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    @app.route("/static/plots/<path:filename>")
    def serve_plot(filename: str):
        return send_from_directory(str(plots_dir), filename)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/dossier")
    @app.route("/report")
    def dossier():
        docs_dir = root / "docs"
        return send_from_directory(str(docs_dir), "PROJECT_EVALUATION_DOSSIER.html")

    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "service": "Fraud Shield Intelligence Dashboard",
            "model": "XGBoost v1.0",
        })

    return app


def main():
    cfg = get_config()
    host = cfg.get("app", {}).get("host", "0.0.0.0")
    port = int(cfg.get("app", {}).get("port", 5000))
    debug = bool(cfg.get("app", {}).get("debug", False))

    app = create_app()
    print("=" * 80)
    print(" FRAUD SHIELD — RISK INTELLIGENCE DASHBOARD")
    print(f" * Server running on: http://127.0.0.1:{port}")
    print("=" * 80)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
