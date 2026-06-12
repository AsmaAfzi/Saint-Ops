"""Demo Airflow UI proxy: server-side auto-login on GET /, then reverse-proxy."""

from __future__ import annotations

import os
import re
from urllib.parse import urljoin, urlparse

import requests
from flask import Flask, Response, redirect, request

UPSTREAM = os.environ.get("AIRFLOW_UPSTREAM", "http://airflow-webserver:8080").rstrip("/")
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "http://localhost:8081").rstrip("/")

# Airflow serves UI assets under /static/ — disable Flask's own static handler.
app = Flask(__name__, static_folder=None, static_url_path=None)

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}

CSRF_RE = re.compile(r'name="csrf_token"[^>]*value="([^"]+)"')


def _upstream_url(path: str = "") -> str:
    path = path.lstrip("/")
    return f"{UPSTREAM}/{path}" if path else f"{UPSTREAM}/"


def _forward_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers:
        lower = key.lower()
        if lower in {"host", "content-length", "connection"}:
            continue
        headers[key] = value
    headers["X-Forwarded-Host"] = request.host
    headers["X-Forwarded-Proto"] = request.scheme
    headers["X-Forwarded-For"] = request.remote_addr or "127.0.0.1"
    return headers


def _public_base() -> str:
    return f"{request.scheme}://{request.host}"


def _rewrite_location(location: str | None) -> str | None:
    if not location:
        return location
    base = _public_base()
    parsed = urlparse(location)
    if parsed.netloc in {"airflow-webserver:8080", "localhost", "127.0.0.1"}:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return f"{base}{path}"
    if location.startswith("/"):
        return f"{base}{location}"
    return location


def _proxy_response(upstream: requests.Response) -> Response:
    headers = []
    for key, value in upstream.headers.items():
        if key.lower() in HOP_BY_HOP:
            continue
        if key.lower() == "location":
            value = _rewrite_location(value) or value
        headers.append((key, value))

    return Response(upstream.content, upstream.status_code, headers)


def _proxy_request(path: str = "") -> Response:
    url = _upstream_url(path)
    if request.query_string:
        url = f"{url}?{request.query_string.decode()}"

    upstream = requests.request(
        method=request.method,
        url=url,
        headers=_forward_headers(),
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        timeout=300,
    )
    return _proxy_response(upstream)


@app.route("/", methods=["GET"])
def auto_login() -> Response:
    session = requests.Session()
    login_page = session.get(
        _upstream_url("login/"),
        headers={"X-Forwarded-Host": request.host, "X-Forwarded-Proto": request.scheme},
        timeout=60,
    )
    if login_page.status_code != 200:
        return Response(
            f"Airflow login page unavailable (HTTP {login_page.status_code})",
            status=502,
        )

    match = CSRF_RE.search(login_page.text)
    if not match:
        return Response("Could not read CSRF token from Airflow login page", status=502)

    session.post(
        _upstream_url("login/"),
        data={
            "csrf_token": match.group(1),
            "username": "admin",
            "password": "admin",
        },
        headers={"X-Forwarded-Host": request.host, "X-Forwarded-Proto": request.scheme},
        allow_redirects=False,
        timeout=60,
    )

    resp = redirect(f"{request.scheme}://{request.host}/home")
    for cookie in session.cookies:
        resp.set_cookie(
            cookie.name,
            cookie.value,
            path=cookie.path or "/",
            secure=False,
            httponly=True,
            samesite="Lax",
        )
    return resp


@app.route("/", defaults={"path": ""}, methods=["POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def proxy(path: str) -> Response:
    return _proxy_request(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
