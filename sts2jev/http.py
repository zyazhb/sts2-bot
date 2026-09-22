"""Loopback HTTP client for the STS2 AI Agent mod."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

import requests


class Sts2HttpError(Exception):
    """Structured error from the mod HTTP envelope."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}
        self.status_code = status_code


class Sts2Client:
    """Calls /health, /decision-snapshot, and /action on 127.0.0.1."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._power_catalog: dict[str, str] | None = None
        self._relic_catalog: dict[str, str] | None = None
        self._potion_catalog: dict[str, str] | None = None
        self._card_catalog: dict[str, str] | None = None
        self._move_catalog: dict[str, str] | None = None

    def connect(self) -> dict[str, Any]:
        """Confirm the mod is up and switch to the advertised api_port."""
        data = self.health()
        service = data.get("service")
        if service != "sts2-ai-agent":
            raise Sts2HttpError(
                "unexpected_service",
                f"expected sts2-ai-agent, got {service!r}",
                details={"health": data},
            )
        host = data.get("api_host") or "127.0.0.1"
        port = data.get("api_port")
        if port:
            rewritten = _rewrite_authority(self.base_url, str(host), int(port))
            if rewritten != self.base_url:
                self.base_url = rewritten
                data = self.health()
        return data

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def snapshot(self) -> dict[str, Any]:
        """Same-frame compact state plus action descriptors."""
        data = self._get("/decision-snapshot")
        if "state" not in data:
            raise Sts2HttpError(
                "invalid_snapshot",
                "decision-snapshot response is missing state",
                details={"data": data},
            )
        data.setdefault("available_actions", [])
        return data

    def act(self, body: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return self._request("POST", "/action", json_body=body, timeout=timeout or 60.0)

    def power_catalog(self) -> dict[str, str]:
        """id and name to plain-text effect. Loaded once from GET /data/powers."""
        from sts2jev.view import power_catalog_from_items

        return self._named_catalog("_power_catalog", "/data/powers", power_catalog_from_items)

    def relic_catalog(self) -> dict[str, str]:
        from sts2jev.view import power_catalog_from_items

        return self._named_catalog("_relic_catalog", "/data/relics", power_catalog_from_items)

    def potion_catalog(self) -> dict[str, str]:
        from sts2jev.view import power_catalog_from_items

        return self._named_catalog("_potion_catalog", "/data/potions", power_catalog_from_items)

    def card_catalog(self) -> dict[str, str]:
        from sts2jev.view import power_catalog_from_items

        return self._named_catalog("_card_catalog", "/data/cards", power_catalog_from_items)

    def move_catalog(self) -> dict[str, str]:
        from sts2jev.describe import move_catalog_from_items

        return self._named_catalog("_move_catalog", "/data/monsters", move_catalog_from_items)

    def _named_catalog(self, attr: str, path: str, parser: Any) -> dict[str, str]:
        cached = getattr(self, attr)
        if cached is not None:
            return cached
        items = self._load_items(path)
        if not items:
            return {}
        catalog = parser(items)
        if not catalog:
            return {}
        setattr(self, attr, catalog)
        return catalog

    def _load_items(self, path: str) -> list[Any]:
        try:
            resp = self.session.get(f"{self.base_url}{path}", timeout=self.timeout)
            payload = resp.json()
        except (requests.RequestException, ValueError):
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            items = payload.get("data", payload)
            return items if isinstance(items, list) else []
        return []

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(
                method,
                url,
                json=json_body,
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise Sts2HttpError("transport_error", str(exc)) from exc
        payload = _parse_json(resp)
        if not payload.get("ok", resp.ok):
            error = payload.get("error") or {}
            raise Sts2HttpError(
                str(error.get("code") or "http_error"),
                str(error.get("message") or resp.text or resp.reason),
                retryable=bool(error.get("retryable", False)),
                details=error.get("details") or {},
                status_code=resp.status_code,
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise Sts2HttpError(
                "invalid_response",
                "response data is not an object",
                details={"payload": payload},
                status_code=resp.status_code,
            )
        return data


def _parse_json(resp: requests.Response) -> dict[str, Any]:
    try:
        payload = resp.json()
    except ValueError as exc:
        raise Sts2HttpError(
            "invalid_json",
            f"non-JSON response ({resp.status_code})",
            details={"text": resp.text[:500]},
            status_code=resp.status_code,
        ) from exc
    if not isinstance(payload, dict):
        raise Sts2HttpError(
            "invalid_json",
            "JSON root is not an object",
            status_code=resp.status_code,
        )
    return payload


def _rewrite_authority(base_url: str, host: str, port: int) -> str:
    parsed = urlparse(base_url)
    return urlunparse(parsed._replace(netloc=f"{host}:{port}"))
