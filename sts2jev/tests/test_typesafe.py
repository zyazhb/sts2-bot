"""TypeSafe remote Choice client, no live API."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from sts2jev.typesafe import TypeSafeJevClient, jev_key_from_argv


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)
        self.ok = 200 <= status_code < 300

    def json(self) -> dict[str, Any]:
        return self._payload


class TypeSafeClientTests(unittest.TestCase):
    def test_choice_request_uses_labels_and_bearer(self) -> None:
        seen: dict[str, Any] = {}

        def post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> _Response:
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            seen["timeout"] = timeout
            return _Response(
                200,
                {
                    "answers": {
                        "action": {
                            "choice": "end_turn",
                            "probabilities": {"end_turn": 0.8, "UNKNOWN": 0.2},
                            "confidence": 0.7,
                        }
                    }
                },
            )

        client = TypeSafeJevClient(api_key="sk_test")
        with patch("sts2jev.typesafe.requests.post", post):
            decision = client.decide_choice(
                {"screen": "COMBAT", "options": {"end_turn": "end turn"}},
                ["end_turn"],
                criteria="Keep HP high.",
            )
        self.assertEqual(seen["url"], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer sk_test")
        question = seen["json"]["questions"]["action"]
        self.assertEqual(question["criteria"]["end_turn"], "end turn")
        self.assertIn("UNKNOWN", question["criteria"])
        self.assertEqual(decision.value, "end_turn")
        self.assertFalse(decision.abstained)
        self.assertAlmostEqual(decision.confidence, 0.7)

    def test_low_confidence_abstains(self) -> None:
        def post(*_args: Any, **_kwargs: Any) -> _Response:
            return _Response(
                200,
                {"answers": {"action": {"choice": "end_turn", "probabilities": {}, "confidence": 0.2}}},
            )

        client = TypeSafeJevClient(api_key="sk_test", abstain_threshold=0.45)
        with patch("sts2jev.typesafe.requests.post", post):
            decision = client.decide_choice({"options": {}}, ["end_turn"])
        self.assertTrue(decision.abstained)

    def test_jevkey_flag(self) -> None:
        self.assertEqual(jev_key_from_argv(["--jevkey=sk_live"]), "sk_live")
        self.assertEqual(jev_key_from_argv(["--jevkey", "sk_live"]), "sk_live")
        self.assertIsNone(jev_key_from_argv([]))
        with self.assertRaises(SystemExit):
            jev_key_from_argv(["--other"])
