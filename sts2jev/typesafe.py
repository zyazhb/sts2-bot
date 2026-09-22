"""Direct TypeSafe Jev Choice calls. https://docs.typesafe.ai/api"""

from __future__ import annotations

import argparse
import time
from typing import Any

import requests

from openjevpro.schemas import ChoiceDecision

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
QUESTION_ID = "action"


class TypeSafeJevClient:
    """Same decide_choice shape as OpenJevProClient, backed by the remote API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jev-latest",
        abstain_threshold: float = 0.45,
        base_url: str = TYPESAFE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.abstain_threshold = abstain_threshold
        self.base_url = base_url
        self.timeout = timeout

    def decide_choice(
        self,
        state: dict[str, Any],
        candidates: list[str],
        criteria: str | dict[str, str] = "",
        allow_abstain: bool = True,
    ) -> ChoiceDecision:
        options = list(candidates)
        if allow_abstain and "UNKNOWN" not in options:
            options.append("UNKNOWN")
        body = {
            "state": state,
            "model": self.model,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": _instructions(criteria),
                    "criteria": _choice_criteria(state, options),
                }
            },
        }
        payload = self._post(body)
        answer = (payload.get("answers") or {}).get(QUESTION_ID) or {}
        value = str(answer.get("choice") or "")
        probabilities = {
            str(key): float(prob) for key, prob in (answer.get("probabilities") or {}).items()
        }
        confidence = float(answer.get("confidence") or 0.0)
        abstained = value == "UNKNOWN" or value not in options or confidence < self.abstain_threshold
        return ChoiceDecision(
            value=value,
            probabilities=probabilities,
            confidence=confidence,
            abstained=abstained,
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        delay = 0.5
        response: requests.Response | None = None
        for attempt in range(3):
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout,
            )
            if response.status_code in {429, 529} and attempt < 2:
                time.sleep(delay)
                delay *= 2
                continue
            break
        assert response is not None
        if not response.ok:
            raise RuntimeError(f"typesafe {response.status_code}: {response.text[:300]}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("typesafe response is not an object")
        return payload


def jev_key_from_argv(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jevkey", default=None)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit(f"unknown arguments: {' '.join(unknown)}")
    if args.jevkey is None:
        return None
    key = str(args.jevkey).strip()
    if not key:
        raise SystemExit("--jevkey requires a key")
    return key


def _instructions(criteria: str | dict[str, str]) -> str:
    if isinstance(criteria, dict):
        return "\n".join(f"{key}: {value}" for key, value in criteria.items())
    text = str(criteria).strip()
    return text or "Pick the single best option."


def _choice_criteria(state: dict[str, Any], options: list[str]) -> dict[str, str | None]:
    labels = state.get("options") if isinstance(state.get("options"), dict) else {}
    criteria: dict[str, str | None] = {}
    for option in options:
        if option == "UNKNOWN":
            criteria[option] = "Not enough information to choose."
            continue
        label = labels.get(option) if isinstance(labels, dict) else None
        criteria[option] = str(label) if label else None
    return criteria
