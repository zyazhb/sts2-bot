"""Poll STS2 snapshots, decide with OpenJev, and POST one action at a time."""

from __future__ import annotations

import time
from typing import Any

from openjevpro.client import OpenJevProClient

from sts2jev.candidates import Candidate, ExpandResult, PAUSE_SCREENS, expand
from sts2jev.decide import ActionDecision, decide_action
from sts2jev.http import Sts2Client, Sts2HttpError

RETRYABLE_CODES = frozenset({"action_in_flight", "state_unavailable", "pause_pending", "session_not_ready"})


class StopPlay(Exception):
    def __init__(self, reason: str, *, candidates: list[Candidate] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.candidates = candidates or []


def run_loop(
    game: Sts2Client,
    jev: OpenJevProClient,
    *,
    poll_s: float = 0.4,
) -> None:
    health = game.connect()
    print(f"connected sts2-ai-agent port={health.get('api_port')} role={health.get('instance_role')}")
    blocked_ids: set[str] = set()
    opened_claim: str | None = None

    while True:
        snapshot = game.snapshot()
        state = snapshot.get("state") or {}
        screen = state.get("screen")
        if screen in PAUSE_SCREENS:
            raise StopPlay(f"paused on {screen}")
        if screen != "REWARD":
            blocked_ids.clear()
            opened_claim = None

        if _should_wait_combat(state):
            time.sleep(poll_s)
            continue
        if screen == "MAP" and (state.get("map") or {}).get("local_vote"):
            time.sleep(poll_s)
            continue
        if _waiting_character_select(state):
            time.sleep(poll_s)
            continue
        if screen == "GAME_OVER" and (state.get("game_over") or {}).get("phase") == "summary_animating":
            time.sleep(poll_s)
            continue

        expanded = expand(snapshot, blocked_ids=frozenset(blocked_ids))
        if expanded.stop_reason:
            raise StopPlay(expanded.stop_reason, candidates=expanded.candidates)
        if screen == "GAME_OVER":
            _finish_game_over(game, expanded)
            continue
        if not expanded.candidates:
            raise StopPlay(f"no legal candidates on {screen}")

        decision = decide_action(jev, snapshot, expanded.candidates)
        if decision.abstained or decision.candidate is None:
            raise StopPlay(f"no legal candidates on {screen}", candidates=expanded.candidates)
        outcome = _submit(game, decision, poll_s=poll_s)
        if outcome == "ok" and decision.candidate is not None:
            opened_claim = _note_reward_choice(decision.candidate, blocked_ids, opened_claim)
        if outcome == "stale":
            continue


def _note_reward_choice(
    candidate: Candidate,
    blocked_ids: set[str],
    opened_claim: str | None,
) -> str | None:
    if candidate.action == "claim_reward":
        return candidate.id
    if candidate.action == "skip_reward_cards" and opened_claim:
        blocked_ids.add(opened_claim)
    return None


def _should_wait_combat(state: dict[str, Any]) -> bool:
    if state.get("screen") != "COMBAT":
        return False
    readiness = (state.get("combat") or {}).get("action_readiness") or {}
    if readiness.get("can_use_combat_actions") is not False:
        return False
    return readiness.get("reason") != "modal_open"


def _waiting_character_select(state: dict[str, Any]) -> bool:
    if state.get("screen") != "CHARACTER_SELECT":
        return False
    select = state.get("character_select") or {}
    return bool(select.get("is_waiting_for_players"))


def _finish_game_over(game: Sts2Client, expanded: ExpandResult) -> None:
    continuing = [item for item in expanded.candidates if item.action == "continue_game_over"]
    if continuing:
        print("game over: continue_game_over")
        try:
            result = game.act(continuing[0].body, timeout=90.0)
        except Sts2HttpError as exc:
            if exc.retryable or exc.code in RETRYABLE_CODES:
                time.sleep(0.4)
                return
            raise StopPlay(f"game over continue failed: {exc.code} {exc.message}") from exc
        if result.get("status") == "failed":
            raise StopPlay(f"game over continue failed: {result.get('message')}")
        return
    raise StopPlay("game over complete")


def _submit(game: Sts2Client, decision: ActionDecision, *, poll_s: float) -> str:
    candidate = decision.candidate
    choice = decision.choice
    if candidate is None or choice is None:
        raise StopPlay("missing decision")
    body = dict(candidate.body)
    body["client_context"] = {
        "decision_reason": f"choice={candidate.id} confidence={choice.confidence:.4f}"
    }
    note = f" {decision.reason}" if decision.reason else ""
    print(f"act {candidate.id} ({candidate.label}) conf={choice.confidence:.2%}{note}")
    try:
        result = game.act(body)
    except Sts2HttpError as exc:
        if exc.retryable or exc.code in RETRYABLE_CODES:
            time.sleep(poll_s)
            return "retry"
        if exc.code in {"invalid_action", "invalid_target"}:
            corrected = _correct_once(candidate, exc.details)
            if corrected is None:
                print(f"stale {exc.code}: {exc.message}")
                time.sleep(poll_s)
                return "stale"
            print(f"correct {exc.details.get('field')} -> {corrected.body}")
            try:
                game.act(
                    {
                        **corrected.body,
                        "client_context": body["client_context"],
                    }
                )
            except Sts2HttpError as retry_exc:
                print(f"stale {retry_exc.code}: {retry_exc.message}")
                time.sleep(poll_s)
                return "stale"
            return "ok"
        raise StopPlay(f"{exc.code}: {exc.message}", candidates=[candidate]) from exc
    if result.get("status") == "failed":
        raise StopPlay(f"action failed: {result.get('message')}", candidates=[candidate])
    if result.get("status") == "pending":
        time.sleep(poll_s)
    return "ok"


def _correct_once(candidate: Candidate, details: dict[str, Any]) -> Candidate | None:
    field = details.get("field")
    valid = details.get("valid_indices") or []
    if field not in {"card_index", "target_index", "option_index"} or not valid:
        return None
    body = dict(candidate.body)
    if body.get(field) in valid:
        return None
    body[field] = valid[0]
    return Candidate(
        id=f"{candidate.id}->fix:{field}={valid[0]}",
        action=candidate.action,
        label=f"{candidate.label} (corrected {field})",
        body=body,
        index_key=candidate.index_key,
        target_key=candidate.target_key,
    )
