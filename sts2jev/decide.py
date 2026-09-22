"""Layered Choice decisions over STS2 action candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from openjevpro.client import OpenJevProClient
from openjevpro.schemas import ChoiceDecision

from sts2jev.candidates import Candidate

LETTER_LIMIT = 26


@dataclass
class ActionDecision:
    candidate: Candidate | None
    choice: ChoiceDecision | None
    abstained: bool
    reason: str = ""


def decide_action(
    client: OpenJevProClient,
    snapshot: dict[str, Any],
    candidates: list[Candidate],
) -> ActionDecision:
    if not candidates:
        return ActionDecision(None, None, True, "no candidates")
    remaining = candidates
    for grouping in (group_by_action, group_by_index, group_by_target):
        if len(remaining) <= LETTER_LIMIT:
            break
        groups = grouping(remaining)
        if len(groups) <= 1:
            continue
        picked = _choose_group(client, snapshot, groups)
        if picked.abstained or picked.candidate is None:
            return picked
        remaining = groups.get(picked.candidate.id, remaining)
    return _one_shot(client, snapshot, remaining)


def group_by_action(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    return _group(candidates, lambda item: item.action)


def group_by_index(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    return _group(candidates, lambda item: str(item.index_key) if item.index_key is not None else item.id)


def group_by_target(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    return _group(candidates, lambda item: str(item.target_key) if item.target_key is not None else item.id)


def _group(candidates: list[Candidate], key_fn: Callable[[Candidate], str]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for item in candidates:
        grouped.setdefault(key_fn(item), []).append(item)
    return grouped


def _choose_group(
    client: OpenJevProClient,
    snapshot: dict[str, Any],
    groups: dict[str, list[Candidate]],
) -> ActionDecision:
    proxies = [
        Candidate(
            id=key,
            action=key,
            label=_group_label(key, items),
            body={"action": items[0].action},
        )
        for key, items in groups.items()
    ]
    return _one_shot(client, snapshot, proxies)


def _group_label(key: str, items: list[Candidate]) -> str:
    sample = items[0].label
    return f"{key} ({len(items)} options, e.g. {sample})"


def _tournament(
    client: OpenJevProClient,
    snapshot: dict[str, Any],
    candidates: list[Candidate],
) -> list[Candidate] | ActionDecision:
    remaining = candidates
    while len(remaining) > LETTER_LIMIT:
        winners: list[Candidate] = []
        for start in range(0, len(remaining), LETTER_LIMIT):
            batch = remaining[start : start + LETTER_LIMIT]
            picked = _one_shot(client, snapshot, batch)
            if picked.abstained or picked.candidate is None:
                return picked
            winners.append(picked.candidate)
        remaining = winners
    return remaining


def _one_shot(
    client: OpenJevProClient,
    snapshot: dict[str, Any],
    candidates: list[Candidate],
) -> ActionDecision:
    if len(candidates) > LETTER_LIMIT:
        narrowed = _tournament(client, snapshot, candidates)
        if isinstance(narrowed, ActionDecision):
            return narrowed
        candidates = narrowed
    options = [item.id for item in candidates]
    choice = client.decide_choice(
        state=_slice_state(snapshot, candidates),
        candidates=options,
        criteria=_criteria(snapshot),
        allow_abstain=True,
    )
    if choice.abstained or choice.value not in options:
        return ActionDecision(None, choice, True, "abstained")
    selected = next(item for item in candidates if item.id == choice.value)
    return ActionDecision(selected, choice, False)


def _slice_state(snapshot: dict[str, Any], candidates: list[Candidate]) -> dict[str, Any]:
    state = snapshot.get("state") or {}
    screen = state.get("screen")
    combat = state.get("combat") or {}
    player = combat.get("player") or {}
    run = state.get("run") or {}
    sliced: dict[str, Any] = {
        "screen": screen,
        "run_id": state.get("run_id"),
        "hp": _hp(player) or _hp(run),
        "gold": run.get("gold"),
        "options": {item.id: item.label for item in candidates},
    }
    if screen == "COMBAT":
        sliced["energy"] = player.get("energy")
        sliced["block"] = player.get("block")
        sliced["end_turn_will_kill"] = combat.get("end_turn_will_kill_player")
        sliced["enemies"] = [_enemy_slice(enemy) for enemy in combat.get("enemies") or []]
        sliced["hand"] = [_hand_slice(card) for card in combat.get("hand") or []]
    elif screen == "MAP":
        mapping = state.get("map") or {}
        sliced["nodes"] = mapping.get("options") or mapping.get("available_nodes") or []
    elif screen == "SHOP":
        shop = state.get("shop") or {}
        sliced["shop_open"] = shop.get("open", shop.get("is_open"))
        sliced["cards"] = shop.get("cards") or []
        sliced["relics"] = shop.get("relics") or []
        sliced["potions"] = shop.get("potions") or []
    elif screen == "REWARD":
        sliced["reward"] = state.get("reward") or {}
    elif screen == "EVENT":
        sliced["event"] = state.get("event") or {}
    elif screen == "REST":
        sliced["rest"] = state.get("rest") or {}
    return sliced


def _hp(entity: dict[str, Any]) -> Any:
    if entity.get("hp") is not None:
        return entity["hp"]
    if entity.get("current_hp") is not None:
        return f"{entity.get('current_hp')}/{entity.get('max_hp')}"
    return None


def _enemy_slice(enemy: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": enemy.get("name") or enemy.get("line") or enemy.get("enemy_id"),
        "hp": _hp(enemy),
        "block": enemy.get("block"),
        "intents": enemy.get("intents") or enemy.get("intent"),
    }


def _hand_slice(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": card.get("name") or card.get("line") or card.get("card_id"),
        "energy_cost": card.get("energy_cost"),
        "playable": card.get("playable"),
        "requires_target": card.get("requires_target"),
    }


def _criteria(snapshot: dict[str, Any]) -> str:
    screen = (snapshot.get("state") or {}).get("screen")
    rules = {
        "COMBAT": (
            "Spend energy on playable cards. Prioritize lethal or high incoming intent damage. "
            "Do not end the turn with unused efficient plays. Avoid ending the turn if it kills the player."
        ),
        "REWARD": "Take cards only when the upgrade is clear; otherwise skip. Prefer relics and potions that fit the deck.",
        "SHOP": "Check relics and card removal before spending gold. Buy only affordable stocked items that clearly help.",
        "EVENT": "Prefer unlocked options. Avoid options marked KILLS unless no alternative remains.",
        "MAP": "Pick a node that advances the run. Prefer rest when wounded, shops when gold is high, elites when strong.",
        "REST": "Heal when missing substantial HP; smith when HP is comfortable.",
        "CHEST": "Take the relic that best fits the current deck.",
        "CARD_SELECTION": "Pick the card that the prompt is asking to remove, upgrade, or transform.",
    }
    return rules.get(screen, "Pick the single best legal action for this screen.")
