"""Layered Choice decisions over STS2 action candidates."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from openjevpro.client import OpenJevProClient
from openjevpro.schemas import ChoiceDecision

from sts2jev.candidates import Candidate
from sts2jev.view import slice_state as _slice_state

LETTER_LIMIT = 26


@dataclass
class ActionDecision:
    candidate: Candidate | None
    choice: ChoiceDecision | None
    abstained: bool
    reason: str = ""


def collapse_equivalent(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the first candidate for each identical action+label pair."""
    return [items[0] for items in equivalent_groups(candidates).values()]


def equivalent_groups(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for item in candidates:
        grouped.setdefault(f"{item.action}\n{item.label}", []).append(item)
    return grouped


def decide_action(
    client: OpenJevProClient,
    snapshot: dict[str, Any],
    candidates: list[Candidate],
) -> ActionDecision:
    if not candidates:
        return ActionDecision(None, None, True, "no candidates")
    pool = candidates
    remaining = collapse_equivalent(pool)
    auto = _auto_pick_equivalent(pool, remaining)
    if auto is not None:
        return auto
    for grouping in (group_by_action, group_by_index, group_by_target):
        if len(remaining) <= LETTER_LIMIT:
            break
        groups = grouping(remaining)
        if len(groups) <= 1:
            continue
        picked = _choose_group(client, snapshot, groups)
        if picked.abstained or picked.candidate is None:
            return _fallback_pick(pool, "model abstained while choosing an action group")
        pool = groups.get(picked.candidate.id, remaining)
        remaining = collapse_equivalent(pool)
        auto = _auto_pick_equivalent(pool, remaining)
        if auto is not None:
            return auto
    remaining = collapse_equivalent(pool)
    auto = _auto_pick_equivalent(pool, remaining)
    if auto is not None:
        return auto
    result = _one_shot(client, snapshot, remaining)
    if result.abstained:
        return _fallback_pick(pool, "model abstained")
    return result


def _fallback_pick(candidates: list[Candidate], why: str) -> ActionDecision:
    majority = _majority_pick(candidates)
    if majority is not None:
        return majority
    return _random_pick(candidates, why)


def _random_pick(candidates: list[Candidate], why: str) -> ActionDecision:
    if not candidates:
        return ActionDecision(None, None, True, "no candidates")
    item = random.choice(candidates)
    print(f"undecided: {why}; random legal action {item.id} ({item.label})")
    return ActionDecision(
        item,
        ChoiceDecision(
            value=item.id,
            probabilities={item.id: 1.0 / len(candidates)},
            confidence=0.0,
            abstained=False,
        ),
        False,
        "random",
    )


def _majority_pick(candidates: list[Candidate]) -> ActionDecision | None:
    groups = equivalent_groups(candidates)
    if not groups:
        return None
    largest = max(groups.values(), key=len)
    if len(largest) < 2:
        return None
    return _auto_pick_equivalent(largest, [largest[0]])


def _auto_pick_equivalent(
    original: list[Candidate],
    collapsed: list[Candidate],
) -> ActionDecision | None:
    if len(original) <= 1 or len(collapsed) != 1:
        return None
    item = collapsed[0]
    return ActionDecision(
        item,
        ChoiceDecision(
            value=item.id,
            probabilities={item.id: 1.0},
            confidence=1.0,
            abstained=False,
        ),
        False,
        "equivalent",
    )


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


def _criteria(snapshot: dict[str, Any]) -> str:
    screen = (snapshot.get("state") or {}).get("screen")
    rules = {
        "COMBAT": (
            "Spend energy on playable cards. Prioritize lethal or high incoming intent damage. "
            "Do not end the turn with unused efficient plays. Avoid ending the turn if it kills the player."
        ),
        "REWARD": (
            "Compare offered cards with the current deck. Take a card that clearly improves it; otherwise skip. "
            "Prefer relics and potions that fit the deck."
        ),
        "SHOP": "Check relics and card removal before spending gold. Buy only affordable stocked items that clearly help.",
        "EVENT": "Prefer unlocked options. Avoid options marked KILLS unless no alternative remains.",
        "MAP": (
            "Choose the next node from this character's deck, relics, potions, HP, and gold. "
            "Prefer rest when wounded, a shop when gold can buy something useful, and an elite when the deck is strong."
        ),
        "REST": "Heal when missing substantial HP; smith when HP is comfortable.",
        "CHEST": "Take the relic that best fits the current deck.",
        "CARD_SELECTION": "Pick the card that the prompt is asking to remove, upgrade, or transform.",
    }
    return rules.get(screen, "Pick the single best legal action for this screen.")
