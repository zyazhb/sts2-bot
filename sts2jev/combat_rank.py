"""Drop a basic Strike when a stronger attack is already legal."""

from __future__ import annotations

import re

from sts2jev.candidates import Candidate

_DAMAGE = re.compile(r"Deal (\d+) damage", re.IGNORECASE)
_BLOCK = re.compile(r"Gain (\d+) Block", re.IGNORECASE)
_LOSE_HP = re.compile(r"Lose \d+ HP", re.IGNORECASE)
_UPSIDE = re.compile(r"\b(Draw|Strength|Vulnerable|Weak|Dexterity|Focus|Block)\b", re.IGNORECASE)
_BASIC_STRIKE = frozenset({"Strike", "Strike+"})


def narrow_combat(candidates: list[Candidate]) -> list[Candidate]:
    """Drop a basic Strike when a stronger attack is legal. Leave block for the model."""
    plays = [item for item in candidates if item.action == "play_card"]
    if len(plays) < 2:
        return candidates
    attacks = [item for item in plays if _damage(item) is not None and not _loses_hp(item)]
    drop = {item.id for item in plays if _basic_strike(item) and _dominated_strike(item, attacks)}
    for block in plays:
        if not _pure_block(block) or block.id in drop:
            continue
        if _dominated_block(block, plays):
            drop.add(block.id)
    if not drop:
        return candidates
    kept = [item for item in candidates if item.id not in drop]
    if not any(item.action == "play_card" for item in kept):
        return candidates
    return kept


def _dominated_strike(strike: Candidate, attacks: list[Candidate]) -> bool:
    return any(_better_attack(other, strike) for other in attacks if other.id != strike.id)


def _better_attack(other: Candidate, strike: Candidate) -> bool:
    if not _covers(other, strike) or _loses_hp(other):
        return False
    other_damage = _damage(other)
    strike_damage = _damage(strike)
    if other_damage is None or strike_damage is None:
        return False
    if other_damage > strike_damage:
        return True
    return other_damage >= strike_damage and _upside(other) and not _basic_strike(other)


def _dominated_block(block: Candidate, plays: list[Candidate]) -> bool:
    gained = _block_gain(block) or 0
    for other in plays:
        if other.id == block.id or _loses_hp(other):
            continue
        other_gain = _block_gain(other)
        if other_gain is None:
            continue
        if _pure_block(other) and other_gain > gained:
            return True
        if other_gain >= gained and _damage(other) is not None:
            return True
    return False


def _covers(other: Candidate, strike: Candidate) -> bool:
    if other.target_key is None:
        return True
    return other.target_key == strike.target_key


def _basic_strike(candidate: Candidate) -> bool:
    return _card_name(candidate.label) in _BASIC_STRIKE


def _pure_block(candidate: Candidate) -> bool:
    return _block_gain(candidate) is not None and _damage(candidate) is None


def _upside(candidate: Candidate) -> bool:
    return _UPSIDE.search(_rules(candidate.label)) is not None


def _loses_hp(candidate: Candidate) -> bool:
    return _LOSE_HP.search(candidate.label) is not None


def _damage(candidate: Candidate) -> int | None:
    found = _DAMAGE.search(candidate.label)
    return int(found.group(1)) if found else None


def _block_gain(candidate: Candidate) -> int | None:
    found = _BLOCK.search(candidate.label)
    return int(found.group(1)) if found else None


def _card_name(label: str) -> str:
    text = label.removeprefix("play ").strip()
    return re.split(r"\s+[\[(]", text, maxsplit=1)[0].strip()


def _rules(label: str) -> str:
    body = label.split(" at ", 1)[0]
    if "]: " in body:
        return body.split("]: ", 1)[1]
    return body
