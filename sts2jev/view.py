"""Compact slices of an STS2 snapshot for a Choice prompt."""

from __future__ import annotations

import re
from typing import Any

from sts2jev.candidates import Candidate

_MARKUP = re.compile(r"\[/?[a-zA-Z]+\]")


def slice_state(snapshot: dict[str, Any], candidates: list[Candidate]) -> dict[str, Any]:
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
    catalog = snapshot.get("power_catalog") or {}
    if screen == "COMBAT":
        sliced["energy"] = player.get("energy")
        sliced["block"] = player.get("block")
        sliced["powers"] = _power_lines(player.get("powers"), catalog)
        sliced["end_turn_will_kill"] = combat.get("end_turn_will_kill_player")
        sliced["enemies"] = [_enemy_slice(enemy, catalog) for enemy in combat.get("enemies") or []]
        sliced["hand"] = [_hand_slice(card) for card in combat.get("hand") or []]
    elif screen == "MAP":
        mapping = state.get("map") or {}
        sliced["nodes"] = mapping.get("options") or mapping.get("available_nodes") or []
        sliced["character"] = run.get("character") or run.get("character_name")
        sliced["floor"] = run.get("floor")
        sliced["deck"] = _deck_lines(run)
        sliced["relics"] = _relic_lines(run)
        sliced["potions"] = _potion_lines(run)
    elif screen == "SHOP":
        shop = state.get("shop") or {}
        sliced["shop_open"] = shop.get("open", shop.get("is_open"))
        sliced["cards"] = shop.get("cards") or []
        sliced["relics"] = shop.get("relics") or []
        sliced["potions"] = shop.get("potions") or []
        sliced["deck"] = _deck_lines(run)
    elif screen == "REWARD":
        sliced["reward"] = state.get("reward") or {}
        sliced["deck"] = _deck_lines(run)
    elif screen == "CARD_SELECTION":
        sliced["selection"] = state.get("selection") or {}
        sliced["deck"] = _deck_lines(run)
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


def power_catalog_from_items(items: list[Any]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        power_id = str(item.get("id") or "")
        name = str(item.get("name") or power_id)
        description = _plain(str(item.get("description") or ""))
        text = f"{name}: {description}" if description else name
        if power_id:
            catalog[power_id] = text
        if name:
            catalog[name.casefold()] = text
    return catalog


def _enemy_slice(enemy: dict[str, Any], catalog: dict[str, str] | None = None) -> dict[str, Any]:
    raw_intents = enemy.get("intents")
    if not raw_intents and enemy.get("intent"):
        raw_intents = [enemy.get("intent")]
    return {
        "name": enemy.get("name") or enemy.get("line") or enemy.get("enemy_id"),
        "hp": _hp(enemy),
        "block": enemy.get("block"),
        "powers": _power_lines(enemy.get("powers"), catalog),
        "intents": [_intent_line(intent) for intent in raw_intents or []],
    }


def _power_lines(powers: Any, catalog: dict[str, str] | None = None) -> list[str]:
    known = catalog or {}
    return [_power_text(power, known) for power in powers or []]


def _power_text(power: Any, catalog: dict[str, str]) -> str:
    if isinstance(power, str):
        ident, amount, debuff = _split_power_token(power)
        described = _lookup_power(catalog, ident)
        if not described:
            return power
        return _with_amount(described, amount, debuff)
    if not isinstance(power, dict):
        return str(power)
    power_id = str(power.get("power_id") or power.get("id") or "")
    name = str(power.get("name") or "")
    amount = power.get("amount")
    debuff = bool(power.get("is_debuff"))
    own = power.get("description")
    if own not in (None, ""):
        label = name or power_id or "?"
        described = f"{label}: {_plain(str(own))}"
    else:
        described = _lookup_power(catalog, power_id, name, str(power.get("line") or ""))
    if described:
        return _with_amount(described, amount, debuff)
    label = name or power_id or str(power.get("line") or "?")
    text = f"{label} {amount}" if amount is not None else label
    if debuff:
        text = f"{text} [debuff]"
    return text


def _lookup_power(catalog: dict[str, str], *keys: str) -> str:
    for key in keys:
        if not key:
            continue
        found = catalog.get(key) or catalog.get(key.casefold())
        if found:
            return found
    return ""


def _with_amount(described: str, amount: Any, debuff: bool) -> str:
    text = described
    if amount is not None:
        head, sep, tail = described.partition(":")
        if sep and str(amount) not in head.split():
            text = f"{head} {amount}:{tail}"
        elif not sep:
            text = f"{described} {amount}"
    if debuff and "[debuff]" not in text:
        text = f"{text} [debuff]"
    return text


def _split_power_token(text: str) -> tuple[str, str | None, bool]:
    debuff = "[debuff]" in text.casefold()
    cleaned = text.replace("[debuff]", "").replace("[Debuff]", "").strip()
    parts = cleaned.split()
    amount = parts[-1] if parts and parts[-1].lstrip("-").isdigit() else None
    ident = " ".join(parts[:-1]) if amount else cleaned
    return ident, amount, debuff


def _plain(text: str) -> str:
    return _MARKUP.sub("", text).strip()


def _intent_line(intent: Any) -> str:
    if not isinstance(intent, dict):
        return str(intent)
    kind = str(intent.get("intent_type") or intent.get("label") or "intent")
    bits: list[str] = []
    if intent.get("total_damage") is not None:
        hits = intent.get("hits")
        damage = intent.get("damage")
        if hits not in (None, 1) and damage is not None:
            bits.append(f"{damage}x{hits}")
        bits.append(f"{intent['total_damage']} dmg")
    elif intent.get("label"):
        bits.append(str(intent["label"]))
    if intent.get("status_card_count") is not None:
        bits.append(f"{intent['status_card_count']} status")
    return f"{kind} ({', '.join(bits)})" if bits else kind


def _deck_lines(run: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for card in run.get("deck") or []:
        if isinstance(card, str):
            lines.append(card)
            continue
        if not isinstance(card, dict):
            continue
        line = card.get("line")
        if line not in (None, ""):
            lines.append(str(line))
            continue
        name = str(card.get("name") or card.get("card_id") or "?")
        if card.get("upgraded"):
            name = f"{name}+"
        lines.append(name)
    return lines


def _relic_lines(run: dict[str, Any]) -> list[str]:
    relics = run.get("relics") or []
    descriptions = run.get("relic_descriptions") or []
    stacks = run.get("relic_stacks") or []
    lines: list[str] = []
    for index, relic in enumerate(relics):
        if isinstance(relic, str):
            name, desc, stack = relic, _at(descriptions, index), _at(stacks, index)
        elif isinstance(relic, dict):
            name = str(relic.get("name") or relic.get("line") or relic.get("relic_id") or "?")
            desc, stack = relic.get("description"), relic.get("stack")
        else:
            continue
        text = str(name)
        if stack not in (None, ""):
            text = f"{text} x{stack}"
        if desc not in (None, "") and str(desc) != text:
            text = f"{text}: {desc}"
        lines.append(text)
    return lines


def _potion_lines(run: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for potion in run.get("potions") or []:
        if isinstance(potion, str):
            lines.append(potion)
            continue
        if not isinstance(potion, dict) or ("occupied" in potion and not potion["occupied"]):
            continue
        line = potion.get("line")
        if line not in (None, ""):
            lines.append(str(line))
            continue
        name = potion.get("name") or potion.get("potion_id")
        if not name:
            continue
        desc = potion.get("description")
        lines.append(f"{name}: {desc}" if desc not in (None, "") else str(name))
    return lines


def _at(items: list[Any], index: int) -> Any:
    return items[index] if index < len(items) else None


def _hand_slice(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": card.get("name") or card.get("line") or card.get("card_id"),
        "energy_cost": card.get("energy_cost"),
        "playable": card.get("playable"),
        "requires_target": card.get("requires_target"),
    }
