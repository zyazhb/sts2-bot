"""Screen-specific expansion of STS2 compact snapshot fields."""

from __future__ import annotations

from typing import Any, Callable

from sts2jev.candidates import Candidate, action_body, flag, item_index, item_label, target_indices
from sts2jev.view import _plain


def crystal_cells(state: dict[str, Any]) -> list[tuple[int, int]]:
    sphere = state.get("crystal_sphere") or {}
    cells = sphere.get("hidden_cells") or []
    out: list[tuple[int, int]] = []
    for cell in cells:
        if isinstance(cell, (list, tuple)) and len(cell) >= 2:
            out.append((int(cell[0]), int(cell[1])))
    return out


def expand_crystal_cells(cells: list[tuple[int, int]]) -> list[Candidate]:
    return [
        Candidate(
            id=f"crystal_clear_cell:{x},{y}",
            action="crystal_clear_cell",
            label=f"clear crystal cell {x},{y}",
            body=action_body("crystal_clear_cell", x=x, y=y),
            index_key=x,
            target_key=y,
        )
        for x, y in cells
    ]


def indexed(
    action: str,
    items: list[Any],
    *,
    label_keys: tuple[str, ...] = ("name", "line", "title"),
    include: Callable[[dict[str, Any]], bool] | None = None,
    extra: Callable[[dict[str, Any]], str] | None = None,
    index_field: str = "option_index",
) -> list[Candidate]:
    out: list[Candidate] = []
    for offset, raw in enumerate(items):
        item = raw if isinstance(raw, dict) else {"line": str(raw), "i": offset}
        if include is not None and not include(item):
            continue
        option_index = item_index(item, offset)
        label = item_label(item, *label_keys)
        if extra:
            suffix = extra(item)
            if suffix:
                label = f"{label} ({suffix})"
        out.append(
            Candidate(
                id=f"{action}:{option_index}",
                action=action,
                label=f"{action.replace('_', ' ')} {label}",
                body=action_body(action, **{index_field: option_index}),
                index_key=option_index,
            )
        )
    return out


def _play_card(state: dict[str, Any]) -> list[Candidate]:
    combat = state.get("combat") or {}
    enemies = combat.get("enemies") or []
    out: list[Candidate] = []
    for offset, card in enumerate(combat.get("hand") or []):
        if not flag(card, "playable", default=True):
            continue
        card_index = item_index(card, offset)
        name = item_label(card, "name", "line", "card_id")
        cost = card.get("energy_cost")
        cost_bit = f"{cost} energy" if cost is not None else "card"
        if _card_needs_target(card):
            target_ids = target_indices(card) or [
                item_index(enemy, i)
                for i, enemy in enumerate(enemies)
                if flag(enemy, "hittable", "is_hittable", "alive", "is_alive", default=True)
            ]
            for target_index in target_ids:
                enemy = _enemy_by_index(enemies, target_index)
                label = f"play {name} ({cost_bit}) at {_enemy_label(enemy, target_index)}"
                out.append(
                    Candidate(
                        id=f"play_card:{card_index}:target:{target_index}",
                        action="play_card",
                        label=label,
                        body=action_body("play_card", card_index=card_index, target_index=target_index),
                        index_key=card_index,
                        target_key=target_index,
                    )
                )
        else:
            out.append(
                Candidate(
                    id=f"play_card:{card_index}",
                    action="play_card",
                    label=f"play {name} ({cost_bit})",
                    body=action_body("play_card", card_index=card_index),
                    index_key=card_index,
                )
            )
    return out


def _card_needs_target(card: dict[str, Any]) -> bool:
    if flag(card, "requires_target"):
        return True
    if target_indices(card):
        return True
    target = card.get("target") or card.get("target_type")
    return target not in (None, "", "None", "Self", "none")


def _enemy_by_index(enemies: list[dict[str, Any]], target_index: int) -> dict[str, Any]:
    for offset, enemy in enumerate(enemies):
        if item_index(enemy, offset) == target_index:
            return enemy
    return {}


def _enemy_label(enemy: dict[str, Any], target_index: int) -> str:
    name = item_label(enemy, "name", "line", "enemy_id") if enemy else f"#{target_index}"
    hp = enemy.get("hp")
    if hp is None and enemy.get("current_hp") is not None:
        hp = f"{enemy.get('current_hp')}/{enemy.get('max_hp')}"
    intent = _intent_summary(enemy)
    bits = [name if name != "?" else f"enemy {target_index}"]
    if hp:
        bits.append(str(hp))
    if intent:
        bits.append(intent)
    return " ".join(bits) if len(bits) == 1 else f"{bits[0]} ({', '.join(bits[1:])})"


def _intent_summary(enemy: dict[str, Any]) -> str:
    parts: list[str] = []
    for intent in enemy.get("intents") or []:
        total = intent.get("total_damage")
        if total is not None:
            parts.append(f"{total} dmg")
        elif intent.get("label"):
            parts.append(str(intent["label"]))
        elif intent.get("intent_type"):
            parts.append(str(intent["intent_type"]))
    return ", ".join(parts) or str(enemy.get("intent") or "")


def _potion_action(state: dict[str, Any], action: str, ready_keys: tuple[str, ...]) -> list[Candidate]:
    potions = (state.get("run") or {}).get("potions") or []
    enemies = (state.get("combat") or {}).get("enemies") or []
    out: list[Candidate] = []
    verb = "use" if action == "use_potion" else "discard"
    for offset, potion in enumerate(potions):
        if not flag(potion, "occupied", default=True):
            continue
        if not flag(potion, *ready_keys, default=True):
            continue
        option_index = item_index(potion, offset)
        name = _potion_text(potion)
        if action == "use_potion" and _card_needs_target(potion):
            for target_index in target_indices(potion) or [
                item_index(enemy, i)
                for i, enemy in enumerate(enemies)
                if flag(enemy, "hittable", "is_hittable", "alive", "is_alive", default=True)
            ]:
                out.append(
                    Candidate(
                        id=f"{action}:{option_index}:target:{target_index}",
                        action=action,
                        label=f"{verb} {name} at {_enemy_label(_enemy_by_index(enemies, target_index), target_index)}",
                        body=action_body(action, option_index=option_index, target_index=target_index),
                        index_key=option_index,
                        target_key=target_index,
                    )
                )
        else:
            out.append(
                Candidate(
                    id=f"{action}:{option_index}",
                    action=action,
                    label=f"{verb} {name}",
                    body=action_body(action, option_index=option_index),
                    index_key=option_index,
                )
            )
    return out


def _potion_text(potion: dict[str, Any]) -> str:
    name = item_label(potion, "name", "line", "potion_id")
    desc = potion.get("description")
    if desc not in (None, "") and _plain(str(desc)) not in name:
        return f"{name}: {_plain(str(desc))}"
    return name


def _coord(item: dict[str, Any]) -> str:
    if item.get("coord"):
        return str(item["coord"])
    if item.get("row") is not None and item.get("col") is not None:
        return f"r{item['row']}c{item['col']}"
    return ""


def _map_nodes(state: dict[str, Any]) -> list[Candidate]:
    mapping = state.get("map") or {}
    return indexed(
        "choose_map_node",
        mapping.get("options") or mapping.get("available_nodes") or [],
        label_keys=("node_type", "line", "name"),
        extra=_coord,
    )


def _price(item: dict[str, Any]) -> str:
    price = item.get("price")
    sale = " sale" if flag(item, "on_sale") else ""
    return f"{price}g{sale}" if price is not None else sale.strip()


def _shop_items(state: dict[str, Any], action: str, key: str) -> list[Candidate]:
    shop = state.get("shop") or {}
    return indexed(
        action,
        shop.get(key) or [],
        include=lambda item: flag(item, "stocked", "is_stocked", default=True)
        and flag(item, "affordable", "enough_gold", default=True),
        extra=_price,
    )


def _rest_options(state: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for offset, option in enumerate((state.get("rest") or {}).get("options") or []):
        if not flag(option, "enabled", "is_enabled", default=True):
            continue
        option_index = item_index(option, offset)
        name = item_label(option, "option_id", "title", "line")
        if flag(option, "requires_target"):
            for target_index in target_indices(option):
                out.append(
                    Candidate(
                        id=f"choose_rest_option:{option_index}:target:{target_index}",
                        action="choose_rest_option",
                        label=f"rest {name} target {target_index}",
                        body=action_body(
                            "choose_rest_option",
                            option_index=option_index,
                            target_index=target_index,
                        ),
                        index_key=option_index,
                        target_key=target_index,
                    )
                )
        else:
            out.append(
                Candidate(
                    id=f"choose_rest_option:{option_index}",
                    action="choose_rest_option",
                    label=f"rest {name}",
                    body=action_body("choose_rest_option", option_index=option_index),
                    index_key=option_index,
                )
            )
    return out


def _event_options(state: dict[str, Any]) -> list[Candidate]:
    return indexed(
        "choose_event_option",
        (state.get("event") or {}).get("options") or [],
        label_keys=("title", "line", "text_key"),
        include=lambda item: not flag(item, "locked", "is_locked"),
        extra=lambda item: "KILLS" if flag(item, "kill", "will_kill_player") else "",
    )


def _crystal_tools(_state: dict[str, Any]) -> list[Candidate]:
    return [
        Candidate(
            id=f"crystal_set_tool:{tool}",
            action="crystal_set_tool",
            label=f"set crystal tool {tool}",
            body=action_body("crystal_set_tool", tool=tool),
            index_key=tool,
        )
        for tool in ("big", "small")
    ]


def _potion_slots_full(state: dict[str, Any]) -> bool:
    potions = (state.get("run") or {}).get("potions")
    if not isinstance(potions, list) or not potions:
        return False
    slots = [item for item in potions if isinstance(item, dict)]
    if not slots:
        return False
    return all(_slot_occupied(item) for item in slots)


def _slot_occupied(item: dict[str, Any]) -> bool:
    if "occupied" in item and item["occupied"] is not None:
        return bool(item["occupied"])
    return bool(item.get("potion_id") or item.get("name"))


def _is_potion_reward(item: dict[str, Any]) -> bool:
    kind = str(item.get("reward_type") or item.get("type") or "").casefold()
    if kind == "potion":
        return True
    text = str(item.get("description") or item.get("line") or "").casefold()
    return text.startswith("potion:")


def _select_deck_cards(state: dict[str, Any]) -> list[Candidate]:
    selection = state.get("selection") or {}
    selected = selection.get("selected")
    if selected is None:
        selected = selection.get("selected_count")
    maximum = selection.get("max")
    if maximum is None:
        maximum = selection.get("max_select")
    if selected is not None and maximum is not None and int(selected) >= int(maximum):
        return []
    return indexed(
        "select_deck_card",
        selection.get("cards") or [],
        label_keys=("name", "line", "card_id"),
        include=lambda item: not flag(item, "selected"),
    )


def _claim_rewards(state: dict[str, Any]) -> list[Candidate]:
    belt_full = _potion_slots_full(state)
    return indexed(
        "claim_reward",
        (state.get("reward") or {}).get("rewards") or [],
        label_keys=("description", "line", "reward_type"),
        include=lambda item: flag(item, "claimable", default=True)
        and not (belt_full and _is_potion_reward(item)),
    )


def _buy_potion(state: dict[str, Any]) -> list[Candidate]:
    if _potion_slots_full(state):
        return []
    return _shop_items(state, "buy_potion", "potions")


EXPANDERS: dict[str, Callable[[dict[str, Any]], list[Candidate]]] = {
    "play_card": _play_card,
    "use_potion": lambda state: _potion_action(state, "use_potion", ("usable", "can_use")),
    "discard_potion": lambda state: _potion_action(state, "discard_potion", ("discard", "can_discard")),
    "choose_map_node": _map_nodes,
    "claim_reward": _claim_rewards,
    "choose_reward_card": lambda state: indexed(
        "choose_reward_card",
        (state.get("reward") or {}).get("cards") or (state.get("reward") or {}).get("card_options") or [],
        label_keys=("name", "line", "card_id"),
    ),
    "select_deck_card": _select_deck_cards,
    "choose_treasure_relic": lambda state: indexed(
        "choose_treasure_relic",
        (state.get("chest") or {}).get("relics") or (state.get("chest") or {}).get("relic_options") or [],
        label_keys=("name", "line", "relic_id"),
    ),
    "choose_event_option": _event_options,
    "choose_rest_option": _rest_options,
    "buy_card": lambda state: _shop_items(state, "buy_card", "cards"),
    "buy_relic": lambda state: _shop_items(state, "buy_relic", "relics"),
    "buy_potion": _buy_potion,
    "choose_bundle": lambda state: indexed(
        "choose_bundle",
        state.get("bundles") or [],
        label_keys=("line", "name"),
    ),
    "choose_capstone_option": lambda state: indexed(
        "choose_capstone_option",
        (state.get("capstone") or {}).get("options") or [],
        label_keys=("line", "title", "name"),
    ),
    "select_character": lambda state: indexed(
        "select_character",
        (state.get("character_select") or {}).get("characters") or [],
        label_keys=("name", "line", "character_id"),
        include=lambda item: not flag(item, "locked", "is_locked"),
    ),
    "choose_timeline_epoch": lambda state: indexed(
        "choose_timeline_epoch",
        (state.get("timeline") or {}).get("slots") or [],
        label_keys=("line", "name"),
        include=lambda item: flag(item, "actionable", "is_actionable", default=True),
    ),
    "crystal_set_tool": _crystal_tools,
}
