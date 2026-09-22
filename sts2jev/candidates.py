"""Expand a compact decision snapshot into concrete POST /action candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REJECTED_ACTIONS = frozenset(
    {
        "abandon_run",
        "save_and_quit",
        "switch_profile",
        "return_to_main_menu",
        "invite_ai_teammate",
        "continue_ai_teammate",
        "run_console_command",
        "inject_event_churn",
    }
)

PAUSE_SCREENS = frozenset(
    {
        "PAUSE_MENU",
        "SETTINGS",
        "COMPENDIUM",
        "RELIC_COLLECTION",
        "POTION_LAB",
        "BESTIARY",
        "STATS",
        "RUN_HISTORY",
        "FEEDBACK",
    }
)

MAIN_MENU_ACTIONS = frozenset(
    {
        "continue_run",
        "open_character_select",
        "close_main_menu_submenu",
        "open_timeline",
        "choose_timeline_epoch",
        "confirm_timeline_overlay",
        "confirm_unlock",
    }
)

ZERO_ARG_ACTIONS = frozenset(
    {
        "end_turn",
        "proceed",
        "embark",
        "skip_reward_cards",
        "collect_rewards_and_proceed",
        "continue_run",
        "open_character_select",
        "continue_game_over",
        "dismiss_game_over_wait",
        "confirm_unlock",
        "close_main_menu_submenu",
        "confirm_timeline_overlay",
        "confirm_selection",
        "confirm_bundle",
        "open_chest",
        "close_cards_view",
        "open_shop_inventory",
        "close_shop_inventory",
        "remove_card_at_shop",
        "confirm_modal",
        "dismiss_modal",
        "unready",
        "increase_ascension",
        "decrease_ascension",
        "open_timeline",
        "resolve_rewards",
        "ready_multiplayer_lobby",
        "host_multiplayer_lobby",
        "join_multiplayer_lobby",
        "disconnect_multiplayer_lobby",
    }
)


@dataclass(frozen=True)
class Candidate:
    id: str
    action: str
    label: str
    body: dict[str, Any]
    index_key: Any = None
    target_key: Any = None


@dataclass
class ExpandResult:
    candidates: list[Candidate] = field(default_factory=list)
    stop_reason: str | None = None


def action_names(snapshot: dict[str, Any]) -> list[str]:
    raw = snapshot.get("available_actions") or []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    if not names:
        state_names = (snapshot.get("state") or {}).get("available_actions") or []
        names = [str(name) for name in state_names]
    return names


def expand(snapshot: dict[str, Any]) -> ExpandResult:
    from sts2jev.expanders import EXPANDERS, crystal_cells, expand_crystal_cells

    state = snapshot.get("state") or {}
    screen = state.get("screen")
    if screen in PAUSE_SCREENS:
        return ExpandResult(stop_reason=f"paused on {screen}")

    names = [name for name in action_names(snapshot) if name not in REJECTED_ACTIONS]
    if screen in {"MAIN_MENU", "PATCH_NOTES"}:
        names = [name for name in names if name in MAIN_MENU_ACTIONS]

    candidates: list[Candidate] = []
    for name in names:
        if name == "crystal_clear_cell":
            cells = crystal_cells(state)
            if not cells:
                return ExpandResult(
                    stop_reason=(
                        "CRYSTAL_SPHERE: crystal_clear_cell is available but "
                        "compact state has no hidden_cells coordinates"
                    )
                )
            candidates.extend(expand_crystal_cells(cells))
            continue
        expander = EXPANDERS.get(name)
        if expander is not None:
            candidates.extend(expander(state))
            continue
        if name in ZERO_ARG_ACTIONS:
            candidates.append(zero_arg(name))
    return ExpandResult(candidates=candidates)


def item_index(item: dict[str, Any], fallback: int | None = None) -> int:
    if item.get("i") is not None:
        return int(item["i"])
    if item.get("index") is not None:
        return int(item["index"])
    if fallback is not None:
        return fallback
    raise KeyError("item has no compact i or index")


def item_label(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return "?"


def flag(item: dict[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        if key in item and item[key] is not None:
            return bool(item[key])
    return default


def zero_arg(action: str) -> Candidate:
    return Candidate(id=action, action=action, label=action.replace("_", " "), body={"action": action})


def action_body(action: str, **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"action": action}
    for key, value in fields.items():
        if value is not None:
            body[key] = value
    return body


def target_indices(item: dict[str, Any]) -> list[int]:
    raw = item.get("targets")
    if raw is None:
        raw = item.get("valid_target_indices") or []
    return [int(value) for value in raw]
