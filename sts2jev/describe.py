"""Text lines for piles, shops, map routes, and monster moves."""

from __future__ import annotations

from typing import Any


def move_catalog_from_items(items: list[Any]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for monster in items:
        if not isinstance(monster, dict):
            continue
        for move in monster.get("moves") or []:
            if not isinstance(move, dict):
                continue
            move_id = str(move.get("id") or "")
            name = str(move.get("name") or move_id)
            if not move_id:
                continue
            catalog[move_id] = name
            catalog[f"{move_id}_MOVE"] = name
    return catalog


def ascension_lines(run: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for effect in run.get("ascension_effects") or []:
        if isinstance(effect, str):
            lines.append(_plain(effect))
            continue
        if not isinstance(effect, dict):
            continue
        name = str(effect.get("name") or effect.get("id") or "?")
        desc = effect.get("description")
        lines.append(f"{name}: {_plain(str(desc))}" if desc not in (None, "") else name)
    return lines


def glossary(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    return {str(key): _plain(str(value)) if isinstance(value, str) else str(value) for key, value in raw.items()}


def pile_lines(pile: Any) -> list[str]:
    lines: list[str] = []
    for item in pile or []:
        if isinstance(item, str):
            lines.append(item)
        elif isinstance(item, dict) and item.get("line"):
            lines.append(str(item["line"]))
    return lines


def sale_lines(items: Any, catalog: dict[str, str]) -> list[str]:
    from sts2jev.view import _lookup_power

    lines: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            lines.append(str(item))
            continue
        line = str(item.get("line") or item.get("name") or "?")
        desc = item.get("description") or item.get("resolved_rules_text") or item.get("rules_text")
        if desc in (None, ""):
            found = _lookup_power(
                catalog,
                str(item.get("relic_id") or ""),
                str(item.get("potion_id") or ""),
                str(item.get("card_id") or ""),
                str(item.get("name") or ""),
            )
            if found and found not in line:
                head = found.split(":", 1)[0]
                desc = found.split(":", 1)[1].strip() if ":" in found and line in head else found
        if desc not in (None, "") and _plain(str(desc)) not in line:
            line = f"{line}: {_plain(str(desc))}"
        price = item.get("price")
        if price is not None and str(price) not in line:
            sale = " sale" if item.get("on_sale") else ""
            line = f"{line} ({price}g{sale})"
        lines.append(line)
    return lines


def map_routes(mapping: dict[str, Any]) -> list[str]:
    by_coord: dict[str, dict[str, Any]] = {}
    for node in mapping.get("nodes") or []:
        if isinstance(node, dict):
            by_coord[_coord_of(node)] = node
    routes: list[str] = []
    for option in mapping.get("options") or mapping.get("available_nodes") or []:
        if not isinstance(option, dict):
            continue
        coord = _coord_of(option)
        children: list[str] = []
        for child in (by_coord.get(coord) or {}).get("children") or []:
            child_coord = child if isinstance(child, str) else _coord_of(child if isinstance(child, dict) else {})
            child_node = by_coord.get(child_coord) or {}
            children.append(f"{child_node.get('node_type') or '?'} {child_coord}".strip())
        if children:
            routes.append(f"{option.get('node_type') or '?'} {coord} -> {', '.join(children)}")
    return routes


def orb_lines(orbs: Any) -> list[str]:
    lines: list[str] = []
    for orb in orbs or []:
        if not isinstance(orb, dict):
            continue
        name = orb.get("name") or orb.get("orb_id")
        if name:
            lines.append(f"{name} passive {orb.get('passive_value')} evoke {orb.get('evoke_value')}")
    return lines


def pet_lines(pets: Any, catalog: dict[str, str]) -> list[str]:
    from sts2jev.view import _hp, _power_lines

    lines: list[str] = []
    for pet in pets or []:
        if not isinstance(pet, dict):
            continue
        name = pet.get("name") or pet.get("pet_id") or "pet"
        hp = _hp(pet)
        powers = _power_lines(pet.get("powers"), catalog)
        text = f"{name} {hp}" if hp else str(name)
        if powers:
            text = f"{text}; {', '.join(powers)}"
        lines.append(text.strip())
    return lines


def _coord_of(item: dict[str, Any]) -> str:
    if item.get("coord"):
        return str(item["coord"])
    if item.get("row") is not None and item.get("col") is not None:
        return f"{item['row']},{item['col']}"
    return ""


def _plain(text: str) -> str:
    from sts2jev.view import _plain as strip_markup

    return strip_markup(text)
