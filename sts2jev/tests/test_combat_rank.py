"""Combat payoff filtering: attack over a smaller block, better attack over Strike."""

from __future__ import annotations

import unittest

from typing import Any

from openjevpro.schemas import ChoiceDecision

from sts2jev.candidates import expand
from sts2jev.combat_rank import narrow_combat
from sts2jev.loop import StopPlay, run_loop


def _combat(hand: list[dict], enemies: list[dict], *, hp: str = "70/80", block: int = 0) -> dict:
    return {
        "available_actions": ["play_card", "end_turn"],
        "state": {
            "screen": "COMBAT",
            "combat": {
                "player": {"hp": hp, "energy": 3, "block": block},
                "hand": hand,
                "enemies": enemies,
            },
        },
    }


def _strike(index: int, damage: int, *, upgraded: bool = False) -> dict:
    name = "Strike+" if upgraded else "Strike"
    return {
        "i": index,
        "line": f"{name} [1 Energy]: Deal {damage} damage.",
        "energy_cost": 1,
        "playable": True,
        "target": "enemies",
        "targets": [0],
    }


def _defend(index: int, block: int) -> dict:
    return {
        "i": index,
        "line": f"Defend [1 Energy]: Gain {block} Block.",
        "energy_cost": 1,
        "playable": True,
    }


def _enemy(*, hp: str = "40/40", damage: int = 10, block: int = 0, powers: list | None = None) -> dict:
    enemy = {
        "i": 0,
        "name": "Cultist",
        "hp": hp,
        "block": block,
        "alive": True,
        "hittable": True,
        "intents": [{"intent_type": "Attack", "total_damage": damage}],
    }
    if powers:
        enemy["powers"] = powers
    return enemy


class _ScriptedJev:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    def decide_choice(self, state: dict[str, Any], candidates: list[str], criteria: str = "", allow_abstain: bool = True) -> ChoiceDecision:
        del state, criteria, allow_abstain
        self.calls.append(list(candidates))
        value = self.answers.pop(0)
        return ChoiceDecision(value=value, probabilities={value: 0.9}, confidence=0.9, abstained=False)


class _FakeGame:
    def __init__(self, snapshots: list[dict[str, Any]]) -> None:
        self.snapshots = list(snapshots)
        self.acts: list[dict[str, Any]] = []

    def connect(self) -> dict[str, Any]:
        return {"service": "sts2-ai-agent", "api_port": 8080, "instance_role": "human"}

    def snapshot(self) -> dict[str, Any]:
        return self.snapshots[0]

    def act(self, body: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        del timeout
        self.acts.append(body)
        if len(self.snapshots) > 1:
            self.snapshots.pop(0)
        return {"status": "completed"}


def _ids(snapshot: dict) -> set[str]:
    return {item.id for item in narrow_combat(expand(snapshot).candidates)}


class NarrowCombatTests(unittest.TestCase):
    def test_better_attack_replaces_basic_strike(self) -> None:
        hand = [
            _strike(0, 6),
            {
                "i": 1,
                "line": "Pommel Strike [1 Energy]: Deal 9 damage. Draw 1 card.",
                "energy_cost": 1,
                "playable": True,
                "target": "enemies",
                "targets": [0],
            },
        ]
        ids = _ids(_combat(hand, [_enemy()]))
        self.assertIn("play_card:1:target:0", ids)
        self.assertNotIn("play_card:0:target:0", ids)

    def test_setup_strike_replaces_basic_strike(self) -> None:
        hand = [
            _strike(0, 6),
            {
                "i": 1,
                "line": "Setup Strike [1 Energy]: Deal 6 damage. Gain 3 Strength this turn.",
                "energy_cost": 1,
                "playable": True,
                "target": "enemies",
                "targets": [0],
            },
        ]
        ids = _ids(_combat(hand, [_enemy()]))
        self.assertIn("play_card:1:target:0", ids)
        self.assertNotIn("play_card:0:target:0", ids)

    def test_higher_damage_strike_replaces_the_basic_one(self) -> None:
        ids = _ids(_combat([_strike(0, 6), _strike(1, 10, upgraded=True)], [_enemy()]))
        self.assertIn("play_card:1:target:0", ids)
        self.assertNotIn("play_card:0:target:0", ids)

    def test_block_stays_for_the_model(self) -> None:
        ids = _ids(_combat([_strike(0, 9), _defend(1, 5)], [_enemy(hp="8/40", damage=14)], hp="70/80"))
        self.assertIn("play_card:0:target:0", ids)
        self.assertIn("play_card:1", ids)

    def test_iron_wave_replaces_an_equal_defend(self) -> None:
        hand = [
            _defend(0, 5),
            {
                "i": 1,
                "line": "Iron Wave [1 Energy]: Gain 5 Block. Deal 5 damage.",
                "energy_cost": 1,
                "playable": True,
                "target": "enemies",
                "targets": [0],
            },
        ]
        ids = _ids(_combat(hand, [_enemy(damage=14)]))
        self.assertIn("play_card:1:target:0", ids)
        self.assertNotIn("play_card:0", ids)

    def test_loop_does_not_offer_the_basic_strike(self) -> None:
        snapshot = _combat(
            [
                _strike(0, 6),
                {
                    "i": 1,
                    "line": "Pommel Strike [1 Energy]: Deal 9 damage. Draw 1 card.",
                    "energy_cost": 1,
                    "playable": True,
                    "target": "enemies",
                    "targets": [0],
                },
            ],
            [_enemy()],
        )
        paused = {"available_actions": [], "state": {"screen": "PAUSE_MENU"}}
        game = _FakeGame([snapshot, paused])
        jev = _ScriptedJev(["play_card:1:target:0"])
        with self.assertRaises(StopPlay):
            run_loop(game, jev, poll_s=0)
        self.assertNotIn("play_card:0:target:0", jev.calls[0])
        self.assertEqual(game.acts[0]["card_index"], 1)


if __name__ == "__main__":
    unittest.main()
