"""Candidate expansion and layered Choice tests (no live game)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from openjevpro.schemas import ChoiceDecision

from sts2jev.candidates import Candidate, expand
from sts2jev.decide import _slice_state, collapse_equivalent, decide_action, group_by_action, group_by_index
from sts2jev.loop import StopPlay, _correct_once, _should_wait_combat, run_loop


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class ScriptedJev:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    def decide_choice(self, state, candidates, criteria="", allow_abstain=True):
        options = list(candidates)
        self.calls.append(options)
        value = self.answers.pop(0)
        if value not in options and value != "UNKNOWN":
            raise AssertionError(f"{value!r} not in {options}")
        return ChoiceDecision(
            value=value,
            probabilities={value: 0.9},
            confidence=0.9,
            abstained=value == "UNKNOWN",
        )


class ExpandTests(unittest.TestCase):
    def test_combat_compact_keys(self) -> None:
        result = expand(load_fixture("combat_compact.json"))
        ids = {item.id for item in result.candidates}
        self.assertIsNone(result.stop_reason)
        self.assertIn("play_card:0:target:0", ids)
        self.assertIn("play_card:1", ids)
        self.assertIn("end_turn", ids)
        self.assertNotIn("play_card:2:target:0", ids)
        strike = next(item for item in result.candidates if item.id == "play_card:0:target:0")
        self.assertEqual(strike.body, {"action": "play_card", "card_index": 0, "target_index": 0})
        self.assertIn("7 dmg", strike.label)

    def test_compact_target_array_expands_enemies(self) -> None:
        result = expand(
            {
                "available_actions": ["play_card", "end_turn"],
                "state": {
                    "screen": "COMBAT",
                    "combat": {
                        "hand": [
                            {
                                "i": 0,
                                "line": "Strike [1 Energy]: Deal 6 damage.",
                                "playable": True,
                                "energy_cost": 1,
                                "target": "enemies",
                                "targets": [0, 1],
                            }
                        ],
                        "enemies": [
                            {"i": 0, "name": "Slime A", "hp": "8/8", "alive": True, "hittable": True},
                            {"i": 1, "name": "Slime B", "hp": "13/13", "alive": True, "hittable": True},
                        ],
                    },
                },
            }
        )
        ids = [item.id for item in result.candidates]
        self.assertEqual(ids, ["play_card:0:target:0", "play_card:0:target:1", "end_turn"])

    def test_shop_filters_unaffordable_and_unstocked(self) -> None:
        result = expand(load_fixture("shop.json"))
        ids = [item.id for item in result.candidates]
        self.assertEqual(ids, ["buy_card:0", "close_shop_inventory"])
        self.assertIn("45g", result.candidates[0].label)

    def test_event_skips_locked_and_marks_kill(self) -> None:
        result = expand(load_fixture("event.json"))
        ids = [item.id for item in result.candidates]
        self.assertEqual(ids, ["choose_event_option:0", "choose_event_option:1"])
        kill = result.candidates[1]
        self.assertIn("KILLS", kill.label)

    def test_crystal_missing_cells_stops(self) -> None:
        result = expand(load_fixture("crystal_missing.json"))
        self.assertEqual(result.candidates, [])
        self.assertIn("hidden_cells", result.stop_reason or "")

    def test_main_menu_rejects_listed_actions(self) -> None:
        result = expand(load_fixture("main_menu.json"))
        ids = {item.id for item in result.candidates}
        self.assertEqual(ids, {"continue_run", "open_character_select"})

    def test_pause_screen_stops(self) -> None:
        result = expand(
            {
                "available_actions": ["close_main_menu_submenu"],
                "state": {"screen": "PAUSE_MENU"},
            }
        )
        self.assertEqual(result.stop_reason, "paused on PAUSE_MENU")

    def test_modal_confirm_is_a_candidate(self) -> None:
        result = expand(
            {
                "available_actions": ["confirm_modal"],
                "state": {"screen": "MODAL", "modal": {"confirm": True}},
            }
        )
        self.assertEqual([item.id for item in result.candidates], ["confirm_modal"])

    def test_card_reward_overlay_does_not_claim_row(self) -> None:
        result = expand(
            {
                "available_actions": [
                    "claim_reward",
                    "skip_reward_cards",
                    "choose_reward_card",
                ],
                "state": {
                    "screen": "REWARD",
                    "reward": {
                        "pending_card_choice": True,
                        "cards": [{"i": 0, "name": "Pommel Strike"}],
                        "rewards": [{"i": 0, "reward_type": "Card", "description": "Add a card", "claimable": True}],
                    },
                },
            }
        )
        ids = [item.id for item in result.candidates]
        self.assertEqual(ids, ["skip_reward_cards", "choose_reward_card:0"])
        self.assertNotIn("claim_reward:0", ids)

    def test_combat_slice_includes_hp_powers_and_intents(self) -> None:
        sliced = _slice_state(
            {
                "state": {
                    "screen": "COMBAT",
                    "combat": {
                        "player": {
                            "hp": "40/80",
                            "energy": 3,
                            "block": 5,
                            "powers": [
                                {"name": "Strength", "amount": 2},
                                "Vulnerable 1 [debuff]",
                            ],
                        },
                        "enemies": [
                            {
                                "name": "Cultist",
                                "current_hp": 48,
                                "max_hp": 50,
                                "block": 0,
                                "powers": [{"power_id": "RITUAL", "amount": 3}],
                                "intents": [
                                    {
                                        "intent_type": "Attack",
                                        "damage": 6,
                                        "hits": 2,
                                        "total_damage": 12,
                                    }
                                ],
                            }
                        ],
                        "hand": [],
                    },
                }
            },
            [],
        )
        self.assertEqual(sliced["hp"], "40/80")
        self.assertEqual(sliced["powers"], ["Strength 2", "Vulnerable 1 [debuff]"])
        enemy = sliced["enemies"][0]
        self.assertEqual(enemy["hp"], "48/50")
        self.assertEqual(enemy["powers"], ["RITUAL 3"])
        self.assertEqual(enemy["intents"], ["Attack (6x2, 12 dmg)"])

    def test_reward_choice_includes_current_deck(self) -> None:
        sliced = _slice_state(
            {
                "state": {
                    "screen": "REWARD",
                    "run": {
                        "deck": [
                            {"line": "Strike [1 Energy]: Deal 6 damage. *5"},
                            {"name": "Bash", "upgraded": True},
                            "Defend *4",
                        ]
                    },
                    "reward": {
                        "pending_card_choice": True,
                        "cards": [{"i": 0, "line": "Pommel Strike [1 Energy]: Deal 9 damage. Draw 1 card."}],
                    },
                }
            },
            [],
        )
        self.assertEqual(
            sliced["deck"],
            [
                "Strike [1 Energy]: Deal 6 damage. *5",
                "Bash+",
                "Defend *4",
            ],
        )
        self.assertEqual(sliced["reward"]["cards"][0]["line"].split()[0], "Pommel")

    def test_selected_card_is_not_offered_again(self) -> None:
        result = expand(
            {
                "available_actions": ["select_deck_card", "confirm_selection"],
                "state": {
                    "screen": "CARD_SELECTION",
                    "selection": {
                        "prompt": "Choose up to 2 cards to put into your Hand.",
                        "min": 0,
                        "max": 2,
                        "selected": 1,
                        "confirm": True,
                        "cards": [
                            {"i": 0, "line": "Strike [1 Energy]: Deal 6 damage.", "selected": True},
                        ],
                    },
                },
            }
        )
        self.assertEqual([item.id for item in result.candidates], ["confirm_selection"])

    def test_blocked_claim_is_omitted(self) -> None:
        result = expand(
            {
                "available_actions": ["claim_reward", "collect_rewards_and_proceed"],
                "state": {
                    "screen": "REWARD",
                    "reward": {
                        "rewards": [
                            {"i": 0, "line": "Card: Add a card to your deck.", "claimable": True},
                            {"i": 1, "line": "Gold: 13 Gold", "claimable": True},
                        ]
                    },
                },
            },
            blocked_ids=frozenset({"claim_reward:0"}),
        )
        self.assertEqual(
            [item.id for item in result.candidates],
            ["claim_reward:1", "collect_rewards_and_proceed"],
        )

    def test_full_potion_belt_skips_potion_rewards(self) -> None:
        result = expand(
            {
                "available_actions": ["claim_reward", "collect_rewards_and_proceed"],
                "state": {
                    "screen": "REWARD",
                    "run": {
                        "potions": [
                            {"i": 0, "name": "Fire Potion", "occupied": True},
                            {"i": 1, "name": "Block Potion", "occupied": True},
                            {"i": 2, "name": "Weak Potion", "occupied": True},
                        ]
                    },
                    "reward": {
                        "rewards": [
                            {"i": 0, "reward_type": "Potion", "description": "Potion: Weak Potion", "claimable": True},
                            {"i": 1, "reward_type": "Gold", "description": "Gold: 25", "claimable": True},
                            {"i": 2, "description": "Potion: Explosive Ampoule", "claimable": True},
                        ]
                    },
                },
            }
        )
        ids = [item.id for item in result.candidates]
        self.assertEqual(ids, ["claim_reward:1", "collect_rewards_and_proceed"])

    def test_open_potion_slot_keeps_potion_reward(self) -> None:
        result = expand(
            {
                "available_actions": ["claim_reward"],
                "state": {
                    "screen": "REWARD",
                    "run": {
                        "potions": [
                            {"i": 0, "name": "Fire Potion", "occupied": True},
                            {"i": 1, "potion_id": None, "occupied": False},
                        ]
                    },
                    "reward": {
                        "rewards": [
                            {"i": 0, "reward_type": "Potion", "description": "Potion: Weak Potion", "claimable": True},
                        ]
                    },
                },
            }
        )
        self.assertEqual([item.id for item in result.candidates], ["claim_reward:0"])

    def test_full_potion_belt_skips_shop_potions(self) -> None:
        result = expand(
            {
                "available_actions": ["buy_potion", "close_shop_inventory"],
                "state": {
                    "screen": "SHOP",
                    "run": {"potions": [{"i": 0, "name": "Fire Potion", "occupied": True}]},
                    "shop": {
                        "potions": [
                            {"i": 0, "name": "Weak Potion", "price": 50, "stocked": True, "affordable": True},
                        ]
                    },
                },
            }
        )
        self.assertEqual([item.id for item in result.candidates], ["close_shop_inventory"])


class LayerTests(unittest.TestCase):
    def _layered_snapshot(self) -> dict[str, Any]:
        snapshot = load_fixture("layered_combat.json")
        snapshot["state"]["combat"]["hand"] = [
            {
                "i": index,
                "name": f"Strike {index}",
                "energy_cost": 1,
                "playable": True,
                "requires_target": True,
                "targets": [0, 1, 2],
            }
            for index in range(10)
        ]
        return snapshot

    def test_over_letter_limit_splits_by_action_then_index(self) -> None:
        result = expand(self._layered_snapshot())
        self.assertGreater(len(result.candidates), 26)
        actions = group_by_action(result.candidates)
        self.assertEqual(set(actions), {"play_card", "end_turn", "use_potion"})
        self.assertEqual(len(actions["play_card"]), 30)
        self.assertEqual(len(group_by_index(actions["play_card"])), 10)

    def test_layered_decide_picks_card_then_target(self) -> None:
        snapshot = self._layered_snapshot()
        result = expand(snapshot)
        jev = ScriptedJev(["play_card", "2", "play_card:2:target:1"])
        decision = decide_action(jev, snapshot, result.candidates)
        self.assertFalse(decision.abstained)
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.id, "play_card:2:target:1")
        self.assertEqual(decision.candidate.body["card_index"], 2)
        self.assertEqual(decision.candidate.body["target_index"], 1)
        self.assertEqual(len(jev.calls[0]), 3)
        self.assertEqual(len(jev.calls[1]), 10)
        self.assertEqual(len(jev.calls[2]), 3)

    def test_abstain_on_first_layer(self) -> None:
        snapshot = self._layered_snapshot()
        result = expand(snapshot)
        jev = ScriptedJev(["UNKNOWN"])
        with patch("sts2jev.decide.random.choice", lambda items: items[0]):
            decision = decide_action(jev, snapshot, result.candidates)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.reason, "random")
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.id, result.candidates[0].id)

    def test_collapse_identical_labels_keeps_first(self) -> None:
        cards = [
            Candidate(
                id=f"select_deck_card:{index}",
                action="select_deck_card",
                label=label,
                body={"action": "select_deck_card", "option_index": index},
                index_key=index,
            )
            for index, label in enumerate(
                ["Strike"] * 5 + ["Defend"] * 4 + ["Bash"]
            )
        ]
        unique = collapse_equivalent(cards)
        self.assertEqual([item.id for item in unique], [
            "select_deck_card:0",
            "select_deck_card:5",
            "select_deck_card:9",
        ])
        jev = ScriptedJev(["select_deck_card:0"])
        decision = decide_action(jev, {"state": {"screen": "CARD_SELECTION"}}, cards)
        self.assertFalse(decision.abstained)
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.id, "select_deck_card:0")
        self.assertEqual(len(jev.calls[0]), 3)

    def test_all_equivalent_picks_without_model(self) -> None:
        cards = [
            Candidate(
                id=f"select_deck_card:{index}",
                action="select_deck_card",
                label="Strike",
                body={"action": "select_deck_card", "option_index": index},
                index_key=index,
            )
            for index in range(5)
        ]
        jev = ScriptedJev([])
        decision = decide_action(jev, {"state": {"screen": "CARD_SELECTION"}}, cards)
        self.assertEqual(decision.reason, "equivalent")
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.id, "select_deck_card:0")
        self.assertEqual(jev.calls, [])

    def test_abstain_falls_back_to_majority_duplicate(self) -> None:
        cards = [
            Candidate(
                id=f"select_deck_card:{index}",
                action="select_deck_card",
                label=label,
                body={"action": "select_deck_card", "option_index": index},
                index_key=index,
            )
            for index, label in enumerate(["Strike"] * 5 + ["Defend"] * 4 + ["Bash"])
        ]
        jev = ScriptedJev(["UNKNOWN"])
        decision = decide_action(jev, {"state": {"screen": "CARD_SELECTION"}}, cards)
        self.assertEqual(decision.reason, "equivalent")
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.id, "select_deck_card:0")


class FakeGame:
    def __init__(self, snapshots: list[dict[str, Any]], act_results: list[dict[str, Any]] | None = None) -> None:
        self.snapshots = list(snapshots)
        self.act_results = list(act_results or [])
        self.acts: list[dict[str, Any]] = []

    def connect(self) -> dict[str, Any]:
        return {"service": "sts2-ai-agent", "api_port": 8080, "instance_role": "human"}

    def snapshot(self) -> dict[str, Any]:
        if not self.snapshots:
            raise AssertionError("snapshot exhausted")
        if len(self.snapshots) == 1:
            return self.snapshots[0]
        return self.snapshots.pop(0)

    def act(self, body: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.acts.append(body)
        if self.act_results:
            return self.act_results.pop(0)
        return {"status": "completed"}


class LoopTests(unittest.TestCase):
    def test_correct_once_rewrites_index(self) -> None:
        candidate = Candidate(
            id="play_card:9",
            action="play_card",
            label="play",
            body={"action": "play_card", "card_index": 9},
        )
        fixed = _correct_once(candidate, {"field": "card_index", "valid_indices": [0, 1]})
        assert fixed is not None
        self.assertEqual(fixed.body["card_index"], 0)

    def test_wait_combat_except_modal(self) -> None:
        waiting = {
            "screen": "COMBAT",
            "combat": {"action_readiness": {"can_use_combat_actions": False, "reason": "hand_in_card_play"}},
        }
        modal = {
            "screen": "COMBAT",
            "combat": {"action_readiness": {"can_use_combat_actions": False, "reason": "modal_open"}},
        }
        self.assertTrue(_should_wait_combat(waiting))
        self.assertFalse(_should_wait_combat(modal))

    def test_game_over_continues_then_stops(self) -> None:
        intro = {
            "available_actions": ["continue_game_over", "return_to_main_menu"],
            "state": {"screen": "GAME_OVER", "game_over": {"phase": "intro"}},
        }
        done = {
            "available_actions": ["return_to_main_menu"],
            "state": {"screen": "GAME_OVER", "game_over": {"phase": "summary_ready"}},
        }
        game = FakeGame([intro, done])
        with self.assertRaises(StopPlay) as raised:
            run_loop(game, ScriptedJev([]), poll_s=0)
        self.assertEqual(raised.exception.reason, "game over complete")
        self.assertEqual(game.acts[0]["action"], "continue_game_over")

    def test_skip_disables_the_claim_that_opened_the_picker(self) -> None:
        reward = {
            "available_actions": ["claim_reward", "collect_rewards_and_proceed"],
            "state": {
                "screen": "REWARD",
                "reward": {
                    "rewards": [
                        {"i": 0, "line": "Card: Add a card to your deck.", "claimable": True},
                        {"i": 1, "line": "Gold: 13 Gold", "claimable": True},
                    ]
                },
            },
        }
        picker = {
            "available_actions": ["skip_reward_cards", "choose_reward_card"],
            "state": {
                "screen": "REWARD",
                "reward": {
                    "pending_card_choice": True,
                    "cards": [{"i": 0, "name": "Strike"}],
                },
            },
        }
        paused = {"available_actions": [], "state": {"screen": "PAUSE_MENU"}}
        game = FakeGame([reward, picker, reward, paused])
        jev = ScriptedJev(["claim_reward:0", "skip_reward_cards", "claim_reward:1"])
        with self.assertRaises(StopPlay):
            run_loop(game, jev, poll_s=0)
        self.assertNotIn("claim_reward:0", jev.calls[2])
        self.assertIn("claim_reward:1", jev.calls[2])
        self.assertEqual(game.acts[2]["option_index"], 1)

    def test_abstain_plays_random_legal_action(self) -> None:
        snapshot = load_fixture("combat_compact.json")
        result = expand(snapshot)
        jev = ScriptedJev(["UNKNOWN"])
        with patch("sts2jev.decide.random.choice", lambda items: items[-1]):
            decision = decide_action(jev, snapshot, result.candidates)
        self.assertEqual(decision.reason, "random")
        assert decision.candidate is not None
        self.assertEqual(decision.candidate.id, result.candidates[-1].id)


if __name__ == "__main__":
    unittest.main()
