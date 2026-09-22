"""Candidate expansion and layered Choice tests (no live game)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from openjevpro.schemas import ChoiceDecision

from sts2jev.candidates import Candidate, expand
from sts2jev.decide import decide_action, group_by_action, group_by_index
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
        decision = decide_action(jev, snapshot, result.candidates)
        self.assertTrue(decision.abstained)
        self.assertIsNone(decision.candidate)


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
            run_loop(game, ScriptedJev([]), poll_s=0, abstain_sleep_s=0)
        self.assertEqual(raised.exception.reason, "game over complete")
        self.assertEqual(game.acts[0]["action"], "continue_game_over")

    def test_repeated_abstain_stops(self) -> None:
        game = FakeGame([load_fixture("combat_compact.json")])
        with self.assertRaises(StopPlay) as raised:
            run_loop(
                game,
                ScriptedJev(["UNKNOWN", "UNKNOWN"]),
                poll_s=0,
                abstain_sleep_s=0,
                max_same_abstain=2,
            )
        self.assertIn("repeated abstain", raised.exception.reason)


if __name__ == "__main__":
    unittest.main()
