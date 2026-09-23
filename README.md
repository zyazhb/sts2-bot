# sts2-bot

OpenJevPro asks an open-weight language model to pick one option from a closed set, then returns that choice with a probability for every option. `sts2jev` uses those choices to play [Slay the Spire 2](https://www.megacrit.com/) through the game's local AI-agent HTTP API, one legal action at a time.

The model never writes the action. It only chooses among candidates the game already allows.

## Packages

**`openjevpro`** is the decision client. `decide_choice` sends the current state, a short criterion, and a lettered list of options (`A`, `B`, `C`, …) to an OpenAI-compatible server. It reads the logprobs of that single token, turns them into a distribution with temperature scaling, and abstains when the top option is `UNKNOWN` or the confidence is below a threshold. `decide_noul` is the same call restricted to `TRUE` / `FALSE`.

The client talks to three backends:

| Endpoint | Backend | How the choice is scored |
| :--- | :--- | :--- |
| `:1234` (LM Studio) | `lmstudio` | One chat token, reasoning off, letter logprobs |
| `:11434` (Ollama) | `ollama` | JSON scores for each option, then the same calibration |
| anything else | `openai` | One completion token and its logprobs (vLLM, SGLang) |

`TypeSafeJevClient` in `sts2jev` implements the same `decide_choice` shape against the [TypeSafe](https://docs.typesafe.ai/api) Jev API, so the game loop can use either engine.

**`sts2jev`** is the player. It polls the STS2 AI Agent mod on loopback (`service` name `sts2-ai-agent`), expands the snapshot into concrete `POST /action` bodies, and submits the one the model picked.

## How a turn is chosen

1. `Sts2Client` reads `/decision-snapshot`: the screen, the run, and the actions the mod will accept.
2. `expand` turns that snapshot into candidates. Quit, abandon, console, and similar actions are dropped. Pause, settings, and compendium screens stop the loop.
3. Combat is narrowed before the model sees it. A basic Strike is removed when a stronger attack is already legal, and `end_turn` is hidden while a safe card can still be played. An attack into Thorns or Reflect stays paired with ending the turn. If ending the turn is the only remaining action, it is taken without a model call.
4. The prompt is a slice of the snapshot for that screen: HP, energy, hand, intents, deck, relics, shop prices, map routes, and the option labels. Catalog text for powers, relics, potions, cards, and moves is attached when the mod provides it.
5. `decide_action` asks for one letter. More than 26 options are grouped first (by action, then index, then target) and narrowed in passes of at most 26. Identical action-and-label pairs collapse to one candidate. If the model abstains, a random remaining legal action is played.
6. The chosen body is posted to `/action`. A stale index is corrected once from the mod's `valid_indices`. If the screen moved on while the model was thinking, the action is dropped and the loop reads a fresh snapshot.

Screen criteria live in `sts2jev/decide.py`. Combat favors finishing the fight with HP left and prefers damage when it beats block. Rewards, shops, events, the map, rest sites, chests, and card-select screens each have their own one-line rule.

## Run Slay the Spire 2

Python 3.10 or newer. The game needs the STS2 AI Agent mod listening on `http://127.0.0.1:8080`. `examples/sts2_play.py` expects a local model at `http://127.0.0.1:1234/v1` named `gemma-4-e4b-it` (LM Studio). Change those two constants in the example to point at another server.

```bash
uv sync
uv run python examples/sts2_play.py
```

Pass a TypeSafe key to use remote Jev for the same loop:

```bash
uv run python examples/sts2_play.py --jevkey "$JEV_API_KEY"
```

The loop prints each action it submits, then stops on a pause screen, a game-over summary it cannot continue, or a screen with no legal candidate.

`examples/gemma4_local.py` is the same client outside the game: one support-ticket `Choice` and one `Noul` against the local Gemma server.

## Layout

```
openjevpro/     Choice client, temperature calibration, decision schemas
sts2jev/        snapshot client, candidate expansion, combat narrowing, play loop
examples/       local Gemma calls and the STS2 play entry point
```

## License

Noncommercial use is under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use needs a separate grant; see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).
