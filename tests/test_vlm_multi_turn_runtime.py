"""CPU regressions for environment-token masking and per-turn generation limits."""

import ast
import asyncio
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def rollout():
    # Execute the actual rollout functions without importing the SGLang server
    # stack; network, processor, and environment boundaries are supplied below.
    path = Path(__file__).resolve().parents[1] / "examples/geo3k_vlm_multi_turn/rollout.py"
    tree = ast.parse(path.read_text())
    names = {
        "_append_to_sample",
        "_should_stop_on_finish",
        "_update_budget",
        "_merge_multimodal_train_inputs",
        "_finalize_sample",
        "_update_multimodal_state",
        "generate",
    }
    body = ast.parse("from __future__ import annotations").body
    body += [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ModuleType("isolated_vlm_rollout")
    module.Sample = Sample
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.unit
def test_environment_tokens_have_zero_loss_mask_and_logprobs(rollout):
    sample = Sample(tokens=[10])
    responses = []
    rollout._append_to_sample(sample, responses, [11], [-0.5], loss_mask_val=1)
    rollout._append_to_sample(sample, responses, [20, 21], [0.0, 0.0], loss_mask_val=0)
    assert sample.tokens == [10, 11, 20, 21]
    assert sample.loss_mask == [1, 0, 0]
    assert sample.rollout_log_probs == [-0.5, 0.0, 0.0]
    assert sample.response_length == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    ("budget", "expected_limits", "expected_status"),
    [(6, [2, 2], Sample.Status.COMPLETED), (4, [2, 1], Sample.Status.TRUNCATED)],
)
def test_per_turn_limit_preserves_environment_steps_and_total_budget(
    rollout, budget, expected_limits, expected_status
):
    sample = Sample(tokens=[10])
    limits = []
    transitions = []
    closed = []
    env = SimpleNamespace(reset=lambda: None, close=lambda: closed.append(True))
    state = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda tokens, **kw: str(tokens)), processor=None)
    rollout._initialize_resources = lambda args, sample: (env, None, {"max_turns": 3}, state, "unused")
    rollout._prepare_start_state = lambda *args: (None, [], budget, [])

    async def inference(url, tokens, parameters, images, tokenizer):
        count = parameters["max_new_tokens"]
        limits.append(count)
        return '{"objective_id":"C0"}', [11] * count, [-0.5] * count, "length", None

    def environment_step(*args):
        transitions.append(True)
        return [20], None, None, None, len(transitions) == 2

    rollout._run_inference_step = inference
    rollout._process_env_step = environment_step
    parameters = {"max_new_tokens": 2}
    result = asyncio.run(rollout.generate(SimpleNamespace(partial_rollout=False), sample, parameters))
    assert limits == expected_limits
    assert result.status == expected_status
    assert len(transitions) == (2 if expected_status == Sample.Status.COMPLETED else 1)
    assert len(result.rollout_log_probs) == len(result.loss_mask) == result.response_length
    assert parameters == {"max_new_tokens": 2}
    assert closed == [True]


@pytest.mark.unit
def test_abort_still_stops_interaction(rollout):
    sample = Sample()
    assert rollout._should_stop_on_finish(sample, "abort")
    assert sample.status == Sample.Status.ABORTED
