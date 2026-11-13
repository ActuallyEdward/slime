# Pac-Man runtime integration

These changes were used by the Pac-Man training installation based on upstream
Slime commit `41014d1f29e201137fdffce737bb8bac65bc5219`. They are ported onto this
fork's newer `main`; its existing CUDA, SGLang, and Megatron version pins are
preserved. This is a source port, not a claim that the newer dependency stack
has completed the same training experiment.

## Runtime changes

- The multi-turn VLM example passes no model log probabilities for environment
  tokens. `Sample.append_response_tokens` supplies the zero mask/logprob entries
  required for those observation tokens.
- A per-turn token limit remains separate from the full-context budget. A
  `length` finish can still pass the response to the environment, allowing its
  parser to recover a valid objective from a truncated reason. Aborts and an
  exhausted context budget still terminate the rollout.
- Prompt-length filtering uses the images/videos already stored on the sample.
  Rendered prompt strings do not contain the original image objects.
- Packed Qwen3.5 vision-language inputs with context parallelism set to one
  retain every token, including odd-length sequences.
- `scripts/models/qwen3.5-9B-vl.sh` selects the native vision-language provider
  while keeping the standard 9B model dimensions.
- The Conda build installs SOCKS support and protobuf/protoc before its first
  pip request, including after environment recreation.

The Pac-Man game, prompts, snapshot guards, and Slime environment adapter live
in [ActuallyEdward/pac-man-agentic-rl](https://github.com/ActuallyEdward/pac-man-agentic-rl).
They are configured through the example's pluggable environment interface.

## Container bootstrap

`install_slime_in_areal.sh` is the portable version of the installation wrapper
used inside the AReaL container. It invokes `build_conda.sh` and accepts
`SLIME_DIR`, `BASE_DIR`, `MAMBA_ROOT_PREFIX`, and optional `SLIME_PROXY_URL`.
Existing proxy variables are preserved unless `SLIME_PROXY_URL` is supplied;
set `NO_PROXY`/`no_proxy` for your local inference services.

```bash
BASE_DIR=/path/to/slime-runtime \
MAMBA_ROOT_PREFIX=/path/to/micromamba \
bash install_slime_in_areal.sh
```

## Focused validation

Run the CPU checks in an environment with Slime's Python dependencies:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest \
  tests/test_sample.py \
  tests/test_filter_long_prompt.py \
  tests/test_qwen3_5_vl_native.py \
  tests/test_vlm_multi_turn_runtime.py
bash -n build_conda.sh install_slime_in_areal.sh scripts/models/qwen3.5-9B-vl.sh
```

The full dependency build and distributed GPU training are separate validation
steps. Publishing these patches does not update a running training installation.
