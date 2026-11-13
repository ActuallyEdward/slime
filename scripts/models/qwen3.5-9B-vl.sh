source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen3.5-9B.sh"

# Qwen3.5-9B is an omni-modal checkpoint, but slime's stock 9B profile uses
# the text-only provider. Select the native VL provider while retaining the
# official 9B language-model dimensions above.
MODEL_ARGS[1]="slime_plugins.models.qwen3_5_vl"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"
