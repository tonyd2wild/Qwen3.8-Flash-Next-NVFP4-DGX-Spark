# Draft upstream issue: ModelOpt mixed-precision resolver cannot find MTP expert quant algo for nvidia/Qwen3.8-Flash-Next-NVFP4

**vLLM:** main nightly 8a728663 (2026-09-04), also checked main HEAD 2026-09-05 (no change to `_resolve_quant_algo`).
**Model:** `nvidia/Qwen3.8-Flash-Next-NVFP4`, `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.

**Symptom** (load fails at the last shard):
```
AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'
for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'
```

**Cause:** the checkpoint's `hf_quant_config.json` keys the MTP experts by the draft-local index, `mtp.layers.0.mlp.experts: {quant_algo: FP8_BLOCK_SCALES, group_size: 128}`, while vLLM's `Qwen4ExpMultiTokenPredictor` names draft layers `mtp.layers.{num_hidden_layers + i}` (48 here), and the checkpoint's own tensor names also use 48. `ModelOptMixedPrecisionConfig._resolve_quant_algo("mtp.layers.48.mlp.experts")` therefore returns None, the RoutedExperts layer is built without the block-scale parameters, and the FP8 scale tensors have nowhere to land.

**Fix used locally:** in `_quantized_layer_prefix_candidates`, when the prefix matches `mtp.layers.<k>.<rest>`, also offer `mtp.layers.<j>.<rest>` for the draft-local indices (patch: `patch/modelopt_mtp_index.diff`). A cleaner upstream fix would subtract `num_hidden_layers` when the config is available, or normalize the keys in `apply_vllm_mapper`.

Reported by Tony / Tech2Wild (tonyd2wild) and Kai; found while serving the checkpoint on a single DGX Spark.

## Second gap found the same day (2026-09-05, after the index fix)

With the index fixed, `_resolve_quant_algo` returns `FP8_BLOCK_SCALES` for the MTP experts, but `ModelOptMixedPrecisionConfig.get_quant_method` only handles `FP8` (per-tensor, `ModelOptFp8MoEMethod`), `NVFP4`, `W4A16_NVFP4` and `MXFP8` for `RoutedExperts`, so it returns `None` and the experts are built unquantized, giving the same `w2_weight_scale_inv` error. Local fix: route `FP8_BLOCK_SCALES` (group_size from the config entry, 128 here) to `Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128]), layer)`, whose block mode already names the parameters `w13_weight_scale_inv` / `w2_weight_scale_inv` as the loader expects.

