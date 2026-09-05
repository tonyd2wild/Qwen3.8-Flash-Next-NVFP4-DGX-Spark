# PLE mmap gather vs vLLM main @ 8a728663c1c3eeace834a95f5654fa653cc1998c

Source of truth: tarball extracted at
/private/tmp/claude-501/-Users-clawdbot/ef3eadcb-21f6-4df8-b0d8-0fccf98884bb/scratchpad/research/runner/vllm-8a728663c1c3eeace834a95f5654fa653cc1998c
Local qwen4exp copy is byte-identical to vllm/models/qwen4_exp at this commit (diff -rq empty).

## Q1 split ops
- op registered: vllm/models/qwen4_exp/nvidia/ple_layer.py:907-912 (mutates_args=["output"])
- listed in CompilationConfig._attention_ops: vllm/config/compilation.py:765-783 (line 771)
- default splitting_ops = list(_attention_ops) (+kv_cache_update ops): compilation.py:1153-1191, called from vllm/config/vllm.py:1679-1682
- fx split: vllm/compilation/backends.py:1174-1179 -> split_graph 553-627; should_split partition_rules.py:14-38
- split-op submodules not compiled/wrapped: backends.py:1202-1206
- inductor partition path: partition_rules.py:41-75 (custom_should_partition_ops), backends.py:149-160
- splitting_ops_contain_attention requires ALL entries: compilation.py:1277-1280 (gates 1412,1430,1459,1305)
- default opt level O2: vllm.py:423 -> cudagraph_mode FULL_AND_PIECEWISE: vllm.py:305
- backends for this model UNIFORM_BATCH: short_conv_attn.py:108, nvidia/qsa.py:64, mamba_attn.py:93; enum backend.py:567-581
- resolve_cudagraph_mode_and_sizes: compilation.py:1375-1519; V2 call gpu/model_runner.py:641-650
- V2 FULL capture wraps whole model(**model_inputs): gpu/cudagraph_utils.py:385-403 + 578-601; replay 449-462
- candidates FULL for uniform decode: cudagraph_utils.py:264-303, 307-322; dispatch 419-447 via dp_utils.py:120/202
- enforce_eager: vllm.py:1378-1384 and 1592-1596; set_splitting_ops early return compilation.py:1145-1148

## Q2 runner
- use_v2_model_runner: vllm.py:652-684; unsupported list 2547-2620; worker pick gpu_worker.py:464-481
- model raises without kwargs: nvidia/model.py:297-298; V1 has no ngram plumbing (gpu_model_runner.py:1086-1125 only token_type_ids)
- _maybe_add_ngram_kwargs does not exist (only comments model.py:673, amd/model.py:670)
- ModelState selection: gpu/model_states/__init__.py:28-33; get_model_state_cls model.py:659-663, 862-866; init gpu/model_runner.py:415-417
- prepare_inputs merge: gpu/model_runner.py:1721-1729; forward 1761-1796; dummy capture cudagraph_utils.py:550-554
- Qwen4ExpModelState: model_state.py:59-63 fixed qsl buffer, 65-92 ctx gather, 94-110, 112-140
- input_ids assembled on GPU: gpu/model_runner.py:1276-1284, 1314-1325; states.py:31-39 (UVA all_token_ids), 66-80
- async scheduling default True on CUDA: vllm.py:1270-1319 (disabled for mtp 1282-1294)
- TP sharded embedding + all_reduce: common/ple.py:82-100; vocab_parallel_embedding.py:279-346, 501-540
- fp8 dequant expects fp8 output: ple_layer.py:611-625

## Q3 MTP
- mtp.py:171 start idx = num_hidden_layers; 205-212 layer prefix; 313-321 passes None kwargs
- PLE built only if (layer_idx+1) in ple_layer_ids: model.py:196-208; validated 1..num_hidden_layers: qwen4_exp/config.py:122-132
- spec config override speculative.py:820-841; registry.py:678; V2 MTPSpeculator mtp/speculator.py:12-35; load_eagle_model eagle/utils.py:36-76

## Q4 quant
- ModelOptMixedPrecisionConfig modelopt.py:1466-1760; override 1520-1526; extractor 267-284; _from_config 1529-1605; _resolve_quant_algo 1607-1688; get_quant_method 1703-1749
- auto-detect vllm/config/model.py:1260-1334 (mismatch error 1328-1334); registry quantization/__init__.py:21,149
- hf_quant_config.json load transformers_utils/config.py:781-790; weight_utils.py:271-287
- PLE method select ple_layer.py:133-161; preselected honored vocab_parallel_embedding.py:289-292

## Q5 residency
- create_weights ple_layer.py:88-113 / vocab_parallel_embedding.py:50-60, 338-346
- shard loader ple_layer.py:490-523 -> common/ple.py:65-66 raises on small destination
- process_weights_after_loading ple_layer.py:115-119; loader loop model_loader/utils.py:97-146
- low_latency_gemm.py:230-265 only LinearBase/ParallelLMHead
- warmup qwen4_exp_qsa_warmup.py:17-66; kernel_warmup.py:170
- sleep gpu_worker.py:240-300; reload_weights 515-517 / gpu/model_runner.py:506-510
- weight transfer gpu_worker.py:504-506, 978-980, 1360-1433
