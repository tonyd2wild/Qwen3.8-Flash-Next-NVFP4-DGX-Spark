# Community-reported SM121 kernel fix (NOT tested by us)

`sm121_varlen.py` is a reference copy of the Triton packed-varlen QSA decode
fallback that multiple DGX Spark / GB10 operators report fixes the token-0
`!!!!!!` loop at the **kernel level** (keeping radix + CUDA graphs ON), rather
than disabling them as our shipped config does.

It forces the FlashInfer TRT-LLM sparse-decode path off on SM121 (which silently
corrupts long-context decode on GB10) and substitutes this one-query packed
varlen kernel, reading `cu_seqlens` on-device so CUDA-graph replay stays valid.

Source: extracted from `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` (`start.sh`),
which ports sgl-project/sglang PRs **#36806** + **#36845**.

⚠️ **We have NOT validated this on our deployment.** Provided for reference and
for anyone who wants to try reclaiming the throughput our `--disable-radix-cache`
/ `--disable-cuda-graph-padding` workaround gives up. See the main README's
agent-safety section.
