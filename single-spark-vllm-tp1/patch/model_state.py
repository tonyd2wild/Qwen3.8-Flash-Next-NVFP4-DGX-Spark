# SPDX-License-Identifier: Apache-2.0
# Base: vLLM main (nightly 8a728663) vllm/models/qwen4_exp/nvidia/model_state.py, Apache-2.0, Copyright contributors to the vLLM project.
# Addition (Kai / 2Wild, 2026-09-05): when QWEN4EXP_PLE_STAGED=1, gather the PLE rows for this step from the disk-backed
# table (our preadv reader) inside prepare_inputs, i.e. on the host BEFORE the model forward or the FULL CUDA graph replay,
# into a fixed GPU buffer the graph reads. Lets one DGX Spark run decode CUDA graphs with the 47.7 GB table on disk.
# Shape of the idea: Trosfy's vLLM PR #54129; independent implementation.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

import os as _os

from .ple_layer import Qwen4ExpNGramEmbedding


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        # Kai/2Wild staged gather: find the n-gram embedding modules that serve rows from disk.
        self._staged_ple: list[Qwen4ExpNGramEmbedding] = []
        if _os.environ.get("QWEN4EXP_PLE_STAGED", "0") == "1":
            for m in model.modules():
                if isinstance(m, Qwen4ExpNGramEmbedding) and getattr(m.ngram_embedding, "_ple_mmap", False):
                    m._ple_init_staging(self.max_num_tokens, self.device)
                    self._staged_ple.append(m)

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        if self._staged_ple:
            num_tokens = int(input_batch.num_tokens)
            num_reqs = int(input_batch.num_reqs)
            padded = int(input_batch.num_tokens_after_padding)
            # The profile/dummy batch has no real prompts: zero the rows instead of touching the table.
            real = num_reqs > 0 and int(input_batch.prefill_len_np[:num_reqs].max()) > 0 if hasattr(input_batch, "prefill_len_np") else num_reqs > 0
            for m in self._staged_ple:
                m._ple_stage_rows(
                    input_batch.input_ids,
                    query_start_loc[: num_reqs + 1],
                    ngram_context[:num_reqs],
                    num_tokens,
                    padded,
                    real=real,
                )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        for m in self._staged_ple:
            m._ple_stage_zero(num_tokens)  # capture-time: nothing host-side may be enqueued here
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
