# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tony DeAngelo (Tech2Wild / 2Wild); written by Kai
# Kai / 2Wild, 2026-09-05. Disk-backed (mmap) n-gram PLE table for Qwen3.8-Flash-Next
# (NVIDIA ModelOpt checkpoint) so the 47.7 GiB FP8 table never has to be resident.
# Own implementation, written for the single-DGX-Spark (GB10, 128 GB unified) case.
"""Memory-mapped reader for the Qwen4Exp PLE n-gram embedding table.

The checkpoint stores the table as ``split_ngram_parts`` row shards
(``...ngram_embedding.shard_<i>.weight``, FP8 E4M3, [rows, head_dim]) inside
safetensors files. Instead of copying them into one GPU tensor we keep the files
left on disk; only the rows a forward pass needs are read (positional reads on
a thread pool, page cache keeps the hot rows) into a pinned staging buffer that
is then copied to a fixed GPU buffer.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4}
_TORCH_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


class PLEMmapTable:
    """One PLE n-gram table (all row shards of one PLE layer), mmap-backed."""

    def __init__(
        self,
        model_dir: str,
        layer_idx: int,
        *,
        threads: int | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.layer_idx = int(layer_idx)
        self.threads = threads or max(8, min(48, 2 * (os.cpu_count() or 8)))
        self.chunk_rows = int(os.environ.get("QWEN4EXP_PLE_MMAP_CHUNK", "8"))
        self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="ple-mmap")
        self._lock = threading.Lock()
        # shard_index -> (path, data_begin_abs, rows, row_bytes, dtype)
        shards: dict[int, tuple[str, int, int, int, str]] = {}
        pat = re.compile(
            rf"(?:^|\.)layers\.{self.layer_idx}\.ple\..*ngram_embedding\.shard_(\d+)\.weight$"
        )
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted({v for k, v in weight_map.items() if pat.search(k)})
        else:
            files = sorted(
                fn for fn in os.listdir(model_dir) if fn.endswith(".safetensors")
            )
        for fn in files:
            path = os.path.join(model_dir, fn)
            header, base = _read_header(path)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                m = pat.search(name)
                if not m:
                    continue
                shard = int(m.group(1))
                dtype = meta["dtype"]
                rows, width = meta["shape"]
                begin, end = meta["data_offsets"]
                row_bytes = width * _DTYPE_BYTES[dtype]
                if end - begin != rows * row_bytes:
                    raise ValueError(f"PLE shard {name}: size mismatch")
                if shard in shards:
                    raise ValueError(f"PLE shard {shard} appears twice")
                shards[shard] = (path, base + begin, rows, row_bytes, dtype)
        if not shards:
            raise ValueError(
                f"no PLE n-gram shards for layer {self.layer_idx} under {model_dir}"
            )
        expected = list(range(len(shards)))
        if sorted(shards) != expected:
            raise ValueError(f"PLE shards are not contiguous: {sorted(shards)[:5]}...")
        self.num_shards = len(shards)
        self.row_bytes = shards[0][3]
        self.dtype = _TORCH_DTYPE[shards[0][4]]
        self.shard_rows = shards[0][2]
        if any(s[3] != self.row_bytes or s[4] != shards[0][4] for s in shards.values()):
            raise ValueError("PLE shards disagree on dtype/width")
        self.rows_total = sum(s[2] for s in shards.values())
        # Every shard but the last must hold shard_rows rows (checkpoint_start
        # arithmetic in the stock loader assumes this).
        for i in range(self.num_shards - 1):
            if shards[i][2] != self.shard_rows:
                raise ValueError("PLE shards are not equal-sized")
        # Positional-read backend: one fd per file, rows fetched with preadv
        # straight into the staging buffer. (An mmap backend was measured first:
        # page faults serialize on the process mmap_lock, ~10-20K rows/s cold no
        # matter the thread count. preadv scales with NVMe queue depth.)
        self._fds: dict[str, int] = {}
        self.shard_base: list[tuple[int, int]] = []  # (fd, abs byte offset of row 0)
        for i in range(self.num_shards):
            path, begin, rows, row_bytes, _ = shards[i]
            self.shard_base.append((self._open(path), begin))
        self._staging: torch.Tensor | None = None

    def _open(self, path: str) -> int:
        if path in self._fds:
            return self._fds[path]
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass
        self._fds[path] = fd
        return fd

    # ------------------------------------------------------------------ gather
    def _staging_for(self, n: int) -> torch.Tensor:
        if self._staging is None or self._staging.shape[0] < n:
            cap = max(n, 4096)
            try:
                self._staging = torch.empty(
                    (cap, self.row_bytes), dtype=torch.uint8, pin_memory=True
                )
            except Exception:
                # No CUDA context (CPU test) or pinned pool exhausted: pageable
                # staging still works, just without async H2D overlap.
                self._staging = torch.empty((cap, self.row_bytes), dtype=torch.uint8)
        return self._staging[:n]

    def _gather_chunk(self, ids: list[int], mv: memoryview, start: int) -> None:
        """Read rows for ids into staging rows start..start+len(ids) (mv = flat bytes)."""
        rb = self.row_bytes
        sr = self.shard_rows
        base = self.shard_base
        preadv = os.preadv
        for k, i in enumerate(ids):
            s, l = divmod(i, sr)
            fd, off = base[s]
            dst = (start + k) * rb
            n = preadv(fd, [mv[dst : dst + rb]], off + l * rb)
            if n != rb:
                raise IOError(f"short PLE row read: {n} of {rb} bytes (row {i})")

    def gather_cpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather rows for flat int64 CPU ids -> (pinned) uint8 [n, row_bytes].

        Rows are read with preadv on a thread pool: the syscall releases the
        GIL and each thread keeps its own request in flight, which is what
        gives the NVMe queue depth. Page-cache hits cost a few microseconds.
        """
        assert ids.device.type == "cpu"
        n = ids.numel()
        out = self._staging_for(n)
        if n == 0:
            return out
        id_list = ids.view(-1).tolist()
        mv = memoryview(out.numpy()).cast("B")  # flat writable bytes of the staging rows
        chunk = self.chunk_rows
        if n <= chunk:
            self._gather_chunk(id_list, mv, 0)
            return out
        futs = [
            self._pool.submit(self._gather_chunk, id_list[a : a + chunk], mv, a)
            for a in range(0, n, chunk)
        ]
        for f in futs:
            f.result()
        return out

    def gather(self, ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Gather rows for ids (any device) into a GPU tensor of self.dtype.

        Returns [n, head_dim] on ids.device (or ``out`` if given).
        """
        device = ids.device
        cpu_ids = ids.detach().to("cpu", dtype=torch.int64)  # sync point
        rows = self.gather_cpu(cpu_ids)
        if out is None:
            out = torch.empty((rows.shape[0], self.row_bytes), dtype=torch.uint8, device=device)
        out[: rows.shape[0]].copy_(rows, non_blocking=True)
        return out[: rows.shape[0]].view(self.dtype)

    # ------------------------------------------------------------------ info
    def describe(self) -> str:
        return (
            f"PLEMmapTable(layer={self.layer_idx}, shards={self.num_shards}, "
            f"rows={self.rows_total:,}, row_bytes={self.row_bytes}, dtype={self.dtype}, "
            f"threads={self.threads}, files={len(self._fds)})"
        )


__all__ = ["PLEMmapTable"]
