"""Read a fastText ``.bin`` model without loading it: the input matrix stays on disk and only the rows a word needs are read.

The NCHLT isiZulu models are 5.7 GB each (2.18 M rows of 600 floats: 183 k words and 2 M hashed word-piece buckets) and the machine has 6.9 GB of RAM, so
``fasttext.load_model`` would take the whole box.  A word's vector in fastText is the mean of its own row (when it is in the vocabulary) and the rows of its
character n-grams, so a memory-mapped matrix gives exactly the same vectors, for words the model has seen and for any form it has not (Zulu makes new forms
all the time).  The n-gram and hash rules below are fastText's own (``Dictionary::computeSubwords`` and ``Dictionary::hash``).

Format (version 12): magic, version, 12 int32 args and a double, the dictionary (size, nwords, nlabels, ntokens, pruneidx_size, then word / count / type for
each entry, then the prune pairs), a quantisation flag, and the input matrix as int64 rows, int64 columns, float32 data.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np

MAGIC = 793712314
VERSION = 12
_BOW, _EOW = "<", ">"


def fnv1a(data: bytes) -> int:
    """fastText's hash of an n-gram: 32-bit FNV-1a over the bytes sign-extended as signed chars (as ``uint32_t(int8_t(c))`` in C++)."""
    h = 2166136261
    for byte in data:
        h ^= (byte - 256 if byte > 127 else byte) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def subword_grams(word: str, minn: int, maxn: int) -> list[bytes]:
    """The character n-grams of ``<word>`` fastText hashes, as UTF-8 bytes (an n-gram of one character is never taken at the word's edge)."""
    raw = (_BOW + word + _EOW).encode("utf-8")
    grams: list[bytes] = []
    for i in range(len(raw)):
        if raw[i] & 0xC0 == 0x80:                    # inside a multi-byte character: an n-gram starts only at a character's first byte
            continue
        j, n = i, 1
        while j < len(raw) and n <= maxn:
            j += 1
            while j < len(raw) and raw[j] & 0xC0 == 0x80:
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == len(raw))):
                grams.append(raw[i:j])
            n += 1
    return grams


class FastTextFile:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic, version = struct.unpack("<ii", f.read(8))
            if magic != MAGIC or version != VERSION:
                raise ValueError(f"not a fastText v12 model: magic {magic}, version {version}")
            (self.dim, self.ws, self.epoch, self.min_count, self.neg, self.word_ngrams, self.loss, self.model,
             self.bucket, self.minn, self.maxn, self.lr_update_rate) = struct.unpack("<12i", f.read(48))
            f.read(8)                                          # sampling threshold t (a double)
            size, self.nwords, nlabels = struct.unpack("<iii", f.read(12))
            self.ntokens, pruneidx_size = struct.unpack("<qq", f.read(16))
            if pruneidx_size > 0:
                raise ValueError("pruned (quantised) models are not supported")
            words: list[str] = []
            for _ in range(size):
                chars = bytearray()
                while (c := f.read(1)) != b"\x00":
                    chars += c
                f.read(9)                                      # count (int64) and entry type (int8)
                words.append(chars.decode("utf-8", "replace"))
            if f.read(1) != b"\x00":
                raise ValueError("quantised models are not supported")
            rows, cols = struct.unpack("<qq", f.read(16))
            offset = f.tell()
        self.words = words[: self.nwords]
        self.index = {w: i for i, w in enumerate(self.words)}
        if cols != self.dim or rows != self.nwords + self.bucket:
            raise ValueError(f"unexpected matrix shape {rows}x{cols}")
        self.matrix = np.memmap(self.path, dtype="<f4", mode="r", offset=offset, shape=(rows, cols))

    # -- ids -------------------------------------------------------------------------------------------------------------------------------
    def row_ids(self, word: str) -> list[int]:
        """The matrix rows fastText averages for ``word``: its own (if in the vocabulary) and its n-grams' (none for the end-of-sentence token)."""
        ids: list[int] = []
        if word in self.index:
            ids.append(self.index[word])
            if word == "</s>":
                return ids
        ids += [self.nwords + fnv1a(g) % self.bucket for g in subword_grams(word, self.minn, self.maxn)]
        return ids

    def vector(self, word: str) -> np.ndarray:
        ids = self.row_ids(word)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        return np.asarray(self.matrix[ids], dtype=np.float32).mean(axis=0)

    def vectors(self, words: Iterable[str], batch: int = 1500) -> np.ndarray:
        """Vectors of many words, read in batches so memory stays at a few hundred MB."""
        words = list(words)
        out = np.zeros((len(words), self.dim), dtype=np.float32)
        for start in range(0, len(words), batch):
            chunk = words[start:start + batch]
            ids = [self.row_ids(w) for w in chunk]
            lengths = np.array([max(len(i), 1) for i in ids])
            flat = np.fromiter((r for group in ids for r in group), dtype=np.int64)
            if flat.size:
                rows = np.asarray(self.matrix[flat], dtype=np.float32)
                bounds = np.concatenate([[0], np.cumsum([len(i) for i in ids])[:-1]])
                sums = np.add.reduceat(rows, bounds, axis=0) if len(ids) else rows
                has = np.array([len(i) > 0 for i in ids])
                out[start:start + len(chunk)][has] = sums[has] / lengths[has, None]
        return out
