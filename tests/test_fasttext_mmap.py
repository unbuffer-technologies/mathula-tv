import struct

import numpy as np
import pytest

from mathula_tv.fasttext_mmap import FastTextFile, fnv1a, subword_grams


def _reference_hash(data: bytes) -> int:
    h = 2166136261
    for (c,) in struct.iter_unpack("<b", data):            # signed char, as C++ int8_t
        h = ((h ^ (c & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h


def test_fnv1a_matches_the_published_value_and_sign_extends_bytes_above_127():
    assert fnv1a(b"") == 2166136261
    assert fnv1a(b"a") == 0xE40C292C                        # the standard 32-bit FNV-1a of "a"
    assert fnv1a("é".encode()) == _reference_hash("é".encode()) != fnv1a(b"\xc3")   # two-byte character: both bytes sign-extended


def test_subword_grams_follow_fasttexts_rules():
    assert subword_grams("where", 3, 3) == [b"<wh", b"whe", b"her", b"ere", b"re>"]       # the fastText paper's example
    assert subword_grams("where", 2, 2) == [b"<w", b"wh", b"he", b"er", b"re", b"e>"]
    assert b"<where" in subword_grams("where", 2, 6) and b"<where>" not in subword_grams("where", 2, 6)    # 7 characters: past maxn
    assert subword_grams("é", 2, 4) == ["<é".encode(), "<é>".encode(), "é>".encode()]     # a grams starts on a character, never inside one


def _write_model(path, words, bucket, minn, maxn, dim, rows):
    out = bytearray()
    out += struct.pack("<ii", 793712314, 12)
    out += struct.pack("<12i", dim, 5, 5, 5, 5, 1, 2, 2, bucket, minn, maxn, 100)
    out += struct.pack("<d", 1e-4)
    out += struct.pack("<iii", len(words), len(words), 0)
    out += struct.pack("<qq", 1000, -1)
    for w in words:
        out += w.encode() + b"\x00" + struct.pack("<q", 7) + b"\x00"
    out += b"\x00"                                           # not quantised
    out += struct.pack("<qq", rows.shape[0], rows.shape[1])
    out += rows.astype("<f4").tobytes()
    path.write_bytes(bytes(out))


def test_the_reader_gives_the_mean_of_the_word_row_and_its_ngram_rows(tmp_path):
    words, bucket, dim = ["ab", "cde", "x"], 11, 4
    rows = np.arange((len(words) + bucket) * dim, dtype=np.float32).reshape(-1, dim) * 0.5
    path = tmp_path / "tiny.bin"
    _write_model(path, words, bucket, 2, 3, dim, rows)
    model = FastTextFile(path)
    assert model.words == words and model.dim == dim and model.bucket == bucket

    grams = subword_grams("ab", 2, 3)
    expect_ids = [0] + [len(words) + fnv1a(g) % bucket for g in grams]
    np.testing.assert_allclose(model.vector("ab"), rows[expect_ids].mean(axis=0))
    assert model.row_ids("ab") == expect_ids
    # an unseen form uses its n-grams only
    oov = model.row_ids("abx")
    assert 0 not in oov and len(oov) == len(subword_grams("abx", 2, 3))
    np.testing.assert_allclose(model.vectors(["ab", "abx"]), np.stack([model.vector("ab"), model.vector("abx")]), rtol=1e-6)


def test_a_file_that_is_not_a_fasttext_model_is_refused(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        FastTextFile(path)
