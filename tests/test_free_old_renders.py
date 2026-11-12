import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import free_old_renders as fr


def _run(root, run, clip, mtime, size=10):
    folder = root / run / clip / "r1"
    (folder / "render").mkdir(parents=True)
    (folder / "render" / "dub.wav").write_bytes(b"x" * size)
    (folder / "submission.json").write_text("{}", encoding="utf-8")
    os.utime(folder / "render", (mtime, mtime))


def test_only_the_newest_renders_of_each_clip_are_kept_and_nothing_else_of_a_run_is_touched(tmp_path):
    for i in range(5):
        _run(tmp_path, f"prod-{i}", "clipA", 1000 + i)
    _run(tmp_path, "prod-9", "clipB", 5000)
    gone = fr.candidates(tmp_path, keep=2)
    assert sorted(p.parts[-4] for p in gone) == ["prod-0", "prod-1", "prod-2"] and all(p.parts[-3] == "clipA" for p in gone)
    assert fr.main(["--root", str(tmp_path), "--keep", "2"]) == 0 and (tmp_path / "prod-0" / "clipA" / "r1" / "render").is_dir()          # a dry run deletes nothing
    assert fr.main(["--root", str(tmp_path), "--keep", "2", "--apply"]) == 0
    assert not (tmp_path / "prod-0" / "clipA" / "r1" / "render").exists() and (tmp_path / "prod-0" / "clipA" / "r1" / "submission.json").is_file()
    assert (tmp_path / "prod-4" / "clipA" / "r1" / "render").is_dir() and (tmp_path / "prod-3" / "clipA" / "r1" / "render").is_dir() and (tmp_path / "prod-9" / "clipB" / "r1" / "render").is_dir()
