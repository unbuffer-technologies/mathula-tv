import json

import pytest

from mathula_tv.dub_publish import publish_handoff
from mathula_tv.experience_store import ExperienceStore
from mathula_tv.native_dub import native_dub_paths

JOB = "a537c3077a094523b7c15056ccc556a5"


def _rendered_run(tmp_path, with_hook=True):
    store = ExperienceStore(tmp_path)
    render_root = store.repeat_dir("prod-1b", "a537c307", 1) / "render"
    paths = native_dub_paths(render_root)
    paths.video.parent.mkdir(parents=True, exist_ok=True)
    paths.video.write_bytes(b"master")
    if with_hook:
        paths.publication_video.write_bytes(b"with-hook")
    paths.seo_package.write_text(json.dumps({
        "caption": "The MK Party has removed its Secretary General.",
        "tiktok_caption": "The MK Party has removed its Secretary General.",
        "tiktok_hashtags": ["#SboneloNomvalo", "#MKParty", "#SouthAfricaPolitics"],
        "search_keywords": ["MK Party leadership changes", "Sbonelo Nomvalo removed"],
        "cover_hook": "I-MK Party iguqule ubuholi bayo", "title": "I-MK Party iguqule ubuholi bayo"}), encoding="utf-8")
    paths.hook_text.write_text("I-MK Party iguqule ubuholi bayo\n", encoding="utf-8")


def test_the_handoff_is_named_with_the_job_id_and_the_caption_file_is_paste_ready(tmp_path):
    _rendered_run(tmp_path)
    result = publish_handoff(tmp_path, job_id=JOB, run_id="prod-1b", clip_id="a537c307")
    out = tmp_path / "jobs" / JOB / "output"
    assert (out / f"final_dubbed_{JOB}.mp4").read_bytes() == b"with-hook"          # the post-ready video is the hook version
    assert (out / f"dubbed_master_{JOB}.mp4").read_bytes() == b"master"
    caption = (out / f"tiktok_caption_zu_{JOB}.txt").read_text(encoding="utf-8")
    text, blank, tags = caption.rstrip("\n").split("\n")
    assert text == "The MK Party has removed its Secretary General." and blank == ""
    assert tags.split()[0] == "#ZuluTikTok" and "#MKParty" in tags and len(tags.split()) <= 5
    for name in ("tiktok_hashtags_zu", "tiktok_cover_zu", "tiktok_search_keywords_zu"):
        assert (out / f"{name}_{JOB}.txt").is_file()
    assert (out / f"tiktok_seo_zu_{JOB}.json").is_file() and result["final_video"].endswith(f"final_dubbed_{JOB}.mp4")


def test_publishing_without_a_hook_render_says_what_to_do(tmp_path):
    _rendered_run(tmp_path, with_hook=False)
    with pytest.raises(FileNotFoundError, match="render the run with the hook first"):
        publish_handoff(tmp_path, job_id=JOB, run_id="prod-1b", clip_id="a537c307")
