import json

from mathula_tv import youtube_captions as yc


def _write(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"events": events}), encoding="utf-8")


def test_events_are_caption_lines_without_empty_breaks_and_reading_returns_the_overlap_once(tmp_path):
    folder = tmp_path / "input" / "youtube_captions"
    _write(folder / "abc.zu-orig.json3", [
        {"tStartMs": 0, "dDurationMs": 3000, "segs": [{"utf8": "imali yethu "}, {"utf8": "siyayibhadala"}]},
        {"tStartMs": 3000, "dDurationMs": 100, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 3000, "dDurationMs": 3000, "segs": [{"utf8": "through tax"}]},
        {"tStartMs": 3000, "dDurationMs": 3000, "segs": [{"utf8": "through tax"}]},              # a repeated rolling line
        {"tStartMs": 9000, "dDurationMs": 2000, "segs": [{"utf8": "nale yabantwana"}]}])
    lines = yc.events(folder / "abc.zu-orig.json3")
    assert [t for _, _, t in lines] == ["imali yethu siyayibhadala", "through tax", "through tax", "nale yabantwana"]
    read = yc.reading(tmp_path)
    assert read(2.0, 5.0) == "imali yethu siyayibhadala through tax"                              # overlapping lines, each once
    assert read(8.0, 12.0) == "nale yabantwana" and read(20.0, 25.0) == ""


def test_manual_subtitles_are_preferred_over_the_automatic_original_track_and_no_captions_means_no_reader(tmp_path):
    assert yc.reading(tmp_path) is None
    folder = tmp_path / "input" / "youtube_captions"
    _write(folder / "abc.zu-orig.json3", [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "automatic"}]}])
    _write(folder / "abc.en.json3", [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "uploaded by the channel"}]}])
    assert [p.name for p in yc.find(tmp_path)] == ["abc.en.json3", "abc.zu-orig.json3"]
    assert yc.reading(tmp_path)(0.0, 1.0) == "uploaded by the channel"


def test_a_caption_download_that_fails_never_fails_the_ingestion(tmp_path, monkeypatch):
    from mathula_tv import youtube_ingestion as yi

    def broken(*args, **kwargs):
        raise yi.YouTubeIngestionError("bot check")

    monkeypatch.setattr(yi, "_run", broken)
    assert yi._download_captions("https://www.youtube.com/watch?v=x", tmp_path) == ()
    (tmp_path / "captions-x.zu-orig.json3").write_text("{}", encoding="utf-8")
    assert [p.name for p in yi._download_captions("https://www.youtube.com/watch?v=x", tmp_path)] == ["captions-x.zu-orig.json3"]


def _segments(*rows):
    return [{"segment_id": sid, "start_ms": int(start * 1000), "end_ms": int(end * 1000), "source_text": text} for sid, start, end, text in rows]


def test_the_captions_are_offered_only_for_the_segments_the_english_recogniser_and_youtube_disagree_on(tmp_path):
    folder = tmp_path / "input" / "youtube_captions"
    _write(folder / "abc.zu-orig.json3", [
        {"tStartMs": 0, "dDurationMs": 4000, "segs": [{"utf8": "good evening to you all"}]},
        {"tStartMs": 4500, "dDurationMs": 4000, "segs": [{"utf8": "nale yabantwana siyayinyusa"}]},
        {"tStartMs": 9000, "dDurationMs": 3000, "segs": [{"utf8": "and the money will be paid"}]}])
    segments = _segments(("seg-1", 0.2, 3.8, "Good evening to you all."),                                 # the same words: no hint
                         ("seg-2", 4.8, 8.0, "Nally a bunch of anna see yay in new sir"),               # nonsense English for isiZulu: a hint
                         ("seg-3", 9.1, 11.8, "And the money will be paid"),
                         ("seg-4", 20.0, 22.0, "Nothing is captioned here at all"),                      # no caption overlaps: nothing to offer
                         ("seg-5", 4.8, 5.0, "Yes ok"))                                                  # under three words: left alone
    found = yc.hints(tmp_path, segments)
    assert found == {"track": "zu-orig", "segments": {"seg-2": "nale yabantwana siyayinyusa"}}
    assert yc.hints(tmp_path, segments[:1]) == {}                                                        # nothing differs
    assert yc.hints(tmp_path / "other", segments) == {}                                                  # a video without captions changes nothing


def test_the_hints_are_capped_to_the_strongest_differences_in_time_order(tmp_path, monkeypatch):
    folder = tmp_path / "input" / "youtube_captions"
    events = [{"tStartMs": i * 10000, "dDurationMs": 3000, "segs": [{"utf8": f"caption words number {i}"}]} for i in range(6)]
    _write(folder / "abc.en-orig.json3", events)
    rows = [(f"seg-{i}", i * 10 + 0.1, i * 10 + 2.9, "zzz qqq xxx yyy" if i in (1, 3, 4) else "caption words number " + str(i) + " and more") for i in range(6)]
    monkeypatch.setattr(yc, "HINT_MAX_SEGMENTS", 2)
    found = yc.hints(tmp_path, _segments(*rows))
    assert found["track"] == "en-orig" and list(found["segments"]) == ["seg-1", "seg-3"]          # three differ equally; two are kept, the earliest, in time order


def test_the_captions_for_a_stretch_the_transcript_has_no_segment_for_are_offered_without_noise_tags(tmp_path):
    folder = tmp_path / "input" / "youtube_captions"
    _write(folder / "abc.zu-orig.json3", [
        {"tStartMs": 0, "dDurationMs": 3000, "segs": [{"utf8": "imali ye grant iphuma"}]},
        {"tStartMs": 4000, "dDurationMs": 3000, "segs": [{"utf8": "nale yabantwana siyayinyusa"}]},
        {"tStartMs": 8000, "dDurationMs": 2000, "segs": [{"utf8": "[ukuhalalisa]"}]},
        {"tStartMs": 12000, "dDurationMs": 2000, "segs": [{"utf8": "and then the next sentence begins"}]}])
    segments = _segments(("seg-1", 0.0, 3.0, "The money from the grant"),
                         ("seg-2", 7.8, 8.2, "Yes"),                                      # a gap of 4.8 s before it: captioned
                         ("seg-3", 9.0, 9.6, "Okay"),                                     # 0.8 s after seg-2: too short a gap to look at
                         ("seg-4", 12.0, 14.0, "And then the next sentence begins"))     # gap 9.6-12.0: only a noise tag there
    found = yc.gaps(tmp_path, segments)
    assert [(g["after"], g["start_ms"], g["end_ms"]) for g in found] == [("seg-1", 3000, 7800)]
    assert found[0]["yt"] == "nale yabantwana siyayinyusa"
    assert yc.gaps(tmp_path / "other", segments) == []                                    # no captions: nothing


def test_the_uploaders_title_and_channel_are_read_from_the_downloaded_source_and_missing_ones_are_empty(tmp_path):
    from mathula_tv.youtube_captions import video_context

    assert video_context(tmp_path) == {}
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "youtube_source.json").write_text(json.dumps({"title": "Kenya end  Mosimane's winning streak after Bafana draw", "channel": "SABC News", "duration": 101.0}), encoding="utf-8")
    assert video_context(tmp_path) == {"title": "Kenya end Mosimane's winning streak after Bafana draw", "channel": "SABC News"}
    (tmp_path / "input" / "youtube_source.json").write_text("not json", encoding="utf-8")
    assert video_context(tmp_path) == {}


def test_the_uploaders_description_and_chapters_are_part_of_the_video_context_and_cut_at_the_limit(tmp_path):
    from mathula_tv.youtube_captions import DESCRIPTION_CHARS, video_context

    (tmp_path / "input").mkdir()
    description = "Advocate Dali Mpofu on the coalition talks.\n\n  \nWatch more at sabcnews.com\n" + "x" * 3000
    (tmp_path / "input" / "youtube_source.json").write_text(json.dumps({"title": "DA and MK Party conclude coalition talks", "channel": "SABC News", "description": description,
                                                                          "chapters": ["Intro", "The talks"]}), encoding="utf-8")
    got = video_context(tmp_path)
    assert got["title"] == "DA and MK Party conclude coalition talks"
    assert got["description"].startswith("Advocate Dali Mpofu on the coalition talks.\nWatch more at sabcnews.com\n") and len(got["description"]) == DESCRIPTION_CHARS
    assert got["chapters"] == "Intro; The talks"


def test_pass_1_is_told_how_to_use_the_title_and_the_payload_carries_it():
    from mathula_tv.native_dub import PASS1_SYSTEM_PROMPT

    assert "THE VIDEO'S OWN TITLE" in PASS1_SYSTEM_PROMPT and '"video"' in PASS1_SYSTEM_PROMPT and "Never take a name from the title" in PASS1_SYSTEM_PROMPT


def test_the_agent_is_shown_the_videos_title_and_description_when_it_opens_the_clip(tmp_path):
    from types import SimpleNamespace

    from mathula_tv.dub_workbench import Workbench

    job = tmp_path / "jobs" / "job-1" / "input"
    job.mkdir(parents=True)
    (job / "youtube_source.json").write_text(json.dumps({"title": "DA and MK Party conclude coalition talks", "channel": "SABC News",
                                                          "description": "Let's cross to SABC News reporter Samkele Maseko."}), encoding="utf-8")
    store = SimpleNamespace(root=tmp_path / "experience", load_corpus=lambda: {"clips": {"c1": {"job_id": "job-1"}}})
    bench = object.__new__(Workbench)
    bench.store = store
    opened: dict = {}
    bench._add_video(opened, "c1")
    assert opened["video"]["title"] == "DA and MK Party conclude coalition talks" and "Samkele Maseko" in opened["video"]["description"] and opened["video"]["note"]
    other: dict = {}
    bench.store = SimpleNamespace(root=tmp_path / "experience", load_corpus=lambda: {"clips": {"c2": {"job_id": "job-none"}}})
    bench._add_video(other, "c2")
    assert "video" not in other                                   # a job that did not come from YouTube has nothing to show
