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
