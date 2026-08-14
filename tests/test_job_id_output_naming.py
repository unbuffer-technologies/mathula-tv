from __future__ import annotations
import json
from pathlib import Path
from mathula_tv.output_naming import load_seo_output_context,publish_job_id_transcript_alias,publish_title_named_outputs

def test_job_id_named_manual_artifacts(tmp_path:Path)->None:
    job_id="1d3c57ad6f0b4718a637c14626ee6857"; root=tmp_path/job_id
    seo=root/"translation/tiktok_zu.json"; seo.parent.mkdir(parents=True)
    seo.write_text(json.dumps({"cover_hook":"SEO hook","tiktok":{"caption":"cap","hashtags":["#one"]},"search_keywords":["one"]}),encoding="utf-8")
    video=root/"direct_dub/final_dubbed.mp4"; video.parent.mkdir(parents=True); video.write_bytes(b"video")
    transcript=root/"analysis/transcript_en.json"; transcript.parent.mkdir(parents=True); transcript.write_text('{"segments":[]}',encoding="utf-8")
    context=load_seo_output_context(root,"zu-ZA")
    outputs=publish_title_named_outputs(root,video,context=context)
    assert outputs.final_video.name==f"final_dubbed_{job_id}.mp4"
    names={p.name for p in outputs.seo_files.values()}
    assert f"tiktok_caption_zu_{job_id}.txt" in names
    assert f"tiktok_cover_zu_{job_id}.txt" in names
    assert f"tiktok_seo_zu_{job_id}.json" in names
    assert not any(name.startswith("youtube_") for name in names)
    alias=publish_job_id_transcript_alias(root)
    assert alias.name==f"transcript_en_{job_id}.json"
    assert alias.read_bytes()==transcript.read_bytes()


def test_flat_manual_seo_fields_are_published_with_content(tmp_path: Path) -> None:
    job_id = "1d3c57ad6f0b4718a637c14626ee6857"
    root = tmp_path / job_id
    seo = root / "translation/tiktok_zu.json"
    seo.parent.mkdir(parents=True)
    seo.write_text(
        json.dumps(
            {
                "cover_hook": "Umbhalo wesithombe",
                "search_keywords": ["izindaba", "isiZulu"],
                "tiktok_caption": "Umbhalo weTikTok",
                "tiktok_hashtags": [
                    "#BantufyZulu",
                    "#MathulaTV",
                    "#PKTT",
                    "#NathiMthethwa",
                    "#MadlangaCommission",
                    "#SAPS",
                ],
            }
        ),
        encoding="utf-8",
    )
    video = root / "direct_dub/final_dubbed.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    outputs = publish_title_named_outputs(
        root,
        video,
        context=load_seo_output_context(root, "zu-ZA"),
    )

    caption = outputs.seo_files["tiktok_caption"].read_text(encoding="utf-8")
    assert caption.startswith("Umbhalo weTikTok\n\n#ZuluTikTok ")
    assert "#PKTT" in caption
    assert "#NathiMthethwa" in caption
    assert "#MadlangaCommission" in caption
    assert "#SAPS" in caption
    assert "#Mzansi" not in caption
    assert "#SouthAfrica" not in caption
    assert "#NgesiZulu" not in caption
    assert len([token for token in caption.split() if token.startswith("#")]) == 5
    hashtags = outputs.seo_files["tiktok_hashtags"].read_text(encoding="utf-8")
    assert "#ZuluTikTok" in hashtags
    assert "#BantufyZulu" not in hashtags
    assert "#MathulaTV" not in hashtags
    assert len(hashtags.split()) == 5
    assert (
        outputs.seo_files["tiktok_cover"].read_text(encoding="utf-8").strip()
        == "Umbhalo wesithombe"
    )
    assert (
        outputs.seo_files["tiktok_search_keywords"]
        .read_text(encoding="utf-8")
        .strip()
        == "izindaba, isiZulu"
    )


def test_tiktok_brand_profile_follows_target_language(tmp_path: Path) -> None:
    profiles = {
        "zu-ZA": ("#ZuluTikTok", "#NgesiZulu"),
        "nso-ZA": ("#SepediTikTok", "#KaSepedi"),
        "st-ZA": ("#SesothoTikTok", "#KaSesotho"),
        "tn-ZA": ("#SetswanaTikTok", "#KaSetswana"),
        "ve-ZA": ("#VendaTikTok", "#NgaTshivenda"),
        "xh-ZA": ("#XhosaTikTok", "#NgesiXhosa"),
        "ts-ZA": ("#TsongaTikTok", "#Hi_Xitsonga"),
        "ss-ZA": ("#SwatiTikTok", "#NgesiSwati"),
        "nr-ZA": ("#NdebeleTikTok", "#NgesiNdebele"),
    }
    for index, (locale, hashtags) in enumerate(profiles.items()):
        community_hashtag, _value_hashtag = hashtags
        root = tmp_path / f"jobid{index:08d}"
        code = locale.split("-", 1)[0]
        seo = root / f"translation/tiktok_{code}.json"
        seo.parent.mkdir(parents=True)
        seo.write_text(
            json.dumps(
                {
                    "cover_hook": "SEO hook",
                    "tiktok_caption": "English searchable caption",
                    "tiktok_hashtags": [
                        "#PersonName",
                        "#Institution",
                        "#Commission",
                        "#CaseTopic",
                    ],
                }
            ),
            encoding="utf-8",
        )
        video = root / "direct_dub/final_dubbed.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")

        outputs = publish_title_named_outputs(
            root,
            video,
            context=load_seo_output_context(root, locale),
        )

        caption = outputs.seo_files["tiktok_caption"].read_text(encoding="utf-8")
        assert f"\n\n{community_hashtag} " in caption
        assert _value_hashtag not in caption
        assert len([token for token in caption.split() if token.startswith("#")]) == 5


def test_tiktok_caption_replaces_existing_hashtags_and_never_exceeds_five(
    tmp_path: Path,
) -> None:
    job_id = "captionlimit0001"
    root = tmp_path / job_id
    seo = root / "translation/tiktok_zu.json"
    seo.parent.mkdir(parents=True)
    seo.write_text(
        json.dumps(
            {
                "cover_hook": "SEO hook",
                "tiktok_caption": "Caption #old1 #old2 #old3 #old4 #old5 #old6",
                "tiktok_hashtags": [
                    "#TopicOne",
                    "#TopicTwo",
                    "#TopicThree",
                    "#TopicFour",
                ],
            }
        ),
        encoding="utf-8",
    )
    video = root / "direct_dub/final_dubbed.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    outputs = publish_title_named_outputs(
        root,
        video,
        context=load_seo_output_context(root, "zu-ZA"),
    )

    caption = outputs.seo_files["tiktok_caption"].read_text(encoding="utf-8")
    hashtag_file = outputs.seo_files["tiktok_hashtags"].read_text(encoding="utf-8")
    assert len([token for token in caption.split() if token.startswith("#")]) == 5
    assert len(hashtag_file.split()) == 5
    assert "#old1" not in caption
    assert caption.startswith("Caption\n\n#ZuluTikTok")


def test_dubbed_master_has_unambiguous_nonfinal_name(tmp_path: Path) -> None:
    from mathula_tv.output_naming import publish_dubbed_master_outputs

    job_id = "1d3c57ad6f0b4718a637c14626ee6857"
    root = tmp_path / job_id
    seo = root / "translation/tiktok_zu.json"
    seo.parent.mkdir(parents=True)
    seo.write_text(json.dumps({"cover_hook": "SEO hook"}), encoding="utf-8")
    canonical = root / "direct_dub/dubbed_master.mp4"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"master")

    outputs = publish_dubbed_master_outputs(
        root,
        canonical,
        context=load_seo_output_context(root, "zu-ZA"),
    )

    assert outputs.final_video.name == f"dubbed_master_{job_id}.mp4"
    assert not (root / "output" / f"final_dubbed_{job_id}.mp4").exists()
