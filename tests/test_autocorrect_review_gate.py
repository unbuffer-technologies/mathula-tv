from mathula_tv.autocorrect_stage import language_ai_review_counts


def test_low_confidence_suggestions_are_advisory_not_blocking():
    assert language_ai_review_counts(
        {
            "review_count": 2,
            "corrections": [
                {"status": "suggested", "source_type": "low_confidence"},
                {"status": "needs_review", "source_type": "low_confidence"},
            ],
        }
    ) == {
        "blocking_review_count": 0,
        "advisory_review_count": 2,
    }


def test_explicit_confusion_and_ambiguity_remain_blocking():
    assert language_ai_review_counts(
        {
            "review_count": 3,
            "corrections": [
                {"status": "suggested", "source_type": "confusion_hint"},
                {"status": "needs_review", "source_type": "ambiguous_hint"},
                {"status": "suggested", "source_type": "low_confidence"},
            ],
        }
    ) == {
        "blocking_review_count": 2,
        "advisory_review_count": 1,
    }
