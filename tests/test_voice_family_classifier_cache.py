import mathula_tv.voice_family_classifier as classifier


def test_runtime_model_load_is_pinned_and_local_only(monkeypatch):
    calls = []

    class _Model:
        def eval(self):
            return self

    def fake_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _Model()

    monkeypatch.setattr(
        classifier.ECAPA_gender,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(classifier, "_model_cache", None)

    first = classifier._load_model()
    second = classifier._load_model()

    assert first is second
    assert calls == [
        (
            classifier.VOICE_GENDER_MODEL_ID,
            {
                "revision": classifier.VOICE_GENDER_MODEL_REVISION,
                "local_files_only": True,
            },
        )
    ]
