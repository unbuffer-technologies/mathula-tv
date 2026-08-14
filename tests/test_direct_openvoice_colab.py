from mathula_tv.direct_openvoice_colab import PLAN_SCHEMA, MANIFEST_SCHEMA

def test_direct_openvoice_schemas_are_explicit():
    assert PLAN_SCHEMA.endswith('plan-v1')
    assert MANIFEST_SCHEMA.endswith('manifest-v1')

def test_worker_uses_original_reference_and_azure_source():
    text=open('scripts/run_direct_openvoice_colab_worker.py',encoding='utf-8').read()
    assert "target_reference" in text
    assert "source_audio" in text
    assert "ToneColorConverter" in text
