import json
from mathula_tv.context_providers import CommissionContextProvider, PoliticsContextProvider, providers_for


def cls(domain, secondary=[]): return {"primary_domain":domain,"secondary_domains":secondary}


def test_missing_context_warns(tmp_path):
    result = CommissionContextProvider(tmp_path/"missing").enrich(cls("commission"), "witness")
    assert not result["enrichment_applied"] and result["warnings"]


def test_commission_excluded_from_unrelated(tmp_path):
    path=tmp_path/"case.json"; path.write_text('{"fact":"commission secretariat"}')
    result=CommissionContextProvider(path).enrich(cls("general_news"), "weather")
    assert not result["enrichment_applied"] and result["source_checksum"] is None


def test_relevant_retrieval_and_multiple_providers(tmp_path):
    case=tmp_path/"case.json"; politics=tmp_path/"politics.json"; case.write_text('{"facts":[{"fact":"Witness Alpha appeared"},{"fact":"Unrelated Beta"}]}'); politics.write_text('{"verified_facts":[]}')
    result=CommissionContextProvider(case).enrich(cls("commission"), "Witness Alpha")
    assert result["enrichment_applied"] and len(result["relevant_facts"]) == 1
    assert len(providers_for(cls("mixed", ["commission","politics"]), case, politics)) == 2

