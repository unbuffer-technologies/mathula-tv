import pytest
from mathula_tv.seo import validate_seo


def package(**overrides):
    value={"title":"Isimangalo SikaX: Uthi Kwenzekeni","description":"Incazelo","tags_csv":"X, izindaba","tags":["X","izindaba"],"thumbnail_hook":"Uthi kwenzekeni?","category":"News","primary_domain":"politics","secondary_domains":[],"entities":["X"],"main_entities":["X"],"claim_attributions":["Claim attributed to X"],"context_providers_used":[],"factual_grounding_notes":[],"uncertainty_warnings":[],"human_review_flags":["politics"]}; value.update(overrides); return value


def test_valid_and_limits(): assert validate_seo(package())["title"].startswith("Isimangalo")
def test_unsupported_oversized_title_rejected():
    with pytest.raises(ValueError): validate_seo(package(title="Exposed! "+"x"*100))
