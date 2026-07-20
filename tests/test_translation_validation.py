import pytest
from mathula_tv.translation import translate, validate_translation


SOURCE={"segments":[{"segment_id":"seg-00001","speaker":"A","start":0,"end":1,"source_text":"He alleged it happened"}]}
def turn(text="Uthe kusolwa ukuthi kwenzeka"):
    return {"segment_id":"seg-00001","speaker":"A","start":0,"end":1,"source_text":"He alleged it happened","translated_text":text,"target_duration":1,"delivery_style":"forceful","emotional_intensity":.7,"statement_type":"allegation","attribution_required":True,"uncertainty_flags":[],"pronunciation_hints":[]}
class Client:
    deployment="gpt-5.4"
    def __init__(self): self.calls=0
    def complete_json(self, system_prompt, payload):
        self.calls+=1
        return {"turns": ([] if self.calls==1 else [turn()])}


def test_validation_repair_and_attribution_preserved():
    client=Client(); result=translate(client,SOURCE,[]); assert client.calls==2 and result["turns"][0]["statement_type"]=="allegation" and result["turns"][0]["delivery_style"]=="forceful"


def test_missing_segment_rejected():
    with pytest.raises(ValueError): validate_translation({"turns":[]},SOURCE["segments"])
