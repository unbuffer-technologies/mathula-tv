from types import SimpleNamespace

import pytest

from mathula_tv.translation import AzureAIError, AzureOpenAIClient, azure_config_from_environment


def test_configuration_aliases_and_precedence():
    env={"AZURE_AI_ENDPOINT":"https://primary","AZURE_OPENAI_ENDPOINT":"https://fallback","AZURE_AI_KEY":"primary-key","AZURE_OPENAI_API_KEY":"fallback-key","AZURE_AI_DEPLOYMENT":"gpt-5.4","AZURE_OPENAI_CHAT_DEPLOYMENT":"chat","AZURE_AI_API_VERSION":"v1"}
    value=azure_config_from_environment(env)
    assert value=={"endpoint":"https://primary","api_key":"primary-key","deployment":"gpt-5.4","api_version":"v1"}
    alias=azure_config_from_environment({"AZURE_OPENAI_ENDPOINT":"https://alias","AZURE_OPENAI_KEY":"alias-key","AZURE_OPENAI_CHAT_DEPLOYMENT":"gpt-5.4"})
    assert alias["api_key"]=="alias-key" and alias["deployment"]=="gpt-5.4"


def test_missing_key_names_only_without_secret_disclosure():
    with pytest.raises(ValueError) as error: azure_config_from_environment({"AZURE_AI_ENDPOINT":"https://endpoint","AZURE_AI_DEPLOYMENT":"gpt-5.4"})
    assert "AZURE_AI_KEY" in str(error.value) and "https://endpoint" not in str(error.value)


class Responses:
    def __init__(self,values): self.values=list(values); self.calls=[]
    def create(self,**kwargs):
        self.calls.append(kwargs); value=self.values.pop(0)
        if isinstance(value,Exception): raise value
        return SimpleNamespace(output_text=value)


def test_malformed_json_is_repaired():
    responses=Responses(["not json",'{"ok":true}']); client=AzureOpenAIClient("https://endpoint","secret","gpt-5.4",sdk_client=SimpleNamespace(responses=responses))
    assert client.complete_json("prompt",{"safe":True})=={"ok":True} and client.repair_attempts==1
    assert "malformed_response_to_repair" in responses.calls[1]["input"][1]["content"]


@pytest.mark.parametrize("status,retryable",[(401,False),(503,True)])
def test_auth_is_permanent_and_transient_exhaustion_is_retryable(status,retryable):
    error=RuntimeError("secret-bearing upstream detail"); error.status_code=status
    count=1 if status==401 else 3
    client=AzureOpenAIClient("https://endpoint","actual-secret","gpt-5.4",sdk_client=SimpleNamespace(responses=Responses([error]*count)),sleep=lambda _:None)
    with pytest.raises(AzureAIError) as caught: client.complete_json("prompt",{})
    assert caught.value.retryable is retryable and "actual-secret" not in str(caught.value) and "upstream detail" not in str(caught.value)
