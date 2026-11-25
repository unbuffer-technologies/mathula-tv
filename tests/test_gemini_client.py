import json

import pytest

from mathula_tv import dub_agent, gemini_client as gc

SIGNED_CALL = {"functionCall": {"name": "measure_tts", "args": {"text": "Sawubona"}, "id": "call_77"}, "thoughtSignature": "SIG=="}


def _history():
    return [
        {"role": "system", "content": "You dub news."},
        {"role": "user", "content": "Dub the clip."},
        {"role": "assistant", "content": None, "_raw": [SIGNED_CALL],
         "tool_calls": [{"id": "call_77", "type": "function", "function": {"name": "measure_tts", "arguments": "{\"text\": \"Sawubona\"}"}}]},
        {"role": "tool", "tool_call_id": "call_77", "content": "{\"measured_ms\": 1180}"},
        {"role": "user", "content": "Note: be shorter."},
    ]


def test_history_becomes_gemini_contents_with_the_signature_replayed_and_results_grouped():
    contents, instruction = gc.to_contents(_history())
    assert instruction == {"parts": [{"text": "You dub news."}]}
    assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
    assert contents[1]["parts"] == [SIGNED_CALL]                                        # the model's own parts, signature included, verbatim
    response = contents[2]["parts"][0]["functionResponse"]
    assert response["name"] == "measure_tts" and response["id"] == "call_77" and response["response"] == {"result": "{\"measured_ms\": 1180}"}
    assert len(contents[2]["parts"]) == 1 and contents[3]["parts"] == [{"text": "Note: be shorter."}]       # a note after the results is a user turn of its OWN: text mixed into the last function-response turn is refused by the API
    assert all("_kind" not in c for c in contents)


def test_several_tool_results_of_one_model_turn_share_one_user_turn():
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "one", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "two", "arguments": "{\"x\": 1}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
    ]
    contents, _ = gc.to_contents(history)
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert [p["functionResponse"]["name"] for p in contents[2]["parts"]] == ["one", "two"]
    assert [p["functionCall"]["name"] for p in contents[1]["parts"]] == ["one", "two"] and contents[1]["parts"][1]["functionCall"]["args"] == {"x": 1}


def test_no_conversation_ever_ends_with_text_mixed_into_a_function_response_turn():
    """The shape the API refused on 2026-11-19 (400 'Requests ending with a model turn are not supported'): harness notes of every kind after tool results, in every order."""
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}}, {"id": "b", "type": "function", "function": {"name": "u", "arguments": "{}"}}]}
    results = [{"role": "tool", "tool_call_id": "a", "content": "A"}, {"role": "tool", "tool_call_id": "b", "content": "B"}]
    for notes in ([], ["first note"], ["first note", "second note"]):
        contents, _ = gc.to_contents([{"role": "user", "content": "go"}, call, *results] + [{"role": "user", "content": n} for n in notes])
        for turn in contents:
            kinds = {"functionResponse" if "functionResponse" in p else "text" for p in turn["parts"]}
            assert len(kinds) == 1, turn                                                    # a turn is all function responses or all text, never both
        assert [p["functionResponse"]["name"] for p in contents[2]["parts"]] == ["t", "u"]          # both results of the model turn stay together
        if notes:
            assert contents[-1]["role"] == "user" and contents[-1]["parts"][0]["text"].startswith("first note") and len(contents) == 4      # consecutive notes share one text turn


def test_tool_specs_pass_their_json_schema_through_and_a_tool_without_parameters_has_none():
    specs = [{"type": "function", "function": {"name": "a", "description": "d", "parameters": {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}}},
             {"type": "function", "function": {"name": "b", "description": "no arguments", "parameters": {"type": "object", "properties": {}}}}]
    declared = gc.to_tools(specs)[0]["functionDeclarations"]
    assert declared[0]["parametersJsonSchema"]["required"] == ["t"] and "parametersJsonSchema" not in declared[1]
    assert gc.to_tools(None) == []


def test_the_answer_becomes_the_chat_shape_with_calls_raw_parts_and_usage():
    answer = {"candidates": [{"content": {"parts": [{"text": "thinking out loud", "thought": True}, SIGNED_CALL]}, "finishReason": "STOP"}],
              "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 300, "cachedContentTokenCount": 40}}
    shaped = gc.to_chat_shape(answer)
    message = shaped["choices"][0]["message"]
    assert shaped["choices"][0]["finish_reason"] == "tool_calls"
    assert message["tool_calls"] == [{"id": "call_77", "type": "function", "function": {"name": "measure_tts", "arguments": "{\"text\": \"Sawubona\"}"}}]
    assert message["_raw"][1] == SIGNED_CALL and message["_reasoning"] == "thinking out loud" and message["content"] is None
    assert shaped["usage"] == {"prompt_tokens": 100, "completion_tokens": 320, "cached_tokens": 40, "thoughts_tokens": 300}          # thinking tokens are billed as output, and reported on their own


def test_a_text_answer_a_cut_off_answer_and_a_blocked_prompt():
    text = gc.to_chat_shape({"candidates": [{"content": {"parts": [{"text": "Sawubona"}]}, "finishReason": "STOP"}]})
    assert text["choices"][0]["message"]["content"] == "Sawubona" and text["choices"][0]["finish_reason"] == "stop"
    cut = gc.to_chat_shape({"candidates": [{"content": {"parts": [{"text": "Saw"}]}, "finishReason": "MAX_TOKENS"}]})
    assert cut["choices"][0]["finish_reason"] == "length"
    with pytest.raises(RuntimeError, match="no candidate"):
        gc.to_chat_shape({"promptFeedback": {"blockReason": "SAFETY"}})
    ids = gc.to_chat_shape({"candidates": [{"content": {"parts": [{"functionCall": {"name": "x", "args": {}}}]}}]})
    assert ids["choices"][0]["message"]["tool_calls"][0]["id"] == "call_0"                  # a call without an id gets a stable one


def test_replies_may_use_the_models_real_output_limit_not_the_azure_default():
    """prod-114: a full-clip submission needed more than the 16,000 the Azure clients allow (thinking tokens count too) and was cut off twice."""
    assert gc.GeminiConfig().max_output_tokens == 65_536 and gc.gemini_config_from_environment({}).max_output_tokens == 65_536
    assert gc.gemini_config_from_environment({"GEMINI_MAX_OUTPUT": "32000"}).max_output_tokens == 32_000
    client = gc.GeminiClient(gc.GeminiConfig(api_key="k"))
    assert client.build_body([{"role": "user", "content": "x"}], tools=None, json_mode=False, max_output_tokens=client.config.max_output_tokens)["generationConfig"]["maxOutputTokens"] == 65_536


def test_complete_uses_the_configured_limit_unless_the_caller_names_one(monkeypatch):
    sent = []

    def fake_post(url, headers, body, *, timeout, retries, sleep):
        sent.append(body["generationConfig"]["maxOutputTokens"])
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}}

    monkeypatch.setattr(dub_agent, "post_with_retry", fake_post)
    client = gc.GeminiClient(gc.GeminiConfig(api_key="k", max_output_tokens=40_000))
    client.complete([{"role": "user", "content": "x"}])
    client.complete([{"role": "user", "content": "x"}], max_output_tokens=500)
    assert sent == [40_000, 500]


def test_the_request_body_sets_thinking_json_mode_and_google_search():
    client = gc.GeminiClient(gc.GeminiConfig(thinking_level="low", api_key="k"))
    tools = [{"type": "function", "function": {"name": "a", "description": "d", "parameters": {"type": "object", "properties": {"t": {"type": "string"}}}}}]
    body = client.build_body(_history(), tools=tools, json_mode=True, max_output_tokens=900)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"} and body["generationConfig"]["maxOutputTokens"] == 900
    assert "responseMimeType" not in body["generationConfig"]                                 # JSON mode and function tools do not mix
    assert body["systemInstruction"]["parts"][0]["text"] == "You dub news."
    plain = client.build_body([{"role": "user", "content": "x"}], tools=None, json_mode=True, max_output_tokens=10)
    assert plain["generationConfig"]["responseMimeType"] == "application/json" and "tools" not in plain
    client.web_budget = 3
    client.reasoning_summary = "detailed"
    searching = client.build_body(_history(), tools=tools, json_mode=False, max_output_tokens=10)
    assert {"googleSearch": {}} in searching["tools"] and searching["generationConfig"]["thinkingConfig"]["includeThoughts"] is True


def test_complete_posts_to_the_right_url_and_counts_the_searches(monkeypatch):
    seen = {}

    def fake_post(url, headers, body, *, timeout, retries, sleep):
        seen.update(url=url, headers=headers, body=body)
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "groundingMetadata": {"webSearchQueries": ["a", "b"]}}], "usageMetadata": {"promptTokenCount": 5}}

    monkeypatch.setattr(dub_agent, "post_with_retry", fake_post)
    client = gc.GeminiClient(gc.GeminiConfig(model="gemini-3.8-flash", api_key="KEY"))
    client.web_budget = 5
    out = client.complete([{"role": "user", "content": "hi"}], max_output_tokens=50)
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent" and seen["headers"]["x-goog-api-key"] == "KEY"
    assert out["choices"][0]["message"]["content"] == "ok" and client.web_used == 2


def test_vertex_url_needs_a_project_and_the_environment_picks_the_provider(monkeypatch):
    config = gc.GeminiConfig(project="mathula-ai", location="global")
    assert config.url() == "https://aiplatform.googleapis.com/v1/projects/mathula-ai/locations/global/publishers/google/models/gemini-3.8-flash:generateContent"
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gc.GeminiConfig().url()
    keyed = gc.gemini_config_from_environment({"GEMINI_API_KEY": "K", "GEMINI_API_FLAVOR": "vertex"})          # a Google Cloud key restricted to Vertex AI: express endpoint, key header, no project
    assert keyed.url() == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.8-flash:generateContent" and keyed.endpoint == "https://aiplatform.googleapis.com"
    assert gc.GeminiClient(keyed)._headers() == {"x-goog-api-key": "K", "Content-Type": "application/json"}
    with pytest.raises(ValueError):
        gc.gemini_config_from_environment({"GEMINI_API_FLAVOR": "azure"})
    env = {"MATHULA_AI_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-3.8-flash", "GOOGLE_CLOUD_PROJECT": "mathula-ai", "GEMINI_THINKING": "low"}
    chosen = dub_agent.native_client_config(env)
    assert isinstance(chosen, gc.GeminiConfig) and chosen.deployment == "gemini-3.8-flash" and chosen.reasoning_effort == "low"
    assert isinstance(dub_agent.new_agent_client(chosen), gc.GeminiClient) and isinstance(dub_agent.new_chat_client(chosen), gc.GeminiClient)
    with pytest.raises(ValueError):
        gc.gemini_config_from_environment({"GEMINI_THINKING": "extreme"})
