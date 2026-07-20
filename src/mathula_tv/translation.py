from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from pathlib import Path

import jsonschema


TRANSLATION_PROMPT_VERSION = "translation-zu-v2"


class JsonClient(Protocol):
    deployment: str
    api_version: str
    request_durations: list[float]
    repair_attempts: int
    def complete_json(self, system_prompt: str, payload: dict) -> dict: ...


class AzureAIError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message); self.retryable = retryable


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict): raise ValueError("Model JSON response must be an object")
    return value


def azure_config_from_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    env = environ if environ is not None else os.environ
    endpoint = env.get("AZURE_AI_ENDPOINT") or env.get("AZURE_OPENAI_ENDPOINT") or ""
    key = env.get("AZURE_AI_KEY") or env.get("AZURE_OPENAI_API_KEY") or env.get("AZURE_OPENAI_KEY") or ""
    deployment = env.get("AZURE_AI_DEPLOYMENT") or env.get("AZURE_OPENAI_CHAT_DEPLOYMENT") or env.get("AZURE_OPENAI_DEPLOYMENT") or ""
    api_version = env.get("AZURE_AI_API_VERSION") or env.get("AZURE_OPENAI_API_VERSION") or "preview"
    missing = []
    if not endpoint: missing.append("AZURE_AI_ENDPOINT or AZURE_OPENAI_ENDPOINT")
    if not key: missing.append("AZURE_AI_KEY or AZURE_OPENAI_API_KEY or AZURE_OPENAI_KEY")
    if not deployment: missing.append("AZURE_AI_DEPLOYMENT or AZURE_OPENAI_CHAT_DEPLOYMENT")
    if missing: raise ValueError("Missing Azure OpenAI configuration: " + ", ".join(missing))
    return {"endpoint": endpoint.rstrip("/"), "api_key": key, "deployment": deployment, "api_version": api_version}


@dataclass
class AzureOpenAIClient:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "preview"
    retries: int = 3
    json_repairs: int = 2
    timeout_seconds: float = 600
    sdk_client: Any = None
    sleep: Any = time.sleep
    request_durations: list[float] = field(default_factory=list)
    repair_attempts: int = 0

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None):
        config = azure_config_from_environment(environ)
        return cls(
            endpoint=config["endpoint"],
            api_key=config["api_key"],
            deployment=config["deployment"],
            api_version=config["api_version"],
        )

    def _client(self):
        if self.sdk_client is not None: return self.sdk_client
        try:
            from openai import OpenAI
        except ImportError as exc: raise RuntimeError("Install openai>=1.40") from exc
        self.sdk_client = OpenAI(base_url=f"{self.endpoint}/openai/v1/", api_key=self.api_key, default_query={"api-version": self.api_version}, timeout=self.timeout_seconds)
        return self.sdk_client

    def complete_json(self, system_prompt: str, payload: dict) -> dict:
        prompt, invalid, use_json_format = system_prompt, None, True
        network_attempt = 0
        while True:
            try:
                request_payload = dict(payload)
                if invalid is not None: request_payload["malformed_response_to_repair"] = invalid[:12000]
                started = time.perf_counter()
                kwargs={"model":self.deployment,"input":[{"role":"system","content":prompt},{"role":"user","content":json.dumps(request_payload,ensure_ascii=False)}],"max_output_tokens":50000}
                if use_json_format: kwargs["text"]={"format":{"type":"json_object"}}
                response = self._client().responses.create(**kwargs)
                self.request_durations.append(time.perf_counter() - started)
                try: return extract_json(response.output_text)
                except (ValueError, json.JSONDecodeError) as exc:
                    if self.repair_attempts >= self.json_repairs:
                        raise AzureAIError(f"Azure OpenAI returned malformed JSON after {self.json_repairs} repairs", retryable=False) from exc
                    self.repair_attempts += 1; invalid = response.output_text
                    prompt = system_prompt + "\nRepair the previous response into strict complete JSON. Do not omit any required item."
                    continue
            except AzureAIError: raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                lowered=str(exc).lower()
                if status==400 and use_json_format and any(term in lowered for term in ("json_object","text.format","response_format","json schema")):
                    use_json_format=False
                    continue
                permanent = status in {400, 401, 403, 404, 422}
                if permanent:
                    label = "authentication/authorization failed" if status in {401,403} else f"request rejected with HTTP {status}"
                    raise AzureAIError(f"Azure OpenAI {label}", retryable=False) from exc
                network_attempt += 1
                if network_attempt >= self.retries:
                    raise AzureAIError(f"Azure OpenAI request failed after {self.retries} attempts: {type(exc).__name__}", retryable=True) from exc
                self.sleep(min(2 ** (network_attempt - 1), 30))


TRANSLATION_PROMPT = """You are GPT-5.4 acting as a politically neutral senior South African broadcast translator. Translate every supplied English speaker segment into natural broadcast isiZulu. Ground yourself only in what was spoken; selected context is background and may never override the transcript. Preserve exactly every segment ID, speaker, start/end timestamp and source text. Preserve names, institutions, numbers, quotations, attribution, uncertainty, allegations/opinions/promises, sensational energy, emotional force, humour, sarcasm and rhetorical style without inventing or strengthening claims. Do not neutralize a sensational speaker. Return strict JSON with schema_version, language and turns. Each turn must contain segment_id, speaker, start, end, source_text, translated_text, target_duration, delivery_style, emotional_intensity, statement_type, attribution_required, uncertainty_flags and pronunciation_hints."""


REQUIRED_TURN = {"segment_id","speaker","start","end","source_text","translated_text","target_duration","delivery_style","emotional_intensity","statement_type","attribution_required","uncertainty_flags","pronunciation_hints"}


def validate_translation(data: dict, source_segments: list[dict]) -> dict:
    turns = data.get("turns")
    if not isinstance(turns,list): raise ValueError("Translation turns must be a list")
    source_by_id = {s["segment_id"]:s for s in source_segments}
    if len(turns) != len(source_segments) or {t.get("segment_id") for t in turns} != set(source_by_id):
        raise ValueError("Translation segment IDs do not match source")
    for turn in turns:
        missing=REQUIRED_TURN-turn.keys()
        if missing or not str(turn.get("translated_text","")).strip(): raise ValueError(f"Invalid translated turn; missing {sorted(missing)}")
        source=source_by_id[turn["segment_id"]]
        for key in ("speaker","start","end","source_text"):
            if turn[key] != source[key]: raise ValueError(f"Translation did not preserve {key} for {turn['segment_id']}")
        if float(turn["target_duration"]) <= 0: raise ValueError("Translation target duration must be positive")
        if str(turn["statement_type"]).lower() in {"allegation","accusation","unverified_claim"} and not turn["attribution_required"]:
            raise ValueError("Allegations must retain attribution")
        if not isinstance(turn["uncertainty_flags"],list) or not isinstance(turn["pronunciation_hints"],list):
            raise ValueError("Translation flags and pronunciation hints must be lists")
    data["schema_version"]="translation-v1"; data["language"]="zu-ZA"
    schema=json.loads((Path(__file__).resolve().parents[2]/"schemas/translation.schema.json").read_text())
    jsonschema.validate(data,schema)
    return data


def translate(client: JsonClient, transcript: dict, context: list[dict], repair_retries: int = 2) -> dict:
    payload={"source_language":"en-ZA","target_language":"zu-ZA","turns":transcript["segments"],"grounded_context":context}
    duration_index=len(getattr(client,"request_durations",[]))
    error=None
    for attempt in range(repair_retries+1):
        result=client.complete_json(TRANSLATION_PROMPT+(f"\nRepair this validation error: {error}" if error else ""),payload)
        try:
            for turn in result.get("turns",[]):
                turn.setdefault("translation_deployment",client.deployment); turn.setdefault("prompt_version",TRANSLATION_PROMPT_VERSION)
            result=validate_translation(result,transcript["segments"]); result["repair_attempts"]=attempt+getattr(client,"repair_attempts",0)
            result["request_duration_seconds"]=round(sum(getattr(client,"request_durations",[])[duration_index:]),3)
            return result
        except (ValueError,KeyError,jsonschema.ValidationError) as exc:
            error=str(exc); payload["invalid_response"]=result
    raise ValueError(f"Translation validation failed after repair: {error}")
