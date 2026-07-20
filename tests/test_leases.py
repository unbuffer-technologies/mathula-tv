import json
from datetime import datetime, timedelta, timezone
import pytest
from mathula_tv.gcs_store import GCSStore, LeaseConflict, LeaseLost, object_name


class Blob:
    def __init__(self, bucket, name): self.bucket, self.name, self.generation, self.size = bucket, name, None, 0
    def exists(self): return self.name in self.bucket.data
    def reload(self): self.generation = self.bucket.data[self.name][1]; self.size = len(self.bucket.data[self.name][0])
    def download_as_text(self): return self.bucket.data[self.name][0]
    def upload_from_string(self, data, **kw):
        expected, current = kw.get("if_generation_match"), self.bucket.data.get(self.name, (None, 0))[1]
        if expected is not None and expected != current: raise RuntimeError("412 precondition failed")
        self.generation = current + 1; self.bucket.data[self.name] = (data, self.generation); self.size = len(data)
class Bucket:
    def __init__(self): self.data = {}
    def blob(self, name): return Blob(self, name)
class Client:
    def __init__(self): self.b = Bucket()
    def bucket(self, name): return self.b


def test_paths_and_duplicate_claim_prevention():
    assert object_name("mathula-tv", "abc", "input/source_audio.wav") == "mathula-tv/jobs/abc/input/source_audio.wav"
    store = GCSStore("bucket", client=Client()); store.claim("abc", "synthesis", "one", 1)
    with pytest.raises(LeaseConflict): store.claim("abc", "synthesis", "two", 1)


def test_expired_lease_reclaim_and_heartbeat():
    client = Client(); store = GCSStore("bucket", client=client); claim = store.claim("abc", "analysis", "one", 1)
    name = store.name("abc", "worker/analysis_claim.json"); data, gen = client.b.data[name]; value = json.loads(data); value["lease_expiry"] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(); client.b.data[name] = (json.dumps(value), gen)
    reclaimed = store.claim("abc", "analysis", "two", 2); assert reclaimed["worker_id"] == "two"
    renewed = store.heartbeat(reclaimed); assert renewed["generation"] > reclaimed["generation"]


def test_generation_safe_lease_completion():
    store = GCSStore("bucket", client=Client()); claim = store.claim("abc", "analysis", "one", 1)
    completed = store.finish_claim(claim, "completed")
    assert completed["status"] == "completed" and completed["generation"] > claim["generation"]
    with pytest.raises(Exception): store.finish_claim(claim, "completed")


def test_expired_lease_is_recoverable_after_promotion_failure():
    client=Client(); store=GCSStore("bucket",client=client); original=store.claim("abc","pyannote_analysis","failed-worker",1)
    name=store.name("abc","worker/pyannote_analysis_claim.json")
    raw,generation=client.b.data[name]; value=json.loads(raw)
    value["lease_expiry"]=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    client.b.data[name]=(json.dumps(value),generation)
    recovered=store.claim("abc","pyannote_analysis","retry-worker",2)
    assert recovered["attempt_number"]==2 and recovered["generation"]>original["generation"]


def test_assert_claim_detects_stale_worker_generation():
    store = GCSStore("bucket", client=Client())
    claim = store.claim("abc", "voice_conversion", "gpu-one", 1)
    assert store.assert_claim(claim)["worker_id"] == "gpu-one"
    renewed = store.heartbeat(claim)
    with pytest.raises(LeaseLost, match="generation changed"):
        store.assert_claim(claim)
    assert store.assert_claim(renewed)["generation"] == renewed["generation"]
