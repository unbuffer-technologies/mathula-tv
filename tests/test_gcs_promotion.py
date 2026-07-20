import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from mathula_tv.gcs_store import GCSStore


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class Blob:
    """Models google-cloud-storage: exists() does not hydrate metadata."""
    def __init__(self, bucket, name):
        self.bucket, self.name, self.size, self.generation = bucket, name, None, None
    def exists(self): return self.name in self.bucket.data
    def reload(self):
        value, generation = self.bucket.data[self.name]
        self.size, self.generation = len(value), generation
    def download_to_filename(self, path):
        open(path, "wb").write(self.bucket.data[self.name][0])
    def delete(self, if_generation_match=None):
        assert self.bucket.data[self.name][1] == if_generation_match
        del self.bucket.data[self.name]


class Bucket:
    def __init__(self, data=None): self.data = data or {}; self.copy_calls = []
    def blob(self, name): return Blob(self, name)
    def copy_blob(self, source, destination, name, if_generation_match=None):
        self.copy_calls.append((source.name, name, if_generation_match))
        if name in self.data and if_generation_match == 0: raise RuntimeError("412 precondition failed")
        value = self.data[source.name][0]
        self.data[name] = (value, max([g for _, g in self.data.values()] + [0]) + 1)
        return Blob(self, name)  # metadata intentionally remains unloaded
class Client:
    def __init__(self, bucket): self.value=bucket
    def bucket(self,name): return self.value


JOB="job"; TEMP="mathula-tv/jobs/job/result.partial.worker"; FINAL="mathula-tv/jobs/job/result.json"


def store(data):
    bucket=Bucket(data); return GCSStore("bucket",client=Client(bucket)),bucket


def test_existing_nonempty_temporary_reloads_metadata_and_promotes():
    value=b'{"ok":true}'; gcs,bucket=store({TEMP:(value,7)})
    uri=gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(value).hexdigest(),sha256)
    assert uri.endswith("result.json") and bucket.data[FINAL][0]==value and TEMP not in bucket.data
    assert bucket.copy_calls[0][2]==0


def test_missing_and_zero_byte_temporary():
    gcs,_=store({})
    with pytest.raises(ValueError,match="missing"): gcs.promote(JOB,"result.partial.worker","result.json","x",sha256)
    gcs,_=store({TEMP:(b"",1)})
    with pytest.raises(ValueError,match="empty"): gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(b"").hexdigest(),sha256)


def test_temporary_checksum_mismatch_preserves_object():
    gcs,bucket=store({TEMP:(b"value",1)})
    with pytest.raises(ValueError,match="checksum"): gcs.promote(JOB,"result.partial.worker","result.json","0"*64,sha256)
    assert TEMP in bucket.data and FINAL not in bucket.data


def test_final_is_reloaded_and_verified_before_temporary_delete():
    value=b"verified"; gcs,bucket=store({TEMP:(value,2)})
    gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(value).hexdigest(),sha256)
    assert bucket.data[FINAL][0]==value and TEMP not in bucket.data


def test_identical_final_is_idempotent():
    value=b"same"; gcs,bucket=store({TEMP:(value,2),FINAL:(value,3)})
    gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(value).hexdigest(),sha256)
    assert bucket.data[FINAL]==(value,3) and TEMP not in bucket.data and not bucket.copy_calls


def test_conflicting_final_is_rejected_and_temporary_preserved():
    value=b"new"; gcs,bucket=store({TEMP:(value,2),FINAL:(b"old",3)})
    with pytest.raises(ValueError,match="conflicting"): gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(value).hexdigest(),sha256)
    assert TEMP in bucket.data and bucket.data[FINAL][0]==b"old"


def test_stale_nonempty_temporary_can_be_safely_promoted():
    value=b"stale-but-valid"; gcs,bucket=store({TEMP:(value,1)})
    gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(value).hexdigest(),sha256)
    assert bucket.data[FINAL][0]==value


def test_promotion_error_redacts_signed_url_and_credentials():
    class BrokenBucket(Bucket):
        def copy_blob(self,*args,**kwargs):
            raise RuntimeError("failed https://user:password@example.test/object?X-Goog-Signature=secret&safe=yes")
    bucket=BrokenBucket({TEMP:(b"value",1)}); gcs=GCSStore("bucket",client=Client(bucket))
    with pytest.raises(RuntimeError) as error:
        gcs.promote(JOB,"result.partial.worker","result.json",hashlib.sha256(b"value").hexdigest(),sha256)
    message=str(error.value)
    assert "secret" not in message and "password" not in message and "REDACTED" in message
