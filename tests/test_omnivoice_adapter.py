from types import SimpleNamespace

import numpy as np
import pytest

from mathula_tv.omnivoice_adapter import RealOmniVoiceAdapter


class SF:
    def __init__(self): self.calls=[]
    def write(self,path,samples,rate,subtype=None): self.calls.append((path,samples,rate,subtype))
class CUDA:
    @staticmethod
    def is_available(): return True
    @staticmethod
    def empty_cache(): pass
class Torch:
    cuda=CUDA(); float16="fp16"
class Model:
    loads=[]; output=np.ones(2400,dtype=np.float32)*.2
    @classmethod
    def from_pretrained(cls,*args,**kwargs): cls.loads.append((args,kwargs)); return cls()
    def generate(self,**kwargs): self.kwargs=kwargs; return [self.output]
    def create_voice_clone_prompt(self,**kwargs): self.prompt_kwargs=kwargs; return "prompt"
class Config:
    def __init__(self,**kwargs): self.values=kwargs


def adapter(tmp_path,output=None):
    Model.loads=[]
    if output is not None: Model.output=output
    ref=tmp_path/"ref.wav"; ref.write_bytes(b"wav")
    sf=SF(); value=RealOmniVoiceAdapter(model_class=Model,generation_config_class=Config,torch_module=Torch(),soundfile_module=sf)
    return value,ref,sf


def test_model_load_configuration_and_loaded_once(tmp_path):
    value,ref,sf=adapter(tmp_path); value.load(); value.load()
    assert len(Model.loads)==1 and Model.loads[0][0]==("k2-fsa/OmniVoice",)
    assert Model.loads[0][1]=={"device_map":"cuda:0","dtype":"fp16","load_asr":True}


def test_numpy_output_mono_and_24khz(tmp_path):
    value,ref,sf=adapter(tmp_path,np.ones((2,2400),dtype=np.float32)*.2)
    result=value.synthesize("Sawubona",ref,"Hello",tmp_path/"out.wav")
    assert result.sample_rate==24000 and result.duration==.1 and sf.calls[0][1].ndim==1 and sf.calls[0][2]==24000
    assert value.model.kwargs["language"]=="zu" and value.model.kwargs["voice_clone_prompt"]=="prompt"
    assert value.model.prompt_kwargs["ref_text"]=="Hello"


def test_locale_mapping_and_unsupported_language():
    assert RealOmniVoiceAdapter.resolve_language_id("zu-ZA")=="zu"
    with pytest.raises(ValueError,match="Unsupported"): RealOmniVoiceAdapter(target_locale="sw-KE")


class Tensor:
    def __init__(self,value): self.value=value
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def test_torch_tensor_output_handling(tmp_path):
    value,ref,sf=adapter(tmp_path,Tensor(np.ones(2400)*.2)); result=value.synthesize("Zulu",ref,"English",tmp_path/"x.wav")
    assert result.duration==.1


@pytest.mark.parametrize("output,message",[(np.zeros(100),"silent"),(np.array([0.,np.nan]),"non-finite"),(np.array([1.2,.2]),"clipping")])
def test_invalid_audio_rejected(tmp_path,output,message):
    value,ref,_=adapter(tmp_path,output)
    with pytest.raises(ValueError,match=message): value.synthesize("text",ref,"reference",tmp_path/"x.wav")
