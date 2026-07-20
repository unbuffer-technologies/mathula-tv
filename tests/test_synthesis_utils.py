import wave

from mathula_tv.synthesis import assemble_timeline,extract_reference_wav,plan_synthesis_units,reference_text,select_speaker_references


def wav(path,rate=24000,seconds=1,value=1000):
    with wave.open(str(path),"wb") as w: w.setparams((1,2,rate,0,"NONE","none")); w.writeframes(value.to_bytes(2,"little",signed=True)*int(rate*seconds))


def test_reference_text_is_same_speaker_and_exact_interval():
    words=[{"speaker":"A","start":0,"end":1,"text":"Exact"},{"speaker":"B","start":1,"end":2,"text":"Wrong"},{"speaker":"A","start":2,"end":3,"text":"words"}]
    assert reference_text(words,0,3,"A")=="Exact words"


def test_reference_selection_bounds_and_extraction(tmp_path):
    selected=select_speaker_references([{"speaker":"A","start":0,"end":20,"overlap":False},{"speaker":"B","start":20,"end":25,"overlap":True}])
    assert selected["A"][0]["duration"]==15 and "B" not in selected
    source=tmp_path/"source.wav"; out=tmp_path/"ref.wav"; wav(source,16000,20)
    extract_reference_wav(source,out,2,6)
    with wave.open(str(out),"rb") as w: assert w.getframerate()==16000 and w.getnframes()==64000


def test_timeline_extends_instead_of_cutting_words(tmp_path):
    clip=tmp_path/"clip.wav"; output=tmp_path/"out.wav"; wav(clip,24000,2)
    result=assemble_timeline([{"path":clip,"start":1.5,"duration":2,"segment_id":"s"}],output,2,24000)
    assert result["final_duration"]==3.5 and result["extended"]


def test_sentence_splitting_preserves_identity_text_and_timing():
    turn={"segment_id":"s1","speaker":"A","start":10.,"end":30.,"source_text":"source","translated_text":"Umusho wokuqala. Umusho wesibili!","delivery_style":"strong","emotional_intensity":"high"}
    units=plan_synthesis_units(turn)
    assert len(units)==2 and all(x["segment_id"]=="s1" and x["speaker"]=="A" and x["translated_text"]==turn["translated_text"] for x in units)
    assert units[0]["start"]==10 and units[-1]["end"]==30 and units[0]["end"]==units[1]["start"]
