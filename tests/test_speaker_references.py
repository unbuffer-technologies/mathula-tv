from mathula_tv.synthesis import select_speaker_references, unresolved_speakers


def test_references_are_isolated_and_overlap_rejected():
    turns=[{"speaker":"A","start":0,"end":4,"overlap":False},{"speaker":"B","start":4,"end":8,"overlap":True},{"speaker":"B","start":8,"end":9,"overlap":False}]
    refs=select_speaker_references(turns); assert list(refs)==["A"] and all(r["speaker"]=="A" for r in refs["A"]); assert unresolved_speakers(turns,refs)==["B"]

