from pathlib import Path

from mathula_tv import human_translations as ht
from mathula_tv.dub_agent import TOOL_SPECS

PAIRS = [
    ht.Pair("Good afternoon ladies and gentlemen .", "Ngiyanibingelela nonke bantu bakithi .", "Autshumato 2022 (government)"),
    ht.Pair("Good afternoon .", "Sanibonani .", "Umsuka (news)"),
    ht.Pair("Good afternoon .", "UMasipala wethu usebenza kanzima kanti inhloso yethu ukushintsha ngokushesha izimpilo zabantu baseThekwini kakhulu .", "Autshumato 2022 (government)"),   # shifted by a line
    ht.Pair("GOOD AFTERNOON COMMITTEE MEMBERS", "SANIBONANI BAMALUNGU OKOMKHANDLU", "Autshumato 2022 (government)"),
    ht.Pair("The press briefing was held on Monday .", "Isithangami sabezindaba sibanjwe ngoMsombuluko .", "Umsuka (news)"),
    ht.Pair("A press briefing followed the meeting of the national executive .", "Kwalandela isithangami sabezindaba emhlanganweni wekomidi likazwelonke .", "Umsuka (news)"),
    ht.Pair("She grabbed the microphone and got them to chant my name .", "Wathatha i-microphone wamemeza igama lami .", "Autshumato 2022 (government)"),
]
# a term with six usable pairs (and one shifted by a line) so that the Zulu they share can be found
COST = [ht.Pair(f"The cost of living is rising in {place} .", f"Izindleko zokuphila {verb} e{place.lower()} .", "Autshumato 2022 (government)")
        for place, verb in (("Durban", "ziyanda"), ("Pretoria", "ziyakhula"), ("Soweto", "ziyenyuka"), ("Polokwane", "zinyuka"), ("Mthatha", "ziyaqhubeka"), ("Kimberley", "zikhula"))]
COST.append(ht.Pair("The cost of living was discussed .", "Umgubho wawubanjwe ngoMgqibelo ekhaya lemizi yezindaba zasekhaya kakhulu lapha ngempela nje kodwa futhi kuthe .", "Autshumato 2022 (government)"))
FILLER = [ht.Pair(f"Another sentence number {i} about nothing .", f"Elinye igama {i} ngalutho nje kanjani .", "Umsuka (news)") for i in range(400)]


def _db(tmp_path, extra=()):
    path = tmp_path / "pairs.sqlite"
    ht.build(path, PAIRS + list(extra))
    return ht.PairsDB(path)


def test_lookup_finds_the_phrase_shortest_english_first_and_drops_shifted_and_capitalised_pairs(tmp_path):
    db = _db(tmp_path)
    found = db.lookup("Good afternoon", limit=5)
    assert found["pairs_in_all"] == 4                                   # the shifted pair and the capitals pair are counted but not shown
    english = [p["english"] for p in found["pairs"]]
    assert english == ["Good afternoon .", "Good afternoon ladies and gentlemen ."]
    assert found["pairs"][0]["isizulu"] == "Sanibonani ." and found["pairs"][0]["source"] == "Umsuka (news)"
    assert all("UMasipala" not in p["isizulu"] and not p["english"].isupper() for p in found["pairs"])
    assert db.count("press briefing") == 2 and db.count("no such phrase here") == 0


def test_a_missing_database_says_so_instead_of_failing(tmp_path):
    db = ht.PairsDB(tmp_path / "nothing.sqlite")
    assert not db.available and "not installed" in db.lookup("good afternoon")["error"] and db.count("good afternoon") == 0
    assert db.consensus("good afternoon") == [] and db.hints([("seg-1", "Good afternoon.")]) == []


def test_consensus_is_what_the_zulu_of_the_pairs_shares_and_a_shifted_pair_is_outvoted(tmp_path):
    db = _db(tmp_path, COST + FILLER)
    found = db.consensus("cost of living")
    assert found and found[0]["isizulu"] == "izindleko zokuphila" and found[0]["in_pairs"] == 6 and found[0]["of"] == 6      # the 7th pair (shifted, long Zulu) is dropped by the ratio filter
    assert db.consensus("good afternoon") == []                           # too few pairs: no consensus is not evidence of anything
    looked = db.lookup("cost of living", limit=2)
    assert looked["usually_written"][0]["isizulu"] == "izindleko zokuphila" and len(looked["pairs"]) == 2


def test_hints_show_the_exact_line_for_a_stock_phrase_and_the_consensus_for_a_long_sentence(tmp_path):
    db = _db(tmp_path, COST + FILLER)
    hints = db.hints([("seg-2", "Good afternoon."), ("seg-9", "People say that the cost of living keeps rising in the cities, honestly speaking."), ("seg-10", "Nothing here is known.")])
    by_seg = {h["segment_id"]: h for h in hints}
    assert by_seg["seg-2"]["exact_pairs"][0]["isizulu"] == "Sanibonani ." and by_seg["seg-2"]["phrase"] == "good afternoon"
    assert by_seg["seg-9"]["phrase"] == "cost of living" and by_seg["seg-9"]["usually_written"][0]["isizulu"] == "izindleko zokuphila"
    assert "seg-10" not in by_seg
    assert len(db.hints([("s", "Good afternoon.")] * 100, max_total=3)) <= 3


def test_single_words_and_function_word_phrases_are_not_hinted(tmp_path):
    db = _db(tmp_path, COST + FILLER)
    assert db.hints([("seg-3", "It is of the 2026 one."), ("seg-4", "Important.")]) == []


def test_the_agent_has_the_tool():
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert "human_translations" in names and "attested_zulu" in names
    spec = next(t for t in TOOL_SPECS if t["function"]["name"] == "human_translations")["function"]
    assert spec["parameters"]["required"] == ["phrase"] and "limit" in spec["parameters"]["properties"]


def test_the_workbench_reads_the_pairs_from_the_resources_next_to_the_experience_store(tmp_path):
    from mathula_tv.dub_workbench import Workbench

    work = Path(tmp_path)
    ht.build(work / "resources" / "grammar" / "human_pairs.sqlite", PAIRS)

    class Bench:
        class store:
            root = work / "experience"

        human_translations = Workbench.human_translations
        _pairs_db = Workbench._pairs_db

    found = Bench().human_translations("good afternoon", limit=3)
    assert found["pairs"][0]["isizulu"] == "Sanibonani ." and found["pairs_in_all"] == 4
    assert len(Bench().human_translations("good afternoon", limit=99)["pairs"]) <= 8            # the limit is bounded
    Bench.store.root = work / "elsewhere" / "experience"
    assert "not installed" in Bench().human_translations("good afternoon")["error"]



def test_masakhane_pairs_are_loaded_from_each_format_and_machine_translated_sets_are_not(tmp_path):
    import json as _json

    import pytest

    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "masakhane"
    (root / "AfriDocMT" / "data" / "sentence" / "Health").mkdir(parents=True)
    (root / "AfriDocMT" / "data" / "sentence" / "Health" / "train.csv").write_text("am,en,ha,sw,yo,zu\nx,Patients should rest.,x,x,x,Iziguli kufanele ziphumule.\nx,,x,x,x,Isikhala\n", encoding="utf-8")
    (root / "afriscience_mt" / "corpus").mkdir(parents=True)
    pq.write_table(pa.table({"lang_pair": ["eng-zul", "eng-hau"], "source": ["We sampled soil.", "We sampled soil."], "target": ["Sathatha isampula yenhlabathi.", "x"]}), root / "afriscience_mt" / "corpus" / "train-00000.parquet")
    (root / "afriscience_mt" / "glossary").mkdir(parents=True)
    pq.write_table(pa.table({"target_lang": ["zul", "hau"], "eng": ["flora", "flora"], "target": ["izitshalo", "x"]}), root / "afriscience_mt" / "glossary" / "terms-00000.parquet")
    (root / "mafand" / "en-zul").mkdir(parents=True)
    (root / "mafand" / "en-zul" / "train.json").write_text(_json.dumps({"translation": {"en": "The vote was close.", "zul": "Ukhetho belusondele."}}) + "\n", encoding="utf-8")
    (root / "ntrex_african" / "ntrex_african").mkdir(parents=True)
    (root / "ntrex_african" / "ntrex_african" / "zul_Latn.tsv").write_text("sentence_zul_Latn\tsentence_eng_Latn\nIzindaba ezinhle.\tGood news.\n", encoding="utf-8")
    for lang, text in (("eng", "Set an alarm."), ("zul", "Setha i-alamu.")):
        (root / "InjongoIntent" / lang).mkdir(parents=True)
        (root / "InjongoIntent" / lang / "train.jsonl").write_text(_json.dumps({"example_id": "train-1", "text": text}) + "\n", encoding="utf-8")
    (root / "african-ultrachat" / "train").mkdir(parents=True)
    (root / "african-ultrachat" / "train" / "Zulu.jsonl").write_text(_json.dumps({"prompt": "machine", "completion": "machine"}) + "\n", encoding="utf-8")
    pairs = ht.load_masakhane_pairs(root)
    by_source = {p.source: (p.en, p.zu) for p in pairs}
    assert by_source == {
        "AfriDocMT (health and IT documents)": ("Patients should rest.", "Iziguli kufanele ziphumule."),
        "AfriScience-MT (science)": ("We sampled soil.", "Sathatha isampula yenhlabathi."),
        "AfriScience-MT glossary (terms)": ("flora", "izitshalo"),
        "MAFAND-MT (news)": ("The vote was close.", "Ukhetho belusondele."),
        "NTREX (news)": ("Good news.", "Izindaba ezinhle."),
        "InjongoIntent (assistant utterances)": ("Set an alarm.", "Setha i-alamu."),
    }
    assert ht.load_masakhane_pairs(tmp_path / "missing") == []


def test_dsfsi_pairs_keep_good_alignments_and_split_term_alternatives(tmp_path):
    import json as _json

    import pytest

    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "dsfsi"
    nl = chr(10)
    (root / "vukuzenzele-sentence-aligned" / "eng-zul").mkdir(parents=True)
    pq.write_table(pa.table({"eng": ["Crime must stop." + nl, "Misaligned line." + nl], "zul": ["Ubugebengu kumele buphele." + nl, "Okunye ngokuphelele." + nl], "score": [0.84, "0.31"]}),
                   root / "vukuzenzele-sentence-aligned" / "eng-zul" / "train-00000.parquet")
    (root / "za-mafoko-dsac").mkdir(parents=True)
    lines = [_json.dumps({"eng": "account", "zul": "i-akhawunti / ikhawunti"}), _json.dumps({"eng": "empty", "zul": ""})]
    (root / "za-mafoko-dsac" / "dsac_financial.jsonl").write_text(nl.join(lines) + nl, encoding="utf-8")
    pairs = ht.load_dsfsi_pairs(root)
    assert [(p.en, p.zu) for p in pairs if p.source.startswith("Vukuzenzele")] == [("Crime must stop.", "Ubugebengu kumele buphele.")]
    assert sorted((p.en, p.zu) for p in pairs if p.source.startswith("Term list")) == [("account", "i-akhawunti"), ("account", "ikhawunti")]
    assert ht.load_dsfsi_pairs(tmp_path / "missing") == []
