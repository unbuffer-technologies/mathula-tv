from mathula_tv.episode_scoring import describe_missing_literal


def test_a_missing_repeated_acronym_is_described_with_its_counts_and_where_each_occurrence_is():
    from mathula_tv.native_dub import _missing_required_literals

    english = ("But coincidentally, on the day that the former head of IDAC was due to appear, he withdrew. So the IDAC trial was moved, and what IDAC wanted was a statement. "
               "Then IDAC said the Commission should hear it, and IDAC agreed.")
    zulu = "Kodwa ngosuku owayeyinhloko ye-IDAC ayezovela ngalo, wahoxa. Ukuthi i-IDAC ifuna isitatimende."
    missing = _missing_required_literals(source_text=english, candidate_text=zulu)
    assert [m["literal"] for m in missing] == ["IDAC"]
    line = describe_missing_literal(missing[0], zulu)
    assert line.startswith("IDAC x5: the English says it 5 times, your Zulu has it 2 times. In the English: (1) ")
    assert "(5) " in line and "former head of IDAC" in line and "Then IDAC said" in line               # each occurrence with the clause round it, in order


def test_a_single_missing_figure_and_a_long_run_of_occurrences_are_worded_and_capped():
    single = {"literal": "204", "match_literal": "204", "required_count": 1, "source_occurrences": [{"occurrence": 1, "source_context": "…the Section 204 statement…"}]}
    assert describe_missing_literal(single, "isitatimende") == "204 x1: the English says it 1 time, your Zulu has it 0 times. In the English: (1) …the Section 204 statement…"
    many = {"literal": "IDAC", "match_literal": "IDAC", "required_count": 9, "source_occurrences": [{"occurrence": n, "source_context": f"c{n}"} for n in range(1, 10)]}
    assert describe_missing_literal(many, "IDAC")[-12:] == "(and 3 more)" and "(6) c6" in describe_missing_literal(many, "IDAC")


def test_a_flagged_unit_in_a_submit_result_carries_the_evidence_of_each_failed_gate():
    from mathula_tv.dub_workbench import GATE_EVIDENCE_CHARS, Workbench

    group = {"group_id": "u08", "flags": [], "gates": {
        "required_literals": {"failed": True, "evidence": ["IDAC x5: the English says it 5 times, your Zulu has it 3 times. In the English: (1) …" + "x" * 900]},
        "fabricated_entity": {"failed": True, "evidence": ["uMpofu"]}, "untranslated": {"failed": False}}}
    entry = Workbench._flagged_entry(group)
    assert entry["gates"] == ["required_literals", "fabricated_entity"]
    assert entry["gate_evidence"]["fabricated_entity"] == ["uMpofu"]
    assert entry["gate_evidence"]["required_literals"][0].startswith("IDAC x5: the English says it 5 times") and len(entry["gate_evidence"]["required_literals"][0]) == GATE_EVIDENCE_CHARS
    assert "gate_evidence" not in Workbench._flagged_entry({"group_id": "u1", "flags": ["dead_air"], "shortfall_ms": 2500, "tail_pad_ms": 100, "gates": {"x": {"failed": False}}})
    assert Workbench._flagged_entry({"group_id": "u1", "flags": ["dead_air"], "shortfall_ms": 2500, "tail_pad_ms": 100, "gates": {}})["ends_early_ms"] == 2400
