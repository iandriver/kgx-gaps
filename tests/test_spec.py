"""Conformance tests for the reference implementation, keyed to SPEC.md.

Each test names the rule or check it defends. The negative controls matter more than the positives:
a suite that only demonstrates the happy path cannot tell you the rules are enforced.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kgx_gaps import Exporter, Mapping, Rule, conformance, export
from kgx_gaps.mapping import Minted


# --------------------------------------------------------------------------- fixtures

def mapping() -> Mapping:
    m = Mapping(prefix="ex", knowledge_source="infores:example")
    m.rules["SLOPE"] = Rule("biolink:correlated_with", "biolink:GeneToDiseaseAssociation",
                            subject_aspect_qualifier="expression", directional=True)
    m.rules["COLOC"] = Rule("biolink:gene_associated_with_condition")
    m.rules["TRACT"] = Rule(
        m.mint("tractable_for",
               biolink_considered="biolink:target_for",
               rationale="`target_for` asserts the gene IS a therapeutic target for the disease; a "
                         "tractability assessment says only that it could become one."),
        knowledge_level="knowledge_assertion")
    m.gap_types.add("MISSING")
    return m


def evidence() -> pd.DataFrame:
    base = dict(context="cortex", reason="", gap_type="", n_required=None, proposal="",
                kill_condition="")
    return pd.DataFrame([
        # detected -> an association
        {**base, "type": "SLOPE", "subject": "TREM2", "object": "MONDO:0004975",
         "effect": -0.93, "se": 0.079, "floor": 0.22, "detected": None},
        # under its floor -> a gap, NOT a negated assertion (Rule 1)
        {**base, "type": "SLOPE", "subject": "SST", "object": "MONDO:0004975",
         "effect": 0.01, "se": 0.079, "floor": 0.22, "detected": None},
        {**base, "type": "COLOC", "subject": "RABEP1", "object": "MONDO:0004975",
         "effect": 0.868, "se": None, "floor": 0.50, "detected": None},
        # no effect size at all -> not a measurement, Rule 1 must not divert it
        {**base, "type": "TRACT", "subject": "CD33", "object": "MONDO:0004975",
         "effect": None, "se": None, "floor": None, "detected": None},
        # already a gap in the source vocabulary
        {**base, "type": "MISSING", "subject": "SELENBP1", "object": "no_instrument",
         "effect": None, "se": None, "floor": 0.22, "detected": None,
         "gap_type": "no_instrument", "reason": "no usable MR instrument"},
    ])


def run():
    m, ev = mapping(), evidence()
    ex = export(ev, m)
    return m, ev, ex


# --------------------------------------------------------------------------- Rule 1

def test_rule1_under_floor_becomes_a_gap_not_an_assertion():
    m, ev, ex = run()
    _, edges, gaps = ex.frames()
    assert "ex:SST" not in set(edges["subject"]), "an under-floor row was exported as an assertion"
    withheld = gaps[gaps["ex:withheld_from"] != ""]
    assert "ex:SST" in set(withheld["subject"])
    row = withheld[withheld["subject"] == "ex:SST"].iloc[0]
    assert row["ex:withheld_from"] == "biolink:correlated_with"
    assert float(row["ex:detection_floor"]) == 0.22


def test_rule1_does_not_divert_a_row_that_was_never_a_measurement():
    """SPEC.md §2, Rule 1, second paragraph — and C10."""
    m, ev, ex = run()
    _, edges, _ = ex.frames()
    tract = edges[edges["ex:source_edge_type"] == "TRACT"]
    assert len(tract) == 1, "a categorical row with no effect size was wrongly sent to the gap file"
    assert tract.iloc[0]["ex:detected"] == "not_applicable"
    assert tract.iloc[0]["ex:effect_size"] == ""


def test_rule1_never_emits_negated():
    m, ev, ex = run()
    _, edges, gaps = ex.frames()
    assert not [c for c in list(edges.columns) + list(gaps.columns) if c.endswith("negated")]


# --------------------------------------------------------------------------- Rule 2

def test_rule2_minted_predicate_requires_a_rationale():
    with pytest.raises(ValueError, match="no rationale"):
        Minted("ex:foo", "biolink:target_for", "   ")
    with pytest.raises(ValueError, match="BioLink term considered"):
        Minted("ex:foo", "", "some reason")


def test_rule2_gap_predicate_is_registered_automatically():
    m = mapping()
    assert m.gap_predicate in m.minted
    assert "negated" in m.minted[m.gap_predicate].biolink_considered


def test_rule2_unregistered_predicate_is_detected():
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    edges.loc[0, "predicate"] = "ex:sneaked_in"
    rep = conformance.check(nodes, edges, gaps, m, n_input_rows=len(ev))
    c3 = next(c for c in rep.checks if c.id == "C3")
    assert not c3.passed and "ex:sneaked_in" in c3.detail


# --------------------------------------------------------------------------- Rule 3

def test_rule3_ungrounded_ids_are_flagged_not_dropped():
    m, ev, ex = run()
    nodes, _, _ = ex.frames()
    assert "ex:TREM2" in set(nodes["id"])
    assert (nodes["id_grounded"] == "false").all()
    assert nodes["id"].str.match(r"^[A-Za-z0-9._]+:\S+$").all()


def test_rule3_resolver_may_not_claim_an_ontology_prefix_while_ungrounded():
    m = mapping()
    bad = lambda label: (f"CL:{label}", "biolink:Cell", False)      # noqa: E731
    with pytest.raises(ValueError, match="Rule 3"):
        export(evidence().head(1), m, resolver=bad)


def test_rule3_a_real_resolver_grounds_and_is_not_flagged():
    m = mapping()
    def res(label):
        table = {"TREM2": "ENSEMBL:ENSG00000095970", "MONDO:0004975": "MONDO:0004975"}
        hit = table.get(str(label))
        return (hit, "biolink:Gene", True) if hit else (f"ex:{label}", "biolink:NamedThing", False)
    ex = export(evidence(), m, resolver=res)
    nodes, _, _ = ex.frames()
    assert nodes.loc[nodes["id"] == "ENSEMBL:ENSG00000095970", "id_grounded"].iloc[0] == "true"
    assert "TREM2" not in ex.ungrounded


# --------------------------------------------------------------------------- conservation & io

def test_c8_every_input_row_is_accounted_for():
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    assert len(edges) + len(gaps) == len(ev)


def test_full_conformance_suite_passes_on_the_reference_output():
    m, ev, ex = run()
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
    assert rep.passed, str(rep)
    assert rep.n_skipped == 0


def test_c9_no_nan_reaches_the_tsv(tmp_path):
    m, ev, ex = run()
    files = ex.write(tmp_path / "kgx")
    for p in files.values():
        text = p.read_text()
        fields = {f for line in text.splitlines() for f in line.split("\t")}
        assert not (fields & {"nan", "None", "NaN", "<NA>"}), f"{p.name} serialised a null sentinel"


def test_check_dir_round_trips(tmp_path):
    m, ev, ex = run()
    ex.write(tmp_path / "kgx")
    rep = conformance.check_dir(tmp_path / "kgx", m, n_input_rows=len(ev))
    assert rep.passed, str(rep)


# --------------------------------------------------------------------------- the suite must bite

def test_c1_catches_a_foreign_graph_that_asserted_under_its_floor():
    """The check has to fail on bad input or it is decoration."""
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    edges.loc[0, "ex:effect_size"] = 0.01          # now below the 0.22 floor
    rep = conformance.check(nodes, edges, gaps, m, n_input_rows=len(ev))
    c1 = next(c for c in rep.checks if c.id == "C1")
    assert not c1.passed and c1.offenders


def test_a_skip_is_not_a_pass():
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    rep = conformance.check(nodes, edges, gaps, mapping=None, n_input_rows=None)
    assert rep.n_skipped >= 2                       # C3 needs a Mapping, C8 an input count
    assert not rep.passed, "a report with skipped checks must not report success"
    assert "SKIPPED (not a pass)" in str(rep)


def test_c7_catches_a_dangling_reference():
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    edges.loc[0, "object"] = "MONDO:9999999"
    rep = conformance.check(nodes, edges, gaps, m, n_input_rows=len(ev))
    assert not next(c for c in rep.checks if c.id == "C7").passed


def test_c2_catches_negated():
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    edges["negated"] = ["true"] + [""] * (len(edges) - 1)
    rep = conformance.check(nodes, edges, gaps, m, n_input_rows=len(ev))
    assert not next(c for c in rep.checks if c.id == "C2").passed


def test_missing_rule_is_an_error_not_a_silent_drop():
    m = mapping()
    ev = evidence().assign(type="UNDECLARED")
    with pytest.raises(KeyError, match="no Rule for source edge type"):
        export(ev, m)


def test_c8_excludes_structural_edges_but_still_catches_a_dropped_row():
    """The amendment must not turn C8 into a check that cannot fail.

    Found by running the suite against a real graph carrying one legitimate scaffold edge
    (disease --has_attribute--> severity axis), which C8 originally counted as an unexplained extra.
    """
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    structural = edges.iloc[[0]].copy()
    structural["id"] = "ex:e999999"
    structural["predicate"] = "biolink:has_attribute"
    structural["ex:source_edge_type"] = ""          # marks it structural
    edges2 = pd.concat([edges, structural], ignore_index=True)

    rep = conformance.check(nodes, edges2, gaps, m, n_input_rows=len(ev))
    c8 = next(c for c in rep.checks if c.id == "C8")
    assert c8.passed, c8.detail
    assert "1 structural, excluded" in c8.detail

    # and it must still bite when an evidence row really did vanish
    rep2 = conformance.check(nodes, edges2.iloc[1:], gaps, m, n_input_rows=len(ev))
    assert not next(c for c in rep2.checks if c.id == "C8").passed


# --------------------------------------------------------------------------- per-row rules (0.2.0)

def test_predicate_may_depend_on_the_row():
    m = Mapping(prefix="ex", knowledge_source="infores:example")
    weaker = m.mint("tractable_for", biolink_considered="biolink:target_for",
                    rationale="target_for asserts it IS a target; tractability says it could be.")
    m.rules["DRUG"] = Rule(lambda r: "biolink:target_for" if r.get("object") else weaker,
                           knowledge_level="knowledge_assertion")
    m.declare_predicates("DRUG", "biolink:target_for", weaker)
    assert not m.validate()
    ev = pd.DataFrame([
        {"type": "DRUG", "subject": "CD33", "object": "PHASE_3"},
        {"type": "DRUG", "subject": "SIGLEC11", "object": ""},
    ])
    _, edges, _ = export(ev, m).frames()
    assert set(edges["predicate"]) == {"biolink:target_for", weaker}


def test_callable_predicate_must_declare_what_it_can_emit():
    """Otherwise a minted predicate reaches the output without ever passing Rule 2."""
    m = Mapping(prefix="ex", knowledge_source="infores:example")
    m.rules["DRUG"] = Rule(lambda r: "ex:never_registered")
    problems = m.validate()
    assert any("declare_predicates" in p for p in problems), problems


def test_precondition_withholds_with_its_own_reason_and_still_names_the_predicate():
    m = Mapping(prefix="ex", knowledge_source="infores:example")
    m.rules["MR"] = Rule("biolink:affects",
                         precondition=lambda r: "weak instrument" if r.get("weak") else "")
    ev = pd.DataFrame([
        {"type": "MR", "subject": "MERTK", "object": "MONDO:1", "effect": 9.0, "floor": 0.1,
         "weak": True},
        {"type": "MR", "subject": "PLCG2", "object": "MONDO:1", "effect": 0.4, "floor": 0.1,
         "weak": False},
    ])
    ex = export(ev, m)
    _, edges, gaps = ex.frames()
    assert list(edges["subject"]) == ["ex:PLCG2"]
    held = gaps[gaps["ex:withheld_from"] != ""].iloc[0]
    # the effect was huge; the reason must be the instrument, not the floor
    assert held["ex:gap_reason"] == "weak instrument"
    assert held["ex:withheld_from"] == "biolink:affects"
    assert "floor" not in held["ex:gap_reason"]


def test_precondition_rows_still_satisfy_conformance():
    m = Mapping(prefix="ex", knowledge_source="infores:example")
    m.rules["MR"] = Rule("biolink:affects", precondition=lambda r: "weak instrument")
    ev = pd.DataFrame([{"type": "MR", "subject": "A", "object": "B", "effect": 1.0, "floor": 0.1}])
    ex = export(ev, m)
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
    assert rep.passed, str(rep)


def test_an_effect_with_no_floor_is_not_assertable():
    """Found by exporting a new disease: the implementation emitted output its own C10 rejected."""
    m = Mapping(prefix="ex", knowledge_source="infores:ex")
    m.rules["MR"] = Rule("biolink:affects")
    ev = pd.DataFrame([{"type": "MR", "subject": "A", "object": "B", "effect": 0.0001,
                        "floor": None}])
    ex = export(ev, m)
    _, edges, gaps = ex.frames()
    assert len(edges) == 0 and len(gaps) == 1
    g = gaps.iloc[0]
    assert g["ex:gap_type"] == "no_detection_floor"
    assert g["ex:withheld_from"] == "biolink:affects"
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
    assert rep.passed, str(rep)


def test_a_row_with_neither_effect_nor_floor_is_still_assertable():
    """The amendment must not swallow genuine categorical claims (Rule 1, second paragraph)."""
    m = Mapping(prefix="ex", knowledge_source="infores:ex")
    m.rules["TRACT"] = Rule("biolink:target_for", knowledge_level="knowledge_assertion")
    ev = pd.DataFrame([{"type": "TRACT", "subject": "CD33", "object": "MONDO:1"}])
    _, edges, gaps = export(ev, m).frames()
    assert len(edges) == 1 and len(gaps) == 0
    assert edges.iloc[0]["ex:detected"] == "not_applicable"


def test_a_withheld_categorical_row_is_marked_not_a_measurement():
    """C4 demands a floor only from rows that reported a number. A categorical claim refused by a
    precondition has none, and the gap file has to say which kind of refusal it was."""
    m = Mapping(prefix="ex", knowledge_source="infores:ex")
    m.rules["COLOC"] = Rule("biolink:gene_associated_with_condition",
                            precondition=lambda r: "PP4 below threshold" if r.get("weak") else "")
    ev = pd.DataFrame([
        {"type": "COLOC", "subject": "A", "object": "B", "weak": True},                   # no effect
        {"type": "COLOC", "subject": "C", "object": "B", "effect": 0.9, "floor": 0.1,
         "weak": True},                                                                   # measured
    ])
    _, _, gaps = export(ev, m).frames()
    assert list(gaps["ex:was_measurement"]) == ["false", "true"]
    rep = conformance.check(*export(ev, m).frames(), m, n_input_rows=len(ev))
    assert rep.passed, str(rep)


# ---------------------------------------------------------------- gap identity (0.4.0)

def _gap_exporter(**kw):
    m = Mapping(prefix="demo", knowledge_source="infores:demo", **kw)
    m.gap_types.add("GAP")
    return Exporter(mapping=m)


def _gap_row(subject, **over):
    row = {"type": "GAP", "subject": subject, "object": "why", "gap_type": "no_instrument",
           "reason": "no usable instrument", "context": "microglia"}
    row.update(over)
    return row


def test_gap_id_survives_an_earlier_gap_being_added():
    """The property the counter did not have: an id names an absence, not a row position."""
    a = _gap_exporter()
    a.add(_gap_row("G1"))
    b = _gap_exporter()
    b.add(_gap_row("G0"))          # a gap that did not exist in the first build, emitted first
    b.add(_gap_row("G1"))
    assert a.gaps[0]["id"] == b.gaps[1]["id"]


def test_gap_id_ignores_what_is_measured_about_the_gap():
    """A floor, a count, a reason and a proposal move while the gap stays the same gap."""
    a = _gap_exporter()
    a.add(_gap_row("G1", reason="no usable instrument", floor=0.2, n_required=120))
    b = _gap_exporter()
    b.add(_gap_row("G1", reason="rewritten reason", floor=0.9, n_required=480))
    assert a.gaps[0]["id"] == b.gaps[0]["id"]


def test_gap_id_changes_with_what_the_gap_is_about():
    e = _gap_exporter()
    e.add(_gap_row("G1"))
    e.add(_gap_row("G1", context="astrocyte"))
    e.add(_gap_row("G1", gap_type="no_cohort"))
    assert len({g["id"] for g in e.gaps}) == 3


def test_the_same_gap_said_twice_is_written_once():
    """Two evidence rows can state one absence; the second is dropped and counted, not duplicated."""
    e = _gap_exporter()
    e.add(_gap_row("G1"))
    e.add(_gap_row("G1"))
    assert len(e.gaps) == 1 and e.merged_gaps == 1


def test_two_gaps_with_one_identity_that_differ_are_refused():
    """Something distinguishes them that the id cannot see, so the export stops instead of guessing."""
    e = _gap_exporter()
    e.add(_gap_row("G1", floor=0.2))
    with pytest.raises(ValueError, match="share the identity"):
        e.add(_gap_row("G1", floor=0.9))


def test_a_declared_identity_column_tells_two_gaps_apart():
    """A producer whose gaps differ only by one of its own columns declares it, and both survive."""
    m = Mapping(prefix="demo", knowledge_source="infores:demo",
                gap_identity_columns=["demo:cell_type_context"])
    m.gap_types.add("GAP")
    e = Exporter(mapping=m, extra_gap_columns=["demo:cell_type_context"],
                 extras=lambda row, kind: {"demo:cell_type_context": row.get("ct", "")})
    e.add(_gap_row("G1", ct="microglia"))
    e.add(_gap_row("G1", ct="astrocyte"))
    assert len({g["id"] for g in e.gaps}) == 2


def test_gap_ids_are_curies_and_unique_in_the_conformance_report():
    e = _gap_exporter()
    e.add(_gap_row("G1"))
    e.add(_gap_row("G2"))
    nodes, edges, gaps = e.frames()
    c6 = [c for c in conformance.check(nodes, edges, gaps, e.mapping).checks if c.id == "C6"][0]
    assert c6.passed, c6.detail


# ---------------------------------------------------------------- edge identity (0.5.0)

def _edge_mapping(**kw) -> Mapping:
    m = Mapping(prefix="demo", knowledge_source="infores:demo", **kw)
    m.rules["SLOPE"] = Rule("biolink:correlated_with", subject_aspect_qualifier="expression",
                            directional=True)
    m.rules["COLOC"] = Rule("biolink:gene_associated_with_condition")
    return m


def _slope(subject, **over):
    row = {"type": "SLOPE", "subject": subject, "object": "MONDO:1", "context": "cortex",
           "effect": -0.9, "se": 0.08, "floor": 0.2}
    row.update(over)
    return row


def test_edge_id_survives_an_earlier_edge_being_added():
    """The property the counter did not have: an id names a claim, not a row position."""
    a = Exporter(mapping=_edge_mapping())
    a.add(_slope("G1"))
    b = Exporter(mapping=_edge_mapping())
    b.add(_slope("G0"))            # an edge the first build did not have, emitted first
    b.add(_slope("G1"))
    assert a.edges[0]["id"] == b.edges[1]["id"]
    assert not a.edges[0]["id"].endswith("e000000")


def test_edge_id_ignores_what_is_measured_about_the_claim():
    """An effect size, its standard error and its floor move while the claim stays the same claim."""
    a = Exporter(mapping=_edge_mapping())
    a.add(_slope("G1", effect=-0.9, se=0.08, floor=0.2))
    b = Exporter(mapping=_edge_mapping())
    b.add(_slope("G1", effect=-0.4, se=0.03, floor=0.1))
    assert a.edges[0]["id"] == b.edges[0]["id"]


def test_edge_id_changes_with_what_the_edge_claims():
    e = Exporter(mapping=_edge_mapping())
    e.add(_slope("G1"))
    e.add(_slope("G1", object="MONDO:2"))
    e.add(_slope("G1", context="cerebellum"))
    e.add(_slope("G1", type="COLOC", effect=None, floor=None))
    # the direction is part of the claim: `decreased` and `increased` may not share an id, or a
    # citation of one comes to point at the other when the estimate changes sign
    e.add(_slope("G1", effect=0.9))
    assert len({r["id"] for r in e.edges}) == 5


def test_two_exports_of_one_producer_share_no_edge_id():
    """What prompted the rule: every export began at e000000, so three of them could not be loaded
    into one graph. Two exports that assert different edges must spend different ids."""
    ids = []
    for disease in ("MONDO:1", "MONDO:2"):
        e = Exporter(mapping=_edge_mapping())
        for gene in ("G1", "G2", "G3"):
            e.add(_slope(gene, object=disease))
        ids += [r["id"] for r in e.edges]
    assert len(ids) == 6 and len(set(ids)) == 6


def test_the_same_edge_said_twice_is_written_once_and_c8_counts_it():
    m = _edge_mapping()
    ev = pd.DataFrame([_slope("G1"), _slope("G1"), _slope("G2")])
    ex = export(ev, m)
    assert len(ex.edges) == 2 and ex.merged_edges == 1 and ex.merged == 1
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev), n_merged=ex.merged)
    assert rep.passed, str(rep)
    assert "1 repeating a row already written" in next(c for c in rep.checks if c.id == "C8").detail
    # and C8 still bites: without the count the third row is unaccounted for
    silent = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
    assert not next(c for c in silent.checks if c.id == "C8").passed


def test_two_edges_with_one_identity_that_differ_are_refused():
    """Two measurements the id cannot tell apart: the export stops instead of keeping the first."""
    e = Exporter(mapping=_edge_mapping())
    e.add(_slope("G1", effect=-0.9))
    with pytest.raises(ValueError, match="share the identity"):
        e.add(_slope("G1", effect=-0.5))


def test_a_declared_identity_column_tells_two_edges_apart():
    """A producer whose edges differ by one of its own columns declares it, and both survive."""
    m = _edge_mapping(edge_identity_columns=["demo:study"])
    e = Exporter(mapping=m, extra_edge_columns=["demo:study", "demo:pp4"],
                 extras=lambda row, kind: ({"demo:study": row.get("study", ""),
                                            "demo:pp4": row.get("pp4", "")} if kind == "edge" else {}))
    coloc = {"type": "COLOC", "subject": "G1", "object": "MONDO:1", "context": "blood"}
    e.add({**coloc, "study": "A", "pp4": 0.91})
    e.add({**coloc, "study": "B", "pp4": 0.83})
    assert len({r["id"] for r in e.edges}) == 2
    # the undeclared column is measured about the claim, so it does not separate two rows
    with pytest.raises(ValueError, match=r"differ on \['demo:pp4'\]"):
        e.add({**coloc, "study": "A", "pp4": 0.55})


def test_an_identity_column_the_file_does_not_carry_is_refused():
    """A misspelt identity column would hash the empty string for every row and identify nothing."""
    with pytest.raises(KeyError, match="edge_identity_columns"):
        Exporter(mapping=_edge_mapping(edge_identity_columns=["demo:studdy"]),
                 extra_edge_columns=["demo:study"])
    with pytest.raises(KeyError, match="gap_identity_columns"):
        Exporter(mapping=_edge_mapping(gap_identity_columns=["demo:cell_type_context"]))


def test_c6_catches_an_edge_id_used_twice_or_shared_with_a_gap():
    m, ev, ex = run()
    nodes, edges, gaps = ex.frames()
    assert next(c for c in conformance.check(nodes, edges, gaps, m).checks if c.id == "C6").passed

    twice = edges.copy()
    twice.loc[1, "id"] = twice.loc[0, "id"]
    c6 = next(c for c in conformance.check(nodes, twice, gaps, m).checks if c.id == "C6")
    assert not c6.passed and twice.loc[0, "id"] in c6.detail

    shared = edges.copy()
    shared.loc[0, "id"] = gaps.loc[0, "id"]
    c6 = next(c for c in conformance.check(nodes, shared, gaps, m).checks if c.id == "C6")
    assert not c6.passed and "both an edge and a gap" in c6.detail


def test_a_structural_edge_takes_a_content_id_and_stays_out_of_c8():
    m, ev = _edge_mapping(), pd.DataFrame([_slope("G1")])
    ids = []
    for extra_first in (False, True):
        ex = export(ev, m)
        ex.node("demo:AXIS:severity", "biolink:ClinicalAttribute", "severity axis", grounded=False)
        if extra_first:
            ex.add(_slope("G2"))
        ids.append(ex.structural("demo:MONDO:1", "biolink:has_attribute", "demo:AXIS:severity"))
    assert ids[0] == ids[1], "a structural edge's id moved when an evidence row was added before it"
    row = ex.edges[-1]
    assert row["id"] == ids[1] and row["demo:source_edge_type"] == ""
    rep = conformance.check(*ex.frames(), m, n_input_rows=2)
    assert rep.passed, str(rep)
    assert "1 structural, excluded" in next(c for c in rep.checks if c.id == "C8").detail
    # said twice it is still one edge, and it is not an input row, so nothing is added to `merged`
    ex.structural("demo:MONDO:1", "biolink:has_attribute", "demo:AXIS:severity")
    assert sum(r["predicate"] == "biolink:has_attribute" for r in ex.edges) == 1
    assert ex.merged == 0


def test_a_structural_edge_cannot_pose_as_evidence_or_skip_the_rules():
    ex = export(pd.DataFrame([_slope("G1")]), _edge_mapping())
    ex.node("demo:AXIS:severity", "biolink:ClinicalAttribute", grounded=False)
    with pytest.raises(KeyError, match="source_edge_type"):
        ex.structural("demo:MONDO:1", "biolink:has_attribute", "demo:AXIS:severity",
                      fields={"demo:source_edge_type": "SLOPE"})
    with pytest.raises(KeyError, match="not in nodes"):
        ex.structural("demo:MONDO:1", "biolink:has_attribute", "demo:AXIS:never_added")
    with pytest.raises(ValueError, match="unregistered"):
        ex.structural("demo:MONDO:1", "demo:scored_on", "demo:AXIS:severity")


def test_the_id_recipe_is_pinned():
    """An id is a promise to whoever stored it. These two literals fail if the hashed fields, their
    order or the separator change -- which re-addresses every edge and gap already recorded."""
    e = Exporter(mapping=_edge_mapping())
    e.add(_slope("G1"))
    assert e.edges[0]["id"] == "demo:e9dc117c62e4b"
    g = _gap_exporter()
    g.add(_gap_row("G1"))
    assert g.gaps[0]["id"] == "demo:g8db5e576ea19"


# ---------------------------------------------------------------- one id, one name (0.5.0)

def test_a_resolver_may_name_the_node_so_two_labels_give_one_name():
    """A gene reached by its symbol in one row and its accession in another is one node; without a
    name from the resolver it is called whichever label came first."""
    table = {"TREM2": "ENSEMBL:ENSG00000095970", "ENSG00000095970": "ENSEMBL:ENSG00000095970"}

    def res(label):
        hit = table.get(str(label))
        return (hit, "biolink:Gene", True, "TREM2") if hit else (str(label), "biolink:Disease", True)

    names = []
    for first, second in (("TREM2", "ENSG00000095970"), ("ENSG00000095970", "TREM2")):
        ex = Exporter(mapping=_edge_mapping(), resolver=res)
        ex.add(_slope(first))
        ex.add(_slope(second, context="cerebellum"))
        names.append(ex.nodes["ENSEMBL:ENSG00000095970"]["name"])
    assert names == ["TREM2", "TREM2"]
    assert ex.nodes["MONDO:1"]["name"] == "MONDO:1"      # a three-tuple still names by label


def test_a_gap_node_is_named_by_the_mapping_not_by_what_the_export_happened_to_contain():
    m = Mapping(prefix="demo", knowledge_source="infores:demo",
                gap_names={"no_instrument": "no usable instrument"})
    m.gap_types.add("GAP")
    e = Exporter(mapping=m)
    e.add(_gap_row("G1"))
    e.add(_gap_row("G1", gap_type="no_cohort"))
    assert e.nodes["demo:GAP:no_instrument"]["name"] == "no usable instrument"
    assert e.nodes["demo:GAP:no_cohort"]["name"] == "no cohort"
