"""Conformance tests for the reference implementation, keyed to SPEC.md.

Each test names the rule or check it defends. The negative controls matter more than the positives:
a suite that only demonstrates the happy path cannot tell you the rules are enforced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kgx_gaps import Exporter, Mapping, Rule, blank, conformance, export
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


# ---------------------------------------------------------------- an attribute keeps its type (0.5.1)

def _attr_exporter(**kw) -> Exporter:
    """Edges carrying three producer columns, filled from the row's `year`, `n` and `acc`."""
    cols = ["demo:first_year", "demo:n_studies", "demo:accession"]
    return Exporter(mapping=_edge_mapping(**kw), extra_edge_columns=cols,
                    extras=lambda row, kind: ({"demo:first_year": row.get("year"),
                                               "demo:n_studies": row.get("n"),
                                               "demo:accession": row.get("acc")}
                                              if kind == "edge" else {}))


def _written(ex: Exporter, tmp_path, name: str = "edges") -> pd.DataFrame:
    """The file as text, cell for cell: what a consumer reads, before pandas guesses a type."""
    return pd.read_csv(ex.write(tmp_path / "kgx")[name], sep="\t", dtype=str, keep_default_na=False)


@pytest.mark.parametrize("given, expected", [
    (2018, 2018), (np.int64(2018), 2018), ("2018", "2018"), (1, 1), (0, 0), ("0012", "0012"),
    ("1e5", "1e5"), (True, True), (False, False), (np.True_, True), (0.22, 0.22), (2018.0, 2018.0),
    (np.float32(0.5), 0.5), ("GCST0001", "GCST0001"),
    (None, ""), (float("nan"), ""), (pd.NA, ""), (pd.NaT, ""), ("nan", ""), ("NaN", ""),
    ("None", ""), ("<NA>", ""), (float("inf"), ""), (float("-inf"), ""), (np.inf, ""),
])
def test_blank_returns_the_value_it_was_given_or_the_empty_string(given, expected):
    got = blank(given)
    assert got == expected and type(got) is type(expected), f"{given!r} -> {got!r}"


def test_a_whole_number_attribute_is_written_whole(tmp_path):
    """`blank()` tried `float(v)` first, so a year reached the file as 2018.0 and a count as 1.0."""
    e = _attr_exporter()
    e.add(_slope("G1", year=2018, n=1))
    e.add(_slope("G2", year=np.int64(2018), n=np.int64(1)))
    e.add(_slope("G3", year="2018", n="1"))
    out = _written(e, tmp_path)
    assert list(out["demo:first_year"]) == ["2018", "2018", "2018"]
    assert list(out["demo:n_studies"]) == ["1", "1", "1"]
    # and when another row has none, which makes the column one of mixed types
    e.add(_slope("G4"))
    assert list(_written(e, tmp_path)["demo:first_year"]) == ["2018", "2018", "2018", ""]


def test_an_identifier_made_of_digits_keeps_its_leading_zeros(tmp_path):
    """`"0012"` was written `12.0`: a different identifier, and no error to say so."""
    e = _attr_exporter(edge_identity_columns=["demo:accession"])
    e.add(_slope("G1", acc="0012"))
    # as floats these two were one identity; they are two accessions and so two edges
    e.add(_slope("G1", acc="12"))
    # a context is a label too, and one that is the number 0 is not an absent context
    e.add(_slope("G2", context="007"))
    e.add(_slope("G3", context=0))
    out = _written(e, tmp_path)
    assert list(out["demo:accession"]) == ["0012", "12", "", ""]
    assert list(out["demo:context"]) == ["cortex", "cortex", "007", "0"]
    assert out["id"].nunique() == 4


def test_an_absent_attribute_is_still_written_empty(tmp_path):
    """SPEC.md §4 and C9: keeping a value's type must not let a null sentinel through as text."""
    absent = [None, float("nan"), np.nan, pd.NA, pd.NaT, "nan", "NaN", "None", "<NA>",
              float("inf"), float("-inf")]
    e = _attr_exporter()
    for i, v in enumerate(absent):
        e.add(_slope(f"G{i}", year=v, n=v, acc=v, se=v))
    out = _written(e, tmp_path)
    for c in ("demo:first_year", "demo:n_studies", "demo:accession", "demo:standard_error"):
        assert set(out[c]) == {""}, f"{c} serialised {sorted(set(out[c]) - {''})}"
    text = (tmp_path / "kgx" / "edges.tsv").read_text()
    fields = {f for line in text.splitlines() for f in line.split("\t")}
    assert not (fields & {"nan", "None", "NaN", "<NA>", "NaT", "inf", "-inf"})
    rep = conformance.check(*e.frames(), e.mapping, n_input_rows=len(absent))
    assert next(c for c in rep.checks if c.id == "C9").passed and rep.passed, str(rep)


def test_rule1_still_withholds_when_the_numbers_arrive_as_strings(tmp_path):
    """An effect and a floor read from a text file are still a measurement, and Rule 1 still reads
    them as numbers: `"0.01"` under `"0.22"` is withheld, it is not compared as text."""
    m = _edge_mapping()
    ev = pd.DataFrame([_slope("G1", effect="-0.93", se="0.079", floor="0.22"),
                       _slope("G2", effect="0.01", se="0.079", floor="0.22"),
                       # as text "9" sorts above "10"; as numbers it is under its floor
                       _slope("G3", effect="9", se="1", floor="10"),
                       _slope("G4", effect="0.5", se="0.1", floor=None)])
    ex = export(ev, m)
    edges, gaps = _written(ex, tmp_path), _written(ex, tmp_path, "gaps")
    assert list(edges["subject"]) == ["demo:G1"]
    assert edges.iloc[0]["object_direction_qualifier"] == "decreased"
    assert (edges.iloc[0]["demo:effect_size"], edges.iloc[0]["demo:standard_error"],
            edges.iloc[0]["demo:detection_floor"]) == ("-0.93", "0.079", "0.22")
    assert list(gaps["subject"]) == ["demo:G2", "demo:G3", "demo:G4"]
    assert list(gaps["demo:gap_type"]) == ["under_detection_floor"] * 2 + ["no_detection_floor"]
    assert set(gaps["demo:withheld_from"]) == {"biolink:correlated_with"}
    assert list(gaps["demo:detection_floor"]) == ["0.22", "10.0", ""]
    assert ex.withheld == 3
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
    assert rep.passed, str(rep)


def test_a_measurement_is_one_number_however_it_was_typed():
    """Effect, standard error and floor are real-valued, so `"-0.9"`, `-0.9` and an int floor of 2
    are the numbers they state. The same row from a text source and a numeric one is one claim said
    twice, not two that differ."""
    e = Exporter(mapping=_edge_mapping())
    e.add(_slope("G1", effect=-0.9, se=0.08, floor=0.2))
    e.add(_slope("G1", effect="-0.9", se="0.08", floor="0.2"))
    assert len(e.edges) == 1 and e.merged_edges == 1
    e.add(_slope("G2", effect=3, floor=2))
    assert (e.edges[1]["demo:effect_size"], e.edges[1]["demo:detection_floor"]) == (3.0, 2.0)
    assert all(type(e.edges[1][c]) is float for c in ("demo:effect_size", "demo:detection_floor"))


# ---------------------------------------------------------------- a verdict not given is computed (0.5.2)

#: Every way a frame says no verdict was given. `blank()` writes each of them as nothing.
NO_VERDICT = [None, float("nan"), np.nan, pd.NA, pd.NaT, "", " ", "nan", "NaN", "None", "<NA>"]


def _routed(**over) -> tuple[str, Exporter]:
    """Where one SLOPE row goes, 'edge' or 'gap', and the exporter it went into."""
    e = Exporter(mapping=_edge_mapping())
    return e.add(_slope("G1", **over)), e


@pytest.mark.parametrize("absent", NO_VERDICT)
def test_rule1_computes_the_verdict_when_detected_is_absent(absent):
    """Only None counted as not given, and `bool(nan)` is True: a measurement under its floor whose
    `detected` cell was blank was exported as an association, `detected` = "true", and the
    package's own C1 then rejected the file."""
    assert str(blank(absent)).strip() == ""
    kind, e = _routed(effect=0.01, floor=0.22, detected=absent)
    assert kind == "gap" and not e.edges, f"detected={absent!r} asserted a row under its floor"
    assert e.gaps[0]["demo:gap_type"] == "under_detection_floor"
    assert e.gaps[0]["demo:withheld_from"] == "biolink:correlated_with"
    rep = conformance.check(*e.frames(), e.mapping, n_input_rows=1)
    assert rep.passed, str(rep)
    # and absent is not False: the same cell on a row over its floor withholds nothing
    kind, e = _routed(effect=-0.93, floor=0.22, detected=absent)
    assert kind == "edge" and not e.gaps, f"detected={absent!r} withheld a row over its floor"
    assert e.edges[0]["demo:detected"] == "true"


def test_a_blank_detected_cell_read_from_a_file_is_not_a_yes(tmp_path):
    """The path that prompted it. A `detected` column with blanks in it comes back from a file as
    NaN beside the bools, as the empty string when the file is read as text, and as 1.0, 0.0 and
    NaN once anything has made the column numeric. The three are one table and route one way."""
    src = tmp_path / "evidence.csv"
    src.write_text("type,subject,object,context,effect,floor,detected\n"
                   "SLOPE,G1,MONDO:1,cortex,-0.93,0.22,\n"          # over its floor, no verdict
                   "SLOPE,G2,MONDO:1,cortex,0.01,0.22,\n"           # under it, no verdict
                   "SLOPE,G3,MONDO:1,cortex,-0.93,0.22,True\n"
                   "SLOPE,G4,MONDO:1,cortex,0.5,0.22,False\n"       # the producer's own bar is higher
                   "SLOPE,G5,MONDO:1,cortex,0.01,0.22,False\n")
    typed = pd.read_csv(src)
    as_text = pd.read_csv(src, dtype=str, keep_default_na=False)
    numeric = typed.assign(detected=typed["detected"].map({True: 1.0, False: 0.0}))
    # the premise: none of the three holds a None for the blank cell
    assert not any(f["detected"].iloc[1] is None for f in (typed, as_text, numeric))
    assert list(as_text["detected"]) == ["", "", "True", "False", "False"]
    assert list(numeric["detected"].dropna()) == [1.0, 0.0, 0.0]
    for ev in (typed, as_text, numeric):
        m = _edge_mapping()
        ex = export(ev, m)
        assert [r["subject"] for r in ex.edges] == ["demo:G1", "demo:G3"]
        assert [r["subject"] for r in ex.gaps] == ["demo:G2", "demo:G4", "demo:G5"]
        assert {r["demo:gap_type"] for r in ex.gaps} == {"under_detection_floor"}
        rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
        assert rep.passed, str(rep)


@pytest.mark.parametrize("no", [False, np.False_, 0, 0.0, "False", "false", "FALSE", " false ",
                                "f", "no", "N", "0", "0.0"])
def test_a_false_verdict_withholds_however_it_is_spelt(no):
    """`bool("False")` is True, so a producer's "not detected", read from text, asserted the row."""
    for effect in (0.01, -0.93):                    # under the 0.22 floor, and over it
        kind, e = _routed(effect=effect, floor=0.22, detected=no)
        assert kind == "gap" and not e.edges, f"detected={no!r} asserted an effect of {effect}"
        assert e.gaps[0]["demo:gap_type"] == "under_detection_floor"


@pytest.mark.parametrize("yes", [True, np.True_, 1, 1.0, "True", "true", "TRUE", " true ",
                                 "t", "yes", "Y", "1", "1.0"])
def test_a_true_verdict_is_still_asserted_however_it_is_spelt(yes):
    """The control: reading a verdict from text must not make every string a no."""
    kind, e = _routed(effect=-0.93, floor=0.22, detected=yes)
    assert kind == "edge" and not e.gaps, f"detected={yes!r} withheld a row over its floor"
    assert e.edges[0]["demo:detected"] == "true"


@pytest.mark.parametrize("unreadable", ["maybe", "0.37", 0.37, 2, "2", -1])
def test_a_detected_that_is_not_a_verdict_is_refused(unreadable):
    """A p-value or a count in this column was truthy, so it asserted the row. It is not a yes,
    and it is not an absent verdict either: the export stops and shows the value."""
    with pytest.raises(ValueError, match="neither a verdict"):
        _routed(effect=0.01, floor=0.22, detected=unreadable)
    with pytest.raises(ValueError, match="neither a verdict"):
        _routed(effect=-0.93, floor=0.22, detected=unreadable)
    # and only a measurement is refused: Rule 1 does not read this column on a categorical row
    e = Exporter(mapping=_edge_mapping())
    coloc = {"type": "COLOC", "subject": "G1", "object": "MONDO:1", "detected": unreadable}
    assert e.add(coloc) == "edge" and e.edges[0]["demo:detected"] == "not_applicable"


@pytest.mark.parametrize("unset", NO_VERDICT + [False, "False", 0])
def test_rule1_does_not_divert_a_categorical_row_whatever_its_detected_says(unset):
    """SPEC.md §2, Rule 1, third paragraph. Computing a verdict that was not given must not become
    withholding the rows that never had one to give: with no effect size there is nothing to
    compute it from, in a frame where that row's effect, floor and `detected` are all NaN."""
    e = Exporter(mapping=_edge_mapping())
    row = {"type": "COLOC", "subject": "G1", "object": "MONDO:1", "context": "blood",
           "effect": float("nan"), "floor": float("nan"), "detected": unset}
    assert e.add(row) == "edge" and not e.gaps
    assert e.edges[0]["demo:detected"] == "not_applicable"
    rep = conformance.check(*e.frames(), e.mapping, n_input_rows=1)
    assert rep.passed, str(rep)


# ---------------------------------------------------------------- the withheld object (0.6.0)

def _pair_mapping(**kw) -> Mapping:
    """A source of (gene, disease) pairs, some of them withheld by a precondition."""
    m = Mapping(prefix="demo", knowledge_source="infores:demo", **kw)
    m.rules["ASSOC"] = Rule("biolink:gene_associated_with_condition",
                            precondition=lambda r: ("one source, and nothing corroborates it"
                                                    if r.get("withheld") else ""))
    m.rules["SLOPE"] = Rule("biolink:correlated_with", subject_aspect_qualifier="expression",
                            directional=True)
    m.gap_types.add("GAP")
    return m


def _curies(label):
    """A resolver for labels that are CURIEs already; anything else is flagged, under the prefix."""
    label = str(label)
    if label.startswith("HGNC:"):
        return label, "biolink:Gene", True
    if label.startswith("MONDO:"):
        return label, "biolink:Disease", True
    return f"demo:{label}", "biolink:NamedThing", False


def _withheld(disease, **over):
    return {"type": "ASSOC", "subject": "HGNC:1", "object": disease, "withheld": True, **over}


#: The column, and the mapping that makes it identifying.
WO = "demo:withheld_object"
PER_PAIR = dict(gap_identity_columns=[WO])


def _check(rep, *ids):
    return [next(c for c in rep.checks if c.id == i).passed for i in ids]


def test_two_objects_withheld_for_one_subject_are_two_gaps_and_each_names_its_object(tmp_path):
    """One gene withheld from one predicate for two diseases. The gap edge runs to the gap type,
    so the two rows had one id, and nothing in gaps.tsv said which disease either was about."""
    m = _pair_mapping(**PER_PAIR)
    ev = pd.DataFrame([_withheld("MONDO:1"), _withheld("MONDO:2"),
                       _withheld("MONDO:3", withheld=False)])
    ex = export(ev, m, resolver=_curies)
    assert [g[WO] for g in ex.gaps] == ["MONDO:1", "MONDO:2"]
    assert len({g["id"] for g in ex.gaps}) == 2 and ex.merged == 0
    # the edge is still subject -> gap type, and still says which predicate was refused
    assert {g["object"] for g in ex.gaps} == {"demo:GAP:precondition_unmet"}
    assert {g["demo:withheld_from"] for g in ex.gaps} == {"biolink:gene_associated_with_condition"}
    # it names a node as an asserted row's object does, and with the same category
    assert {"MONDO:1", "MONDO:2", "MONDO:3"} <= set(ex.nodes)
    assert {ex.nodes[d]["category"] for d in ("MONDO:1", "MONDO:2", "MONDO:3")} == {"biolink:Disease"}
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev), n_merged=ex.merged)
    assert _check(rep, "C6", "C7", "C8") == [True, True, True] and rep.passed, str(rep)
    # and in the file a consumer reads: beside the predicate it completes
    gaps = _written(ex, tmp_path, "gaps")
    cols = list(gaps.columns)
    assert cols.index(WO) == cols.index("demo:withheld_from") + 1
    assert list(gaps[WO]) == ["MONDO:1", "MONDO:2"]
    assert conformance.check_dir(tmp_path / "kgx", m, n_input_rows=len(ev)).passed


def test_two_objects_withheld_without_the_declaration_are_refused_and_never_merged():
    """What 0.5 did with two rows that agreed in every field it wrote: it kept one gap, counted the
    other as a repeat, and the export no longer held that two claims had been withheld. The object
    is now one of the fields, so the rows differ, and the refusal names the column to declare."""
    ex = Exporter(mapping=_pair_mapping(), resolver=_curies)
    ex.add(_withheld("MONDO:1"))
    with pytest.raises(ValueError, match=r"differ on \['demo:withheld_object'\]") as err:
        ex.add(_withheld("MONDO:2"))
    assert "'MONDO:1' and 'MONDO:2'" in str(err.value)
    assert "add 'demo:withheld_object' to Mapping.gap_identity_columns" in str(err.value)
    assert len(ex.gaps) == 1 and ex.merged_gaps == 0
    # two rows that differ in something measured as well are refused for the object too
    ex = Exporter(mapping=_pair_mapping(), resolver=_curies)
    ex.add(_withheld("MONDO:1", effect=0.3, floor=0.1))
    with pytest.raises(ValueError, match="Two objects were withheld for one subject"):
        ex.add(_withheld("MONDO:2", effect=0.3, floor=0.2))


def test_the_same_withheld_claim_said_twice_is_still_one_gap():
    """The control: one gene, one disease, withheld twice is one absence, with or without the
    declaration, and C8 is given the repeat."""
    for kw in ({}, PER_PAIR):
        m = _pair_mapping(**kw)
        ev = pd.DataFrame([_withheld("MONDO:1"), _withheld("MONDO:1")])
        ex = export(ev, m, resolver=_curies)
        assert len(ex.gaps) == 1 and ex.merged_gaps == 1 and ex.gaps[0][WO] == "MONDO:1"
        rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev), n_merged=ex.merged)
        assert rep.passed, str(rep)


def test_every_way_of_withholding_names_the_object():
    """A precondition, an effect under its floor, and an effect with no floor at all."""
    m = _pair_mapping()
    slope = {"type": "SLOPE", "subject": "HGNC:1", "object": "MONDO:1", "context": "cortex",
             "se": 0.08}
    ev = pd.DataFrame([_withheld("MONDO:1"), {**slope, "effect": 0.01, "floor": 0.2},
                       {**slope, "effect": 0.4, "floor": None}])
    ex = export(ev, m, resolver=_curies)
    assert [g["demo:gap_type"] for g in ex.gaps] == [
        "precondition_unmet", "under_detection_floor", "no_detection_floor"]
    assert [g[WO] for g in ex.gaps] == ["MONDO:1"] * 3 and ex.withheld == 3
    rep = conformance.check(*ex.frames(), m, n_input_rows=len(ev))
    assert rep.passed, str(rep)


def test_the_withheld_object_goes_through_the_resolver():
    """It is the CURIE the association would have carried, not the label the row happened to use,
    and Rule 3 holds for it as for any node."""
    def res(label):
        if label == "Alzheimer disease":
            return "MONDO:0004975", "biolink:Disease", True, "Alzheimer disease"
        return f"demo:{label}", "biolink:NamedThing", False

    m = _pair_mapping(**PER_PAIR)
    ex = Exporter(mapping=m, resolver=res)
    ex.add({"type": "ASSOC", "subject": "TREM2", "object": "Alzheimer disease", "withheld": True})
    ex.add({"type": "ASSOC", "subject": "TREM2", "object": "UNCODED_DISEASE", "withheld": True})
    ex.add({"type": "ASSOC", "subject": "CD33", "object": "Alzheimer disease"})
    assert [g[WO] for g in ex.gaps] == ["MONDO:0004975", "demo:UNCODED_DISEASE"]
    assert ex.gaps[0][WO] == ex.edges[0]["object"]
    assert ex.nodes["MONDO:0004975"]["name"] == "Alzheimer disease"
    assert ex.nodes["MONDO:0004975"]["id_grounded"] == "true"
    assert ex.nodes["demo:UNCODED_DISEASE"]["id_grounded"] == "false"
    assert "UNCODED_DISEASE" in ex.ungrounded
    rep = conformance.check(*ex.frames(), m, n_input_rows=3)
    assert _check(rep, "C6", "C7", "C8") == [True, True, True] and rep.passed, str(rep)
    # an ungrounded withheld object may not wear an ontology prefix either
    bad = Exporter(mapping=_pair_mapping(),
                   resolver=lambda label: (f"MONDO:{label}", "biolink:Disease", False))
    with pytest.raises(ValueError, match="Rule 3"):
        bad.add(_withheld("guessed"))


def test_a_source_vocabulary_gap_is_unchanged():
    """A row that was always a gap had no assertion withheld from it, so it has no withheld object:
    the column is empty, the label in its `object` cell is not made a node, and its id is the one
    0.4.0 gave it whether or not the producer declares the object."""
    for kw in ({}, PER_PAIR):
        e = _gap_exporter(**kw)
        e.add(_gap_row("G1"))
        g = e.gaps[0]
        assert g["id"] == "demo:g8db5e576ea19"       # the literal `test_the_id_recipe_is_pinned` holds
        assert g[WO] == "" and g["demo:withheld_from"] == ""
        assert set(e.nodes) == {"demo:G1", "demo:GAP:no_instrument"}
        assert conformance.check(*e.frames(), e.mapping, n_input_rows=1).passed
    # nor does it move beside a column of the producer's own, wherever the object is listed
    ids = []
    for declared in (["demo:cell_type_context"], [WO, "demo:cell_type_context"],
                     ["demo:cell_type_context", WO]):
        m = Mapping(prefix="demo", knowledge_source="infores:demo", gap_identity_columns=declared)
        m.gap_types.add("GAP")
        e = Exporter(mapping=m, extra_gap_columns=["demo:cell_type_context"],
                     extras=lambda row, kind: {"demo:cell_type_context": row.get("ct", "")})
        e.add(_gap_row("G1", ct="microglia"))
        ids.append(e.gaps[0]["id"])
    assert len(set(ids)) == 1


def test_a_withheld_gap_keeps_its_id_unless_the_object_is_declared():
    """An id is a promise to whoever stored it. These three were computed with the 0.5.2 code,
    before the column existed: writing the object must not re-address a producer that has one
    withheld assertion per gap and never asked for it to identify anything."""
    slope = {"type": "SLOPE", "subject": "HGNC:1", "object": "MONDO:1", "context": "cortex",
             "se": 0.08}
    rows = [_withheld("MONDO:1"), {**slope, "effect": 0.01, "floor": 0.2},
            {**slope, "effect": 0.4, "floor": None}]
    e = Exporter(mapping=_pair_mapping(), resolver=_curies)
    for r in rows:
        e.add(r)
    assert [g["id"] for g in e.gaps] == [
        "demo:g0e9a0a717613", "demo:g222b89a30c6a", "demo:g556ba6c2e07f"]
    assert [g[WO] for g in e.gaps] == ["MONDO:1"] * 3
    # declared, each is a different gap id: the object is part of what the gap is about
    d = Exporter(mapping=_pair_mapping(**PER_PAIR), resolver=_curies)
    for r in rows:
        d.add(r)
    assert not {g["id"] for g in d.gaps} & {g["id"] for g in e.gaps}


def test_declaring_the_object_keeps_the_ids_of_a_producer_that_carried_it_itself():
    """Before the column existed a producer wrote the object in a column of its own and declared
    that as identifying. These two ids are what 0.5.2 gave such a producer; dropping its column and
    keeping the declaration leaves them where they were."""
    e = Exporter(mapping=_pair_mapping(**PER_PAIR), resolver=_curies)
    e.add(_withheld("MONDO:1"))
    e.add(_withheld("MONDO:2"))
    assert [g["id"] for g in e.gaps] == ["demo:gaf119801c7ef", "demo:gca93bdf19ea5"]


def test_a_column_of_the_producers_own_may_not_repeat_one_the_file_carries():
    """The header would name it twice, and the producer's value would lose to the one written
    here. The refusal says what replaced a `withheld_object` column and what to keep."""
    with pytest.raises(KeyError, match="drop it from extra_gap_columns") as err:
        Exporter(mapping=_pair_mapping(**PER_PAIR), extra_gap_columns=[WO],
                 extras=lambda r, kind: {WO: r["object"]} if kind == "gap" else {})
    assert "keep it in Mapping.gap_identity_columns" in str(err.value)
    with pytest.raises(KeyError, match="extra_edge_columns names \\['predicate'\\]"):
        Exporter(mapping=_pair_mapping(), extra_edge_columns=["predicate"])
    # a column that is the producer's own in one file and the file's own in the other is no clash
    Exporter(mapping=_pair_mapping(), extra_edge_columns=[WO], extra_gap_columns=["demo:detected"])


def test_c7_catches_a_withheld_object_that_is_not_a_node():
    """The column is a reference: a reader follows it to a node as it follows `object`."""
    m = _pair_mapping(**PER_PAIR)
    ex = Exporter(mapping=m, resolver=_curies)
    ex.add(_withheld("MONDO:1"))
    nodes, edges, gaps = ex.frames()
    assert conformance.check(nodes, edges, gaps, m, n_input_rows=1).passed
    c7 = next(c for c in conformance.check(nodes[nodes["id"] != "MONDO:1"], edges, gaps, m,
                                           n_input_rows=1).checks if c.id == "C7")
    assert not c7.passed and "MONDO:1" in c7.detail
    # a file written before the column existed refers to nothing through it, and still passes
    old = conformance.check(nodes[nodes["id"] != "MONDO:1"], edges, gaps.drop(columns=[WO]), m,
                            n_input_rows=1)
    assert next(c for c in old.checks if c.id == "C7").passed


@pytest.mark.parametrize("absent", [None, float("nan"), pd.NA, "", " ", "nan", "None"])
def test_a_withheld_row_with_no_object_names_none(absent):
    """An absent object is written as nothing. It is not resolved into a node called `None`."""
    m = _pair_mapping()
    ex = Exporter(mapping=m, resolver=_curies)
    ex.add(_withheld(absent))
    assert ex.gaps[0][WO] == "" and ex.gaps[0]["demo:withheld_from"]
    assert set(ex.nodes) == {"HGNC:1", "demo:GAP:precondition_unmet"}
    rep = conformance.check(*ex.frames(), m, n_input_rows=1)
    assert rep.passed, str(rep)
