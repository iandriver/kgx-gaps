"""Conformance tests for the reference implementation, keyed to SPEC.md.

Each test names the rule or check it defends. The negative controls matter more than the positives:
a suite that only demonstrates the happy path cannot tell you the rules are enforced.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kgx_gaps import Mapping, Rule, conformance, export
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
