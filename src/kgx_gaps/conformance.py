"""SPEC.md §5 — the ten conformance checks, runnable against ANY KGX triple.

This is the part of the package that is useful to someone who has not adopted the exporter. Point it
at three TSVs produced by any pipeline and it reports which of the rules that output satisfies. C1,
C2 and C10 in particular can be evaluated on a foreign graph without knowing anything about how it
was built — they only need the floor and effect columns to be present, and their absence is itself
the finding.

Every check reports the offending rows, not just a verdict. A conformance report that says "C1
failed" and nothing else cannot be acted on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .mapping import BIOLINK, Mapping

CURIE = r"^[A-Za-z0-9._]+:\S+$"
NAN_STRINGS = {"nan", "NaN", "None", "<NA>", "NAN"}


@dataclass
class Check:
    id: str
    title: str
    passed: bool
    detail: str = ""
    offenders: list = field(default_factory=list)
    skipped: bool = False

    def __str__(self) -> str:
        if self.skipped:
            return f"  {self.id}  SKIP  {self.title} — {self.detail}"
        mark = "PASS" if self.passed else "FAIL"
        s = f"  {self.id}  {mark}  {self.title}"
        if not self.passed and self.detail:
            s += f"\n         {self.detail}"
        return s


@dataclass
class Report:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        # A SKIP IS NOT A PASS. A conformance report whose skips fold into its pass count is exactly
        # the ambiguity this specification exists to remove, so a skipped check fails the report and
        # says which one it was.
        return all(c.passed and not c.skipped for c in self.checks)

    @property
    def n_skipped(self) -> int:
        return sum(c.skipped for c in self.checks)

    def __str__(self) -> str:
        body = "\n".join(str(c) for c in self.checks)
        ok = sum(c.passed and not c.skipped for c in self.checks)
        tail = f"\n\n  {ok}/{len(self.checks)} checks passed"
        if self.n_skipped:
            tail += f", {self.n_skipped} SKIPPED (not a pass)"
        return body + tail


def _col(df: pd.DataFrame, *names) -> str | None:
    """Find a column by suffix, so `ex:detection_floor` matches regardless of prefix."""
    for n in names:
        for c in df.columns:
            if c == n or c.endswith(":" + n):
                return c
    return None


def check(nodes: pd.DataFrame, edges: pd.DataFrame, gaps: pd.DataFrame,
          mapping: Mapping | None = None, n_input_rows: int | None = None) -> Report:
    nodes, edges, gaps = (d.fillna("") if len(d) else d for d in (nodes, edges, gaps))
    out: list[Check] = []

    eff = _col(edges, "effect_size", "effect")
    flo = _col(edges, "detection_floor", "floor")
    det = _col(edges, "detected")

    # -- C1 -----------------------------------------------------------------
    if eff is None or flo is None:
        out.append(Check("C1", "no association is under its own detection floor", False,
                         f"edges.tsv has no {'effect' if eff is None else 'floor'} column, so the "
                         f"rule cannot be evaluated — floors were not carried through the export",
                         skipped=True))
    else:
        num = edges[[eff, flo]].apply(pd.to_numeric, errors="coerce")
        bad = edges[(num[eff].abs() <= num[flo]) & num[eff].notna() & num[flo].notna()]
        out.append(Check("C1", "no association is under its own detection floor", not len(bad),
                         f"{len(bad)} association(s) fail to exceed their floor",
                         bad.head(5).to_dict("records")))

    # -- C2 -----------------------------------------------------------------
    neg = [c for c in edges.columns if c == "negated" or c.endswith(":negated")]
    truthy = []
    for c in neg:
        truthy += [r for r in edges[c].astype(str) if r.strip().lower() in ("true", "1", "yes")]
    out.append(Check("C2", "`negated` is not used to express an under-floor result", not truthy,
                     f"{len(truthy)} row(s) set {neg}"))

    # -- C3 -----------------------------------------------------------------
    preds = set(edges.get("predicate", pd.Series(dtype=str))) | set(gaps.get("predicate", pd.Series(dtype=str)))
    if mapping is None:
        unreg = {p for p in preds if p and not p.startswith(BIOLINK)}
        out.append(Check("C3", "every non-BioLink predicate has a registered rationale", False,
                         f"no Mapping supplied; {len(unreg)} non-BioLink predicate(s) present and "
                         f"their rationales cannot be checked: {sorted(unreg)[:4]}", skipped=True))
    else:
        unreg = mapping.unregistered(preds)
        out.append(Check("C3", "every non-BioLink predicate has a registered rationale", not unreg,
                         f"unregistered: {sorted(unreg)}"))

    # -- C4 -----------------------------------------------------------------
    wf, gflo = _col(gaps, "withheld_from"), _col(gaps, "detection_floor")
    if wf is None:
        out.append(Check("C4", "every withheld row names its predicate and floor", False,
                         "gaps.tsv has no withheld_from column", skipped=True))
    else:
        wh = gaps[gaps[wf].astype(str).str.strip() != ""]
        # A row withheld BECAUSE no floor was computed cannot carry one; demanding it would make the
        # third clause of Rule 1 unsatisfiable. Every other withheld row must.
        gt = _col(gaps, "gap_type")
        if gt is not None:
            wh = wh[wh[gt].astype(str) != "no_detection_floor"]
        # A withheld row that was never a measurement has no floor to carry — a categorical claim
        # refused by a precondition, say. Only rows that reported a number owe one.
        wm = _col(gaps, "was_measurement")
        if wm is not None:
            wh = wh[wh[wm].astype(str).str.lower() != "false"]
        miss = wh if gflo is None else wh[wh[gflo].astype(str).str.strip() == ""]
        out.append(Check("C4", "every withheld row names its predicate, and its floor unless none exists",
                         not len(miss),
                         f"{len(miss)} withheld row(s) carry no detection floor",
                         miss.head(5).to_dict("records")))

    # -- C5 -----------------------------------------------------------------
    gr = _col(gaps, "gap_reason", "reason")
    if gr is None:
        out.append(Check("C5", "every gap carries a reason", False,
                         "gaps.tsv has no gap_reason column", skipped=True))
    else:
        blank = gaps[gaps[gr].astype(str).str.strip() == ""]
        out.append(Check("C5", "every gap carries a reason", not len(blank),
                         f"{len(blank)} gap(s) have an empty reason"))

    # -- C6 -----------------------------------------------------------------
    ids = nodes.get("id", pd.Series(dtype=str)).astype(str)
    noncurie = list(ids[~ids.str.match(CURIE)])
    grounded = _col(nodes, "id_grounded")
    wrong_prefix = []
    if grounded is not None and mapping is not None:
        ung = nodes[nodes[grounded].astype(str).str.lower() == "false"]
        wrong_prefix = list(ung[~ung["id"].astype(str).str.startswith(mapping.prefix + ":")]["id"])
    out.append(Check("C6", "ids are CURIEs; ungrounded ids are flagged under the local prefix",
                     not noncurie and not wrong_prefix,
                     f"non-CURIE: {noncurie[:3]}; ungrounded under a real ontology prefix: "
                     f"{wrong_prefix[:3]}"))

    # -- C7 -----------------------------------------------------------------
    refs = set()
    for d in (edges, gaps):
        for c in ("subject", "object"):
            if c in d.columns:
                refs |= set(d[c].astype(str))
    dangling = sorted(r for r in refs - set(ids) if r)
    out.append(Check("C7", "every referenced node exists in nodes.tsv", not dangling,
                     f"{len(dangling)} dangling: {dangling[:3]}"))

    # -- C8 -----------------------------------------------------------------
    if n_input_rows is None:
        out.append(Check("C8", "associations + gaps account for every input row", False,
                         "n_input_rows not supplied, so conservation cannot be checked", skipped=True))
    else:
        # Structural edges -- an ontology hierarchy, a disease-to-severity-axis link -- are not
        # derived from any evidence row and must not be counted against conservation. They are
        # identified by an empty source_edge_type (SPEC.md §4). Counting them here was this check's
        # first bug, found by running it against a real graph that legitimately carried one.
        src = _col(edges, "source_edge_type")
        if src is None:
            derived, structural = edges, edges.iloc[0:0]
        else:
            mask = edges[src].astype(str).str.strip() != ""
            derived, structural = edges[mask], edges[~mask]
        got = len(derived) + len(gaps)
        note = f" (+{len(structural)} structural, excluded)" if len(structural) else ""
        out.append(Check("C8", "evidence-derived associations + gaps account for every input row",
                         got == n_input_rows,
                         f"{len(derived)} edges + {len(gaps)} gaps = {got}, input was "
                         f"{n_input_rows}{note}"))

    # -- C9 -----------------------------------------------------------------
    nan_hits = []
    for name, d in (("edges", edges), ("gaps", gaps), ("nodes", nodes)):
        if len(d):
            m = d.astype(str).isin(NAN_STRINGS)
            if m.to_numpy().any():
                nan_hits.append(f"{name}({int(m.to_numpy().sum())})")
    out.append(Check("C9", "no field serialised as nan/None", not nan_hits, f"found in {nan_hits}"))

    # -- C10 ----------------------------------------------------------------
    if det is None:
        out.append(Check("C10", "rows with no effect size are not diverted to gaps", False,
                         "edges.tsv has no `detected` column", skipped=True))
    else:
        num = pd.to_numeric(edges[eff], errors="coerce") if eff else pd.Series(dtype=float)
        na_rows = edges[edges[det].astype(str) == "not_applicable"]
        wrong = na_rows if eff is None else na_rows[pd.to_numeric(na_rows[eff], errors="coerce").notna()]
        out.append(Check("C10", "rows with no effect size are not diverted to gaps", not len(wrong),
                         f"{len(wrong)} row(s) marked not_applicable yet carry an effect size"))

    return Report(out)


def check_dir(path: Path, mapping: Mapping | None = None, n_input_rows: int | None = None) -> Report:
    """Run the suite over a directory of nodes.tsv / edges.tsv / gaps.tsv."""
    path = Path(path)
    def rd(name):
        p = path / name
        return pd.read_csv(p, sep="\t", dtype=str) if p.exists() else pd.DataFrame()
    return check(rd("nodes.tsv"), rd("edges.tsv"), rd("gaps.tsv"), mapping, n_input_rows)
