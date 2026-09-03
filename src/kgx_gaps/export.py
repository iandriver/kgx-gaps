"""The reference implementation of SPEC.md §2 — evidence rows in, KGX out.

The input is a table of EVIDENCE ROWS. The schema is small and documented in `EvidenceColumns`,
because the specification is worth nothing if using it requires reading this file.

The three rules live in `Exporter.add()` and nowhere else, so there is one place to read to know what
gets refused. In particular there is no path that emits an association without going through Rule 1 —
a caller in a hurry cannot skip it, which is the failure mode a rule written only in prose has.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .mapping import Mapping, Rule


# --------------------------------------------------------------------------- input schema

@dataclass(frozen=True)
class EvidenceColumns:
    """Which column of the caller's frame carries what. Rename here rather than reshaping the frame.

    Only `type`, `subject` and `object` are required. `effect` + `floor` together make a row a
    MEASUREMENT and therefore subject to Rule 1; a row lacking either is a categorical claim and is
    exported on its own terms (SPEC.md §2, Rule 1, second paragraph).
    """
    type: str = "type"
    subject: str = "subject"
    object: str = "object"
    effect: str = "effect"
    se: str = "se"
    floor: str = "floor"
    detected: str = "detected"
    context: str = "context"
    reason: str = "reason"
    gap_type: str = "gap_type"
    n_required: str = "n_required"
    proposal: str = "proposal"
    kill_condition: str = "kill_condition"


NODE_COLS = ["id", "category", "name", "provided_by", "id_grounded", "xref"]


def _edge_cols(prefix: str, extra: list[str] | None = None) -> list[str]:
    p = prefix
    return ["id", "subject", "predicate", "object", "category",
            "primary_knowledge_source", "knowledge_level", "agent_type",
            "subject_aspect_qualifier", "object_direction_qualifier",
            "anatomical_context_qualifier",
            f"{p}:context", f"{p}:effect_size", f"{p}:standard_error", f"{p}:detection_floor",
            f"{p}:detected", f"{p}:source_edge_type"] + list(extra or [])


def _gap_cols(prefix: str, extra: list[str] | None = None) -> list[str]:
    p = prefix
    return ["id", "subject", "predicate", "object", "category",
            "primary_knowledge_source", "knowledge_level", "agent_type",
            "anatomical_context_qualifier", f"{p}:context",
            f"{p}:gap_type", f"{p}:gap_reason", f"{p}:detection_floor", f"{p}:n_required",
            f"{p}:proposal", f"{p}:kill_condition", f"{p}:withheld_from",
            f"{p}:source_edge_type"] + list(extra or [])


def blank(v) -> str | float:
    """A finite float, or the empty string. SPEC.md §4: never the literal 'nan'."""
    if v is None:
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        s = str(v)
        return "" if s.lower() in ("nan", "none", "<na>") else s
    return "" if x != x or x in (float("inf"), float("-inf")) else x


# --------------------------------------------------------------------------- the exporter

@dataclass
class Exporter:
    """Applies the three rules. `nodes`, `edges` and `gaps` accumulate; `write()` serialises."""
    mapping: Mapping
    cols: EvidenceColumns = field(default_factory=EvidenceColumns)
    #: subject/object label -> (curie, category, grounded). Default: flag everything ungrounded.
    resolver: Callable[[str], tuple[str, str, bool]] | None = None
    #: context label -> {"anatomical": curie|None, ...}, merged into the row as qualifiers.
    context_resolver: Callable[[str], dict] | None = None
    #: Domain attribute columns, declared here so their ORDER in the TSV is stable across runs.
    #: A column that appeared only when some row happened to populate it would make two exports of
    #: the same graph diff against each other for no reason.
    extra_edge_columns: list[str] = field(default_factory=list)
    extra_gap_columns: list[str] = field(default_factory=list)
    #: (row, "edge"|"gap") -> {column: value} for the declared columns above. Values are passed
    #: through `blank()`, so an absent one is "" and never the string "nan" (SPEC.md §4).
    extras: Callable[[dict, str], dict] | None = None

    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    withheld: int = 0
    ungrounded: set[str] = field(default_factory=set)

    # -- nodes -------------------------------------------------------------
    def _resolve(self, label: str) -> str:
        if self.resolver is not None:
            curie, category, grounded = self.resolver(label)
        else:
            curie, category, grounded = f"{self.mapping.prefix}:{label}", "biolink:NamedThing", False
        if not grounded:
            self.ungrounded.add(str(label))
        # Rule 3: an ungrounded id must never wear a real ontology prefix.
        if not grounded and not curie.startswith(f"{self.mapping.prefix}:"):
            raise ValueError(
                f"resolver returned grounded=False for {label!r} but the CURIE {curie!r} is not under "
                f"the implementation prefix {self.mapping.prefix!r} (SPEC.md §2, Rule 3)")
        self.node(curie, category, str(label), grounded)
        return curie

    def node(self, curie: str, category: str, name: str = "", grounded: bool = True, xref: str = ""):
        if curie and curie not in self.nodes:
            self.nodes[curie] = dict(id=curie, category=category, name=name or curie,
                                     provided_by=self.mapping.knowledge_source,
                                     id_grounded=str(bool(grounded)).lower(), xref=xref)

    # -- the rules ---------------------------------------------------------
    def _extras(self, row: dict, kind: str) -> dict:
        if self.extras is None:
            return {}
        declared = set(self.extra_edge_columns if kind == "edge" else self.extra_gap_columns)
        got = self.extras(row, kind) or {}
        undeclared = set(got) - declared
        if undeclared:
            raise KeyError(
                f"extras() returned undeclared {kind} column(s) {sorted(undeclared)}; add them to "
                f"extra_{kind}_columns so the TSV column order is stable")
        return {k: blank(v) for k, v in got.items()}

    def add(self, row: dict) -> str:
        """Route one evidence row. Returns 'edge' or 'gap' — never nothing (SPEC.md §5, C8)."""
        c = self.cols
        etype = str(row.get(c.type) or "")
        subj = self._resolve(row.get(c.subject))
        context = str(blank(row.get(c.context)) or "")
        ctx = (self.context_resolver(context) if self.context_resolver else {}) or {}
        common = {
            "anatomical_context_qualifier": ctx.get("anatomical") or "",
            f"{self.mapping.prefix}:context": context,
            f"{self.mapping.prefix}:source_edge_type": etype,
        }
        for extra in ctx.get("nodes", []):
            self.node(*extra)

        # -- rows that are already gaps in the source vocabulary ------------
        if etype in self.mapping.gap_types:
            gap = str(row.get(c.gap_type) or row.get(c.object) or etype)
            self._gap(subj, gap, common, reason=str(row.get(c.reason) or ""),
                      floor=row.get(c.floor), n_required=row.get(c.n_required),
                      proposal=str(row.get(c.proposal) or ""),
                      kill=str(row.get(c.kill_condition) or ""), withheld_from="", row=row)
            return "gap"

        rule = self.mapping.rules.get(etype)
        if rule is None:
            raise KeyError(f"no Rule for source edge type {etype!r}; add it to Mapping.rules")

        # -- RULE 1, preconditions ------------------------------------------
        # Checked BEFORE the floor: a row with an invalid instrument has no assertion to make
        # regardless of how large its effect looks, and reporting "under floor" for it would state
        # the wrong reason.
        predicate = rule.predicate_for(row)
        if rule.precondition is not None:
            why = rule.precondition(row)
            if why:
                self.withheld += 1
                self._gap(subj, "precondition_unmet", common, reason=why,
                          floor=row.get(c.floor), n_required=row.get(c.n_required),
                          proposal="", kill="", withheld_from=predicate)
                return "gap"

        # -- RULE 1, detection floor ----------------------------------------
        effect, floor = blank(row.get(c.effect)), blank(row.get(c.floor))
        measured = effect != "" and floor != ""
        detected = row.get(c.detected)
        if detected is None and measured:
            detected = abs(float(effect)) > float(floor)
        if measured and not bool(detected):
            self.withheld += 1
            self._gap(subj, "under_detection_floor", common,
                      reason=(f"|effect| does not exceed its detection floor, so no {etype} "
                              f"assertion is exported; the effect is bounded, not shown absent"),
                      floor=floor, n_required=row.get(c.n_required), proposal="", kill="",
                      withheld_from=predicate, row=row)
            return "gap"

        obj = self._resolve(row.get(c.object))
        self.edges.append({**{k: "" for k in _edge_cols(self.mapping.prefix, self.extra_edge_columns)},
                           **common, **self._extras(row, "edge"), **{
            "id": f"{self.mapping.prefix}:e{len(self.edges):06d}",
            "subject": subj, "predicate": predicate, "object": obj, "category": rule.category,
            "primary_knowledge_source": self.mapping.knowledge_source,
            "knowledge_level": rule.knowledge_level, "agent_type": rule.agent_type,
            "subject_aspect_qualifier": rule.subject_aspect_qualifier,
            "object_direction_qualifier": (
                ("increased" if float(effect) > 0 else "decreased")
                if rule.directional and effect != "" else ""),
            f"{self.mapping.prefix}:effect_size": effect,
            f"{self.mapping.prefix}:standard_error": blank(row.get(c.se)),
            f"{self.mapping.prefix}:detection_floor": floor,
            # Rule 1's other half: a row with no effect size was never a measurement, and saying
            # 'false' here would claim it failed a test it was never given.
            f"{self.mapping.prefix}:detected": "true" if measured else "not_applicable",
        }})
        return "edge"

    def _gap(self, subj, gap_local, common, *, reason, floor, n_required, proposal, kill,
             withheld_from, row=None):
        p = self.mapping.prefix
        obj = f"{p}:GAP:{gap_local}"
        self.node(obj, f"{p}:KnowledgeGap", gap_local.replace("_", " "), grounded=False)
        self.gaps.append({**{k: "" for k in _gap_cols(p, self.extra_gap_columns)}, **common,
                          **self._extras(row or {}, "gap"), **{
            "id": f"{p}:g{len(self.gaps):06d}", "subject": subj,
            "predicate": self.mapping.gap_predicate, "object": obj,
            "category": f"{p}:KnowledgeGapAssociation",
            "primary_knowledge_source": self.mapping.knowledge_source,
            "knowledge_level": "logical_entailment", "agent_type": "data_analysis_pipeline",
            f"{p}:gap_type": gap_local, f"{p}:gap_reason": reason,
            f"{p}:detection_floor": blank(floor), f"{p}:n_required": blank(n_required),
            f"{p}:proposal": proposal, f"{p}:kill_condition": kill,
            f"{p}:withheld_from": withheld_from,
        }})

    # -- io ----------------------------------------------------------------
    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        p = self.mapping.prefix
        return (pd.DataFrame(list(self.nodes.values()), columns=NODE_COLS),
                pd.DataFrame(self.edges, columns=_edge_cols(p, self.extra_edge_columns)),
                pd.DataFrame(self.gaps, columns=_gap_cols(p, self.extra_gap_columns)))

    def write(self, out: Path) -> dict[str, Path]:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        n, e, g = self.frames()
        files = {"nodes": out / "nodes.tsv", "edges": out / "edges.tsv", "gaps": out / "gaps.tsv"}
        for key, df in (("nodes", n), ("edges", e), ("gaps", g)):
            df.to_csv(files[key], sep="\t", index=False, na_rep="")
        return files


def export(evidence: pd.DataFrame, mapping: Mapping, **kw) -> Exporter:
    """Convenience: run every row of a frame through the rules."""
    ex = Exporter(mapping=mapping, **kw)
    for _, row in evidence.iterrows():
        ex.add(row.to_dict())
    return ex
