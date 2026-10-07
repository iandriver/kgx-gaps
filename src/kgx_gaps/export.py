"""The reference implementation of SPEC.md §2 — evidence rows in, KGX out.

The input is a table of EVIDENCE ROWS. The schema is small and documented in `EvidenceColumns`,
because the specification is worth nothing if using it requires reading this file.

The three rules live in `Exporter.add()` and nowhere else, so there is one place to read to know what
gets refused. In particular there is no path that emits an association without going through Rule 1 —
a caller in a hurry cannot skip it, which is the failure mode a rule written only in prose has.
"""
from __future__ import annotations

import decimal
import hashlib
import math
import numbers

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .mapping import AGENT_TYPES, KNOWLEDGE_LEVELS, Mapping, Rule


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
            f"{p}:was_measurement", f"{p}:source_edge_type"] + list(extra or [])


def edge_identity(prefix: str, extra: list[str] | None = None) -> list[str]:
    """The columns of edges.tsv that say what an edge CLAIMS, in the order `Exporter.edge_id` hashes
    them. `extra` is `Mapping.edge_identity_columns`."""
    return ["subject", "predicate", "object", "subject_aspect_qualifier",
            "object_direction_qualifier", "anatomical_context_qualifier", f"{prefix}:context",
            "primary_knowledge_source", f"{prefix}:source_edge_type"] + list(extra or [])


#: What Python and pandas print for a value that is not there.
_ABSENT = ("nan", "none", "<na>")


def blank(v) -> str | bool | int | float:
    """The value as the producer gave it, or the empty string for an absent one (SPEC.md §4).

    Absent is None, NaN, pandas' NA and NaT, an infinity, and the strings 'nan', 'none' and '<NA>'
    in any case, so the literal 'nan' never reaches a file. A value that is present keeps its type:
    a string is returned as it is, a bool as a bool, a whole number as an int, and any other real
    number as a float.

    WHY NOT `float(v)` FIRST. That is what this did, and whatever parsed as a number came back a
    float. A year passed as an attribute was written `2018.0` and a count `1.0`; an identifier that
    happened to be digits, `"0012"`, was written `12.0`, which is a different identifier. Only a
    number is a number here. The three columns that are real-valued by definition go through
    `_real` instead.
    """
    if v is None or (pd.api.types.is_scalar(v) and pd.isna(v)):
        return ""
    if isinstance(v, str):
        return "" if v.strip().lower() in _ABSENT else v
    if pd.api.types.is_bool(v):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, (numbers.Real, decimal.Decimal)):
        x = float(v)
        return x if math.isfinite(x) else ""
    s = str(v)
    return "" if s.strip().lower() in _ABSENT else s


def _real(v) -> str | float:
    """A finite float, or the empty string: an effect size, its standard error, a detection floor.

    These are measurements, so a numeric string or an int is read as the number it states and an
    infinity as no number at all. A string that states no number is returned as it is, for the
    comparison with the floor to refuse.
    """
    v = blank(v)
    if isinstance(v, str) and not v.strip():
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return x if math.isfinite(x) else ""


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
    #: through `blank()`, so an absent one is "" and never the string "nan" (SPEC.md §4), and a
    #: present one is written with the type it was given: an int as an int, a string as itself.
    extras: Callable[[dict, str], dict] | None = None

    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    #: Gap id -> the row written under it, so a repeat is merged when identical and refused when not.
    _gap_ids: dict[str, dict] = field(default_factory=dict)
    #: Identical gaps dropped because another row already said the same thing.
    merged_gaps: int = 0
    #: The same two for associations.
    _edge_ids: dict[str, dict] = field(default_factory=dict)
    merged_edges: int = 0
    withheld: int = 0
    ungrounded: set[str] = field(default_factory=set)

    def __post_init__(self):
        # An identity column that is not a column of the file identifies nothing: every row would
        # hash the empty string for it, and the rows it was declared to tell apart would be refused
        # as differing -- or, worse, merged. Checked here so a misspelt name fails before any row.
        p = self.mapping.prefix
        for kind, declared, cols in (
                ("edge", self.mapping.edge_identity_columns, _edge_cols(p, self.extra_edge_columns)),
                ("gap", self.mapping.gap_identity_columns, _gap_cols(p, self.extra_gap_columns))):
            unknown = [c for c in declared if c not in cols]
            if unknown:
                raise KeyError(
                    f"Mapping.{kind}_identity_columns names {unknown}, which {kind}s.tsv does not "
                    f"carry; declare them in extra_{kind}_columns, with the prefix, as they appear "
                    f"in the file")

    @property
    def merged(self) -> int:
        """Input rows that repeated a row already written, field for field (C8's third term)."""
        return self.merged_edges + self.merged_gaps

    # -- nodes -------------------------------------------------------------
    def _resolve(self, label: str) -> str:
        name = ""
        if self.resolver is not None:
            # A resolver may also NAME the node, as a fourth element. Without one the name is the
            # label, so a node reached by two labels -- a gene symbol in one row, its accession in
            # another -- is named after whichever came first. A resolver that maps several labels
            # to one CURIE should name it.
            curie, category, grounded, *rest = self.resolver(label)
            name = str(rest[0]) if rest and rest[0] else ""
        else:
            curie, category, grounded = f"{self.mapping.prefix}:{label}", "biolink:NamedThing", False
        if not grounded:
            self.ungrounded.add(str(label))
        # Rule 3: an ungrounded id must never wear a real ontology prefix.
        if not grounded and not curie.startswith(f"{self.mapping.prefix}:"):
            raise ValueError(
                f"resolver returned grounded=False for {label!r} but the CURIE {curie!r} is not under "
                f"the implementation prefix {self.mapping.prefix!r} (SPEC.md §2, Rule 3)")
        self.node(curie, category, name or str(label), grounded)
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
        # `str(... or "")` would drop a context labelled 0, a cluster number say.
        context = str(blank(row.get(c.context)))
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
                          proposal="", kill="", withheld_from=predicate, row=row,
                          was_measurement=_real(row.get(c.effect)) != "")
                return "gap"

        # -- RULE 1, detection floor ----------------------------------------
        effect, floor = _real(row.get(c.effect)), _real(row.get(c.floor))
        measured = effect != "" and floor != ""
        # An effect WITHOUT a floor is the case Rule 1 originally left ambiguous, and the reference
        # implementation resolved it the wrong way: it treated the row as "not a measurement" and
        # asserted it with detected=not_applicable, which its own C10 then flagged. An effect size is
        # a measurement by definition; what is missing is the bound. Such a row cannot be asserted --
        # nothing says whether the number is real -- so it becomes a Gap saying exactly that.
        if effect != "" and floor == "":
            self.withheld += 1
            self._gap(subj, "no_detection_floor", common,
                      reason=("an effect size is reported but no detection floor was computed, so "
                              "nothing bounds it: the estimate cannot be distinguished from one this "
                              "study had no power to see"),
                      floor="", n_required=row.get(c.n_required), proposal="", kill="",
                      withheld_from=predicate, row=row, was_measurement=True)
            return "gap"
        detected = row.get(c.detected)
        if detected is None and measured:
            detected = abs(float(effect)) > float(floor)
        if measured and not bool(detected):
            self.withheld += 1
            self._gap(subj, "under_detection_floor", common,
                      reason=(f"|effect| does not exceed its detection floor, so no {etype} "
                              f"assertion is exported; the effect is bounded, not shown absent"),
                      floor=floor, n_required=row.get(c.n_required), proposal="", kill="",
                      withheld_from=predicate, row=row, was_measurement=True)
            return "gap"

        obj = self._resolve(row.get(c.object))
        self._edge({**{k: "" for k in _edge_cols(self.mapping.prefix, self.extra_edge_columns)},
                    **common, **self._extras(row, "edge"), **{
            "subject": subj, "predicate": predicate, "object": obj, "category": rule.category,
            "primary_knowledge_source": self.mapping.knowledge_source,
            "knowledge_level": rule.knowledge_level, "agent_type": rule.agent_type,
            "subject_aspect_qualifier": rule.subject_aspect_qualifier,
            "object_direction_qualifier": (
                ("increased" if float(effect) > 0 else "decreased")
                if rule.directional and effect != "" else ""),
            f"{self.mapping.prefix}:effect_size": effect,
            f"{self.mapping.prefix}:standard_error": _real(row.get(c.se)),
            f"{self.mapping.prefix}:detection_floor": floor,
            # Rule 1's other half: a row with no effect size was never a measurement, and saying
            # 'false' here would claim it failed a test it was never given.
            f"{self.mapping.prefix}:detected": "true" if measured else "not_applicable",
        }})
        return "edge"

    def edge_id(self, row: dict) -> str:
        """An association's id, derived from what the edge CLAIMS and from nothing measured about it.

        WHY NOT A COUNTER. The first version numbered edges in emission order, `e000000`, `e000001`.
        Every export therefore began at zero: three exports of one producer, concatenated, held 2,814
        edge rows under 2,021 distinct ids, and each of the 434 shared ids named a different
        subject and object in each file. Within one export the counter had the fault the gap counter
        had -- withhold one row and every later id shifts. Identity is the triple, its three
        qualifiers, the context, the knowledge source, the source edge type, and whichever of the
        producer's own columns it declares as identifying. The effect size, its standard error, the
        floor and `detected` are measurements ABOUT the claim and are excluded. The direction
        qualifier is not: an edge that said `decreased` and now says `increased` is a different
        claim, and a citation of the first must not come to point at the second.

        `row` is an output row, keyed by the columns of edges.tsv.
        """
        p = self.mapping.prefix
        parts = ["" if row.get(c) is None else str(row.get(c))
                 for c in edge_identity(p, self.mapping.edge_identity_columns)]
        return f"{p}:e{hashlib.sha256(chr(31).join(parts).encode()).hexdigest()[:12]}"

    def _edge(self, row_out: dict, evidence: bool = True) -> str:
        """Write one association under its content-derived id. One identity, one row -- as `_gap`.

        `evidence` is False for a structural edge: said twice it is still written once, but it was
        never an input row, so it is not counted among the merged rows C8 is given."""
        p = self.mapping.prefix
        eid = row_out["id"] = self.edge_id(row_out)
        # Two evidence rows that agree in every field are one claim said twice: the second is
        # dropped and counted, and C8 is given the count. Two that share an identity and DIFFER --
        # two effect sizes for one gene in one context, say -- are two measurements the id cannot
        # tell apart, so the export refuses and names the field rather than keeping whichever came
        # first.
        first = self._edge_ids.get(eid)
        if first is not None:
            differing = sorted(k for k in set(first) | set(row_out)
                               if k != "id" and str(first.get(k, "")) != str(row_out.get(k, "")))
            if not differing:
                self.merged_edges += evidence
                return eid
            raise ValueError(
                f"two edges share the identity {eid} but differ on {differing[:4]}: "
                f"subject={row_out.get('subject')!r} predicate={row_out.get('predicate')!r} "
                f"object={row_out.get('object')!r} context={row_out.get(f'{p}:context')!r}. Declare "
                f"the column that tells them apart in Mapping.edge_identity_columns, or emit one "
                f"edge.")
        self._edge_ids[eid] = row_out
        self.edges.append(row_out)
        return eid

    def structural(self, subject: str, predicate: str, object: str, *,
                   category: str = "biolink:Association",
                   knowledge_level: str = "knowledge_assertion", agent_type: str = "not_provided",
                   fields: dict | None = None) -> str:
        """Add an edge no evidence row produced, and return its id (SPEC.md §4).

        An ontology hierarchy, a link from a disease to the axis its severity is scored on. Its
        `source_edge_type` stays empty, which is what excludes it from conservation (C8), and its id
        is derived exactly as an evidence-derived edge's is. `subject` and `object` are CURIEs
        already added with `node()`; `fields` fills declared columns of edges.tsv.
        """
        p = self.mapping.prefix
        cols = _edge_cols(p, self.extra_edge_columns)
        fields = dict(fields or {})
        fixed = {"id", "subject", "predicate", "object", "category", "knowledge_level",
                 "agent_type", "primary_knowledge_source", f"{p}:source_edge_type"}
        bad = sorted(k for k in fields if k not in cols or k in fixed)
        if bad:
            raise KeyError(
                f"structural() cannot set {bad}: a field must be a declared column of edges.tsv, "
                f"and the id, the triple, the provenance slots and source_edge_type are not "
                f"`fields`")
        missing = [n for n in (subject, object) if n not in self.nodes]
        if missing:
            raise KeyError(f"structural() refers to {missing}, not in nodes; add them with node()")
        if self.mapping.unregistered([predicate]):
            raise ValueError(f"predicate {predicate!r} is minted but unregistered (SPEC.md §2)")
        if knowledge_level not in KNOWLEDGE_LEVELS or agent_type not in AGENT_TYPES:
            raise ValueError(f"knowledge_level {knowledge_level!r} / agent_type {agent_type!r} "
                             f"is not in BioLink's enum")
        return self._edge({**{k: "" for k in cols}, **{k: blank(v) for k, v in fields.items()}, **{
            "subject": subject, "predicate": predicate, "object": object, "category": category,
            "primary_knowledge_source": self.mapping.knowledge_source,
            "knowledge_level": knowledge_level, "agent_type": agent_type}}, evidence=False)

    def gap_id(self, subj, gap_local, common, withheld_from, extras) -> str:
        """A gap's id, derived from what the gap is ABOUT and from nothing that moves.

        WHY NOT A COUNTER. The first version numbered gaps in emission order, `g000000`, `g000001`.
        Close one gap and every later id shifts by one, so an id named a row position rather than an
        absence, and anything recorded against it -- a closure, a citation, a memory claim -- pointed
        at a different gap after the next build. Identity is the subject, the gap type, the context,
        the row withheld, and whichever of the producer's own columns it declares as identifying.
        The floor, the reason, the proposal and the counts are measurements ABOUT the gap: they are
        expected to move while the gap stays the same gap, so they are excluded.
        """
        p = self.mapping.prefix
        parts = [str(subj), str(gap_local), str(withheld_from or ""),
                 str(common.get("anatomical_context_qualifier") or ""),
                 str(common.get(f"{p}:context") or ""),
                 str(common.get(f"{p}:source_edge_type") or "")]
        parts += [str((extras or {}).get(c, "") or "") for c in self.mapping.gap_identity_columns]
        return f"{p}:g{hashlib.sha256(chr(31).join(parts).encode()).hexdigest()[:12]}"

    def _gap(self, subj, gap_local, common, *, reason, floor, n_required, proposal, kill,
             withheld_from, row=None, was_measurement=None):
        p = self.mapping.prefix
        obj = f"{p}:GAP:{gap_local}"
        self.node(obj, f"{p}:KnowledgeGap",
                  self.mapping.gap_names.get(gap_local) or gap_local.replace("_", " "),
                  grounded=False)
        extras = self._extras(row or {}, "gap")
        gid = self.gap_id(subj, gap_local, common, withheld_from, extras)
        row_out = {**{k: "" for k in _gap_cols(p, self.extra_gap_columns)}, **common,
                   **extras, **{
            "id": gid, "subject": subj,
            "predicate": self.mapping.gap_predicate, "object": obj,
            "category": f"{p}:KnowledgeGapAssociation",
            "primary_knowledge_source": self.mapping.knowledge_source,
            "knowledge_level": "logical_entailment", "agent_type": "data_analysis_pipeline",
            f"{p}:gap_type": gap_local, f"{p}:gap_reason": reason,
            f"{p}:detection_floor": _real(floor), f"{p}:n_required": blank(n_required),
            f"{p}:proposal": proposal, f"{p}:kill_condition": kill,
            f"{p}:withheld_from": withheld_from,
            # Whether the withheld row carried an effect size at all. A gap file that does not say
            # this cannot be audited: "a number that failed to clear its bar" and "a categorical
            # claim that failed a precondition" are different refusals, and only the first can be
            # expected to carry a floor.
            f"{p}:was_measurement": ("" if was_measurement is None
                                     else str(bool(was_measurement)).lower()),
        }}
        # ONE IDENTITY, ONE ROW. Two evidence rows can state the same absence -- two measurements of
        # the same gene under the same floor, from sources the gap file does not carry -- and that is
        # one gap said twice, so the second is dropped and counted. Two rows that share an identity
        # and DIFFER are a modelling error: something distinguishes them that the id does not see, so
        # the export refuses and names the field that differs rather than picking a winner.
        first = self._gap_ids.get(gid)
        if first is not None:
            differing = sorted(k for k in set(first) | set(row_out)
                               if k != "id" and str(first.get(k, "")) != str(row_out.get(k, "")))
            if not differing:
                self.merged_gaps += 1
                return
            raise ValueError(
                f"two gaps share the identity {gid} but differ on {differing[:4]}: "
                f"subject={subj!r} gap_type={gap_local!r} context={common.get(f'{p}:context')!r} "
                f"withheld_from={withheld_from!r}. Declare the column that tells them apart in "
                f"Mapping.gap_identity_columns, or emit one gap.")
        self._gap_ids[gid] = row_out
        self.gaps.append(row_out)

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
