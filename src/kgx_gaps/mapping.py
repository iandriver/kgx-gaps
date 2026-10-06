"""Predicate mapping, and the registry that makes Rule 2 enforceable.

A `Mapping` is the whole of an implementation's domain knowledge: which BioLink predicate each of its
own edge types becomes, and — for the claims BioLink cannot express — which predicates it mints and
why. Everything else in this package is domain-free.

The registry is not documentation. `Mapping.validate()` refuses a mapping that mints a predicate
without a rationale, and the conformance suite (C3) refuses an *export* containing one. SPEC.md §2
asks for a written reason naming the BioLink term considered and why it was wrong; that reason is a
required constructor argument, so it cannot be left for later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

BIOLINK = "biolink:"

#: BioLink KnowledgeLevelEnum, as of BioLink Model 4.x.
KNOWLEDGE_LEVELS = {
    "knowledge_assertion", "logical_entailment", "prediction", "statistical_association",
    "observation", "not_provided",
}
#: BioLink AgentTypeEnum.
AGENT_TYPES = {
    "manual_agent", "automated_agent", "data_analysis_pipeline", "computational_model",
    "text_mining_agent", "image_processing_agent", "manual_validation_of_automated_agent",
    "not_provided",
}


@dataclass(frozen=True)
class Rule:
    """How one source edge type becomes an association.

    `predicate` may be a callable `(row) -> str` when the claim depends on the row rather than only
    on its type — a gene with a real clinical phase earns `biolink:target_for` where the same source
    type otherwise earns a weaker minted predicate.

    `precondition` is Rule 1's other half. A detection floor is not the only thing that can make a
    row unable to support its assertion: an invalid instrument, a failed assay control, a QC flag.
    Return the empty string to proceed, or a reason to withhold. A row withheld this way becomes a
    Gap that still names `withheld_from`, so it is auditable exactly like an under-floor row — the
    alternative, silently emitting a weaker predicate instead, hides that a stronger claim was
    considered and refused.
    """
    predicate: str | Callable[[dict], str]
    category: str = "biolink:Association"
    knowledge_level: str = "statistical_association"
    agent_type: str = "data_analysis_pipeline"
    subject_aspect_qualifier: str = ""
    #: Emit a direction qualifier from the sign of the effect.
    directional: bool = False
    #: (row) -> reason to withhold, or "" to proceed.
    precondition: Callable[[dict], str] | None = None

    def predicate_for(self, row: dict) -> str:
        return self.predicate(row) if callable(self.predicate) else self.predicate

    def predicates(self) -> set[str]:
        """Every predicate this rule can emit — needed so Rule 2 can be checked before any export.

        A callable cannot be enumerated, so it must declare what it may return via
        `Mapping.declare_predicates()`; otherwise a minted predicate could reach the output without
        ever passing validate().
        """
        return set() if callable(self.predicate) else {self.predicate}


@dataclass(frozen=True)
class Minted:
    """A predicate this implementation invents, and the reason BioLink could not be used.

    `biolink_considered` is required and must name a real BioLink term (or the string "none", when
    nothing in the model is even close). Recording which term was rejected is what lets a reviewer
    disagree with the decision; a bare "BioLink has nothing" cannot be argued with.
    """
    predicate: str
    biolink_considered: str
    rationale: str

    def __post_init__(self):
        if not self.rationale.strip():
            raise ValueError(f"minted predicate {self.predicate!r} has no rationale (SPEC.md §2)")
        if not self.biolink_considered.strip():
            raise ValueError(
                f"minted predicate {self.predicate!r} must name the BioLink term considered, "
                f"or 'none' (SPEC.md §2)")


@dataclass
class Mapping:
    """The domain layer: edge-type rules, minted predicates, and the implementation's own prefix."""
    prefix: str
    knowledge_source: str
    rules: dict[str, Rule] = field(default_factory=dict)
    minted: dict[str, Minted] = field(default_factory=dict)
    #: Source edge types that are already gaps in the input vocabulary.
    gap_types: set[str] = field(default_factory=set)
    #: Local name of the predicate every Gap uses. Registered automatically — a gap predicate that
    #: was not minted-and-declared would violate Rule 2 on the very row the spec exists for.
    gap_predicate_local: str = "evidence_missing_for"
    #: etype -> predicates a callable rule may emit (see declare_predicates).
    declared: dict[str, set] = field(default_factory=dict)
    #: Extra gap columns that form part of a gap's IDENTITY, named with the prefix as they appear in
    #: the file. A gap id is derived from what the gap is about -- subject, gap type, context, which
    #: row was withheld -- so the same absence keeps the same id across builds and a closure recorded
    #: against it still points at it. A producer whose gaps are further distinguished by one of its
    #: own columns (a cell type, a dataset) declares those columns here. Never declare one that moves
    #: while the gap stays the same gap: a floor, a count, a reason string, a build stamp.
    gap_identity_columns: list[str] = field(default_factory=list)
    #: The same for associations. An edge id is derived from what the edge CLAIMS -- subject,
    #: predicate, object, the qualifiers, the context, the knowledge source and the source edge type
    #: -- so two exports of one producer never spend the same id on different edges, and an id
    #: survives a rebuild. A producer whose edges are further told apart by one of its own columns (a
    #: study, an assay type) declares those columns here. Never declare one that is measured ABOUT
    #: the claim: an effect size, a posterior, a standard error, a build stamp.
    edge_identity_columns: list[str] = field(default_factory=list)
    #: Gap type -> the name its node carries. A gap node's id is the gap type, so its name has to be
    #: a function of the gap type too; without an entry the name is the type with underscores as
    #: spaces. Naming the node after the export has run instead makes the name depend on whether
    #: that export happened to contain the gap type, and two exports then disagree about one id.
    gap_names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.mint(
            self.gap_predicate_local,
            biolink_considered="biolink:negated (the `negated` association slot)",
            rationale=(
                "BioLink cannot say 'this was looked for, was not found, and here is the effect size "
                "that would have been visible'. `negated: true` asserts the association IS FALSE, "
                "which is a strictly stronger claim than not-detected, and the power number that "
                "makes the distinction has no BioLink slot at all."),
        )

    @property
    def gap_predicate(self) -> str:
        return f"{self.prefix}:{self.gap_predicate_local}"

    def mint(self, local: str, *, biolink_considered: str, rationale: str) -> str:
        """Register a minted predicate and return its CURIE."""
        curie = f"{self.prefix}:{local}"
        self.minted[curie] = Minted(curie, biolink_considered, rationale)
        return curie

    def unregistered(self, predicates) -> set[str]:
        """Predicates that are neither BioLink's nor registered here — Rule 2 violations."""
        return {p for p in predicates
                if p and not p.startswith(BIOLINK) and p not in self.minted}

    def declare_predicates(self, etype: str, *predicates: str) -> None:
        """Declare what a callable `Rule.predicate` may return, so Rule 2 is checkable up front."""
        self.declared.setdefault(etype, set()).update(predicates)

    def validate(self) -> list[str]:
        problems = []
        for etype, r in self.rules.items():
            if r.knowledge_level not in KNOWLEDGE_LEVELS:
                problems.append(f"{etype}: knowledge_level {r.knowledge_level!r} is not in BioLink's enum")
            if r.agent_type not in AGENT_TYPES:
                problems.append(f"{etype}: agent_type {r.agent_type!r} is not in BioLink's enum")
            emits = r.predicates() | self.declared.get(etype, set())
            if callable(r.predicate) and not self.declared.get(etype):
                problems.append(
                    f"{etype}: predicate is a callable but nothing was declared via "
                    f"declare_predicates({etype!r}, ...) — its output cannot be checked against "
                    f"Rule 2 before the export runs")
            problems += [f"{etype}: predicate {p!r} is minted but unregistered (SPEC.md §2)"
                         for p in self.unregistered(emits)]
        return problems
