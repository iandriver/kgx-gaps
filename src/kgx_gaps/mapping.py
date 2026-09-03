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
    """How one source edge type becomes an association."""
    predicate: str
    category: str = "biolink:Association"
    knowledge_level: str = "statistical_association"
    agent_type: str = "data_analysis_pipeline"
    subject_aspect_qualifier: str = ""
    #: Emit a direction qualifier from the sign of the effect.
    directional: bool = False


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

    def validate(self) -> list[str]:
        problems = []
        for etype, r in self.rules.items():
            if r.knowledge_level not in KNOWLEDGE_LEVELS:
                problems.append(f"{etype}: knowledge_level {r.knowledge_level!r} is not in BioLink's enum")
            if r.agent_type not in AGENT_TYPES:
                problems.append(f"{etype}: agent_type {r.agent_type!r} is not in BioLink's enum")
            problems += [f"{etype}: predicate {r.predicate!r} is minted but unregistered (SPEC.md §2)"
                         for _ in self.unregistered([r.predicate])]
        return problems
