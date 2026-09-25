"""kgx-gaps — a BioLink/KGX export that refuses to assert what it could not detect.

The specification is SPEC.md; this package is its reference implementation and its conformance suite.

    from kgx_gaps import Mapping, Rule, export, conformance

    m = Mapping(prefix="mykg", knowledge_source="infores:mykg")
    m.rules["SLOPE"] = Rule("biolink:correlated_with", directional=True)
    ex = export(evidence_frame, m)
    ex.write("kgx/")
    print(conformance.check(*ex.frames(), m, n_input_rows=len(evidence_frame)))
"""
from .mapping import Mapping, Minted, Rule, BIOLINK, KNOWLEDGE_LEVELS, AGENT_TYPES
from .export import Exporter, EvidenceColumns, export, blank
from . import conformance

__version__ = "0.4.0"
__all__ = ["Mapping", "Minted", "Rule", "Exporter", "EvidenceColumns", "export", "blank",
           "conformance", "BIOLINK", "KNOWLEDGE_LEVELS", "AGENT_TYPES", "__version__"]
