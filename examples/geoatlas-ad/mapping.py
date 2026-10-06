"""The Mapping behind `kgx/` — a real Alzheimer's evidence graph, 908 rows in.

This is what a project's domain layer looks like: five source edge types, two minted predicates, and
a rationale for each naming the BioLink term it rejected. Nothing else here is domain-specific.

Run:  python examples/geoatlas-ad/mapping.py
"""
from pathlib import Path

from kgx_gaps import Mapping, Rule, conformance

INPUT_ROWS = 908          # rows in the source evidence table
MERGED = 4                # of those, exact repeats of a gap already written (Exporter.merged)


def geoatlas() -> Mapping:
    # An edge id is a hash of what the edge claims. The atlas adds what tells two colocalisations of
    # one gene in one tissue apart: the molecular trait, the QTL study and the sign.
    m = Mapping(prefix="geoatlas", knowledge_source="infores:geoatlas",
                edge_identity_columns=["geoatlas:coloc_qtl_type", "geoatlas:coloc_study",
                                       "geoatlas:coloc_sign"])

    # A severity slope: expression tracks the disease severity axis. `correlated_with` is exactly
    # "quantitative measurement shows correlation", which is all an observational slope claims --
    # deliberately not `causes` or `contributes_to`, because a gene up in disease is as often
    # compensatory as causative and the two imply opposite drugs.
    m.rules["RESPONDS_TO"] = Rule("biolink:correlated_with", "biolink:GeneToDiseaseAssociation",
                                  subject_aspect_qualifier="expression", directional=True)

    # The same correlation, where decomposition showed it is carried by cell-type ABUNDANCE rather
    # than by regulation inside the cell. BioLink can say this natively.
    m.rules["COMPOSITION_SHIFT"] = Rule("biolink:correlated_with", "biolink:GeneToDiseaseAssociation",
                                        subject_aspect_qualifier="abundance", directional=True)

    # Colocalisation: the same causal variant drives the eQTL and the GWAS signal.
    m.rules["GENETIC_SUPPORT"] = Rule("biolink:gene_associated_with_condition",
                                      "biolink:GeneToDiseaseAssociation")

    # Mendelian randomisation. `affects` is a causal claim, earned only when the instrument is valid
    # AND the estimate clears its floor; everything else goes to the gap file under Rule 1.
    m.rules["QTL"] = Rule("biolink:affects", "biolink:GeneToDiseaseAssociation", directional=True)

    m.rules["DRUGGABLE"] = Rule("biolink:target_for", "biolink:GeneToDiseaseAssociation",
                                knowledge_level="knowledge_assertion")

    m.mint("tractable_for",
           biolink_considered="biolink:target_for",
           rationale=("`target_for` means the gene IS a therapeutic target for the disease. A "
                      "tractability assessment -- a ligandable pocket, a surface epitope -- says "
                      "only that it could become one. Genes carrying a real clinical phase get "
                      "biolink:target_for; the rest get this."))

    m.gap_types.add("MISSING")
    return m


if __name__ == "__main__":
    m = geoatlas()
    assert not m.validate(), m.validate()
    rep = conformance.check_dir(Path(__file__).parent / "kgx", m, n_input_rows=INPUT_ROWS,
                                n_merged=MERGED)
    print(f"\n  geoatlas AD graph — {INPUT_ROWS} evidence rows\n")
    print(rep)
    raise SystemExit(0 if rep.passed else 1)
