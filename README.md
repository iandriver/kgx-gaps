# kgx-gaps

**A BioLink/KGX export that refuses to assert what it could not detect, and stores the gap instead.**

[SPEC.md](SPEC.md) is the specification. This package is its reference implementation and a
conformance suite you can run against *any* KGX export, including one this code did not produce.

---

## The problem

A knowledge graph records associations. It has no way to record that something was measured for and
not found, or how large an effect would have had to be before the measurement could have seen it. So
three different situations reach a consumer as the same absent edge:

- the pair was never examined
- examined, effect well under the detection threshold, genuinely bounded
- examined in a cohort far too small to see anything

Only the second is evidence. BioLink's `negated` slot does not help: it asserts the association **is
false**, which is wrong for all three.

The result is asymmetric and it favours the confident. Graphs accumulate what well-powered studies
found and silently discard the shape of what nobody could see.

## The three rules

**1. No assertion without detection.** A row carrying an effect size and a floor is a *measurement*;
if the effect does not clear the floor it becomes a `Gap` naming the predicate it was withheld from.
A row carrying no effect size was never a measurement and is not diverted — never-measured and
measured-and-under-floor are different states.

**2. Mint only what BioLink lacks, and declare it.** Any non-BioLink predicate must be registered
with a rationale naming *which BioLink term was considered and why it overstates the claim*. The
rationale is a required constructor argument, so it cannot be deferred.

**3. Ground or flag, never guess.** Unresolvable identifiers are emitted under your own prefix with
`id_grounded: false`, never dropped and never assigned an ontology CURIE by fuzzy top-1 match.

> Resolving the label `microglia` against the Cell Ontology returns `CL:4307132`, *microglial cell
> (Mmus)* — a **mouse** term — ahead of `CL:0000129`. On a human atlas, top-hit grounding changes the
> species without erroring.

## Install

```bash
pip install kgx-gaps
```

## Check a graph you already have

The conformance suite needs no adoption. Point it at three TSVs:

```bash
kgx-gaps check path/to/kgx/ --input-rows 912
```

```
  C1  PASS  no association is under its own detection floor
  C2  PASS  `negated` is not used to express an under-floor result
  C3  SKIP  every non-BioLink predicate has a registered rationale — no Mapping supplied
  ...
  9/10 checks passed, 1 SKIPPED (not a pass)
```

**A skip is not a pass.** C3 needs your `Mapping` and C8 an input row count; neither can be inferred
from the TSVs, so the report says so rather than quietly scoring them green.

## Export a graph

```python
from kgx_gaps import Mapping, Rule, export, conformance

m = Mapping(prefix="mykg", knowledge_source="infores:mykg")
m.rules["SLOPE"] = Rule("biolink:correlated_with", subject_aspect_qualifier="expression",
                        directional=True)
m.rules["TRACT"] = Rule(m.mint(
    "tractable_for",
    biolink_considered="biolink:target_for",
    rationale="`target_for` asserts the gene IS a target; tractability says only that it could be."))

ex = export(evidence_df, m)          # columns documented in EvidenceColumns
ex.write("kgx/")
print(conformance.check(*ex.frames(), m, n_input_rows=len(evidence_df), n_merged=ex.merged))
```

Input is a table of evidence rows — `type`, `subject`, `object`, and optionally `effect`, `se`,
`floor`, `detected`, `context`. Rename via `EvidenceColumns` rather than reshaping your frame. How
you compute a floor is your domain's business; the spec only requires that one exists, is on the same
scale as the effect, and travels with the row.

**Ids name the claim, not the row.** An association's id is a hash of what it asserts (the triple,
its qualifiers, the context, the knowledge source, the source edge type) and a gap's of what is
absent. Neither depends on row order, so an id survives a rebuild, and exports written separately
can be loaded into one graph without two edges sharing an id. If your edges are further told apart
by a column of your own, such as a study, declare it in `Mapping.edge_identity_columns`. A row that
repeats another exactly is written once and counted in `ex.merged`; two rows with one identity that
differ are refused.

## Worked example

[`examples/geoatlas-ad/`](examples/geoatlas-ad/) is a real Alzheimer's evidence graph — 912 rows from
severity slopes, colocalisation, Mendelian randomisation and druggability — with the `Mapping` that
produced it. What the rules do to it:

| | |
|---|---|
| input evidence rows | **912** |
| BioLink associations, evidence-derived | **433** (+1 structural, so `edges.tsv` has 434 rows) |
| gap records | **479** |
| withheld from an assertion for want of detection | **42** |
| of 15 colocalisation rows, withheld | **14** |
| of 29 Mendelian-randomisation rows, withheld | **23** |

Half the graph is what it could not see. A pipeline without Rule 1 would have emitted all 912 rows as
associations, and the 42 withheld ones would have been indistinguishable from the rest.

```bash
python examples/geoatlas-ad/mapping.py     # 10/10 checks passed
```

## Status

v0.5.0, alpha. The spec is versioned separately from the code and will change; C8 already has one
amendment, made because running the suite against a real graph found the check forbade legitimate
structural edges. Amendments are recorded in the git history with the graph that prompted them.

Issues and disagreements with the spec are welcome — particularly from anyone who thinks a rule is
wrong rather than merely inconvenient.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
