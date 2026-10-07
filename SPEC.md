# The Detection Floor Convention

**A normative specification for representing, in a BioLink/KGX knowledge graph, what a study looked
for and could not see.**

Version 0.6.0 · Apache-2.0

---

## 0. The problem

A biomedical knowledge graph records associations. It has no way to record that an association was
*measured for and not found*, and no way to record *how large an effect would have had to be* before
the measurement could have seen it.

So three different situations arrive at a consumer as the same thing — an absent edge:

| what happened | what the graph shows |
|---|---|
| the pair was never examined | no edge |
| examined, effect well below the detection threshold, so genuinely bounded | no edge |
| examined in a cohort far too small to see anything | no edge |

Only the second is evidence. The third is the absence of evidence, and it is routinely read as the
second. BioLink's `negated` slot does not fix this: `negated: true` asserts the association **is
false**, which is a stronger claim than any of the three and is wrong for all of them.

The consequence is asymmetric and it favours the confident. A graph built this way accumulates
everything a well-powered study found and silently discards the shape of what nobody could see.

## 1. Scope

This specification defines:

- **§2** three normative rules for emitting BioLink associations from measured evidence,
- **§3** a `Gap` record for the rows the rules refuse,
- **§4** the serialisation of both in KGX TSV,
- **§5** a conformance suite any implementation can be checked against.

It does **not** define how detection floors are computed. That is a domain question — a minimum
detectable effect at a stated power, a limit of quantitation, a coverage threshold. The specification
requires only that a floor exists, is on the same scale as the effect, and travels with the row.

The key words MUST, MUST NOT, SHOULD and MAY are to be interpreted as in RFC 2119.

## 2. The three rules

### Rule 1 — No assertion without detection

An evidence row that carries **both** an effect size and a detection floor is a *measurement*. A
measurement whose effect does not exceed its floor MUST NOT be emitted as a BioLink association. It
MUST be emitted as a `Gap` (§3) recording the predicate it was withheld from and the floor that
bounds it.

An evidence row carrying an effect size but **no** detection floor MUST NOT be emitted as an
association either. An effect size is a measurement by definition; what is absent is the bound, and
without it the estimate cannot be distinguished from one the study had no power to see. Such a row
MUST become a `Gap` stating that no floor was computed.

An evidence row carrying no effect size is **not a measurement** — a tractability call, a
classification, a curated fact. Rule 1 does not apply to it, and it MUST NOT be diverted to a gap on
the grounds that `detected` is unset. *Never-a-measurement and measured-and-under-floor are different
states, and conflating them is the failure this rule exists to prevent.*

Implementations MUST NOT use `negated: true` to express an under-floor result.

**A floor is not the only thing that can withhold an assertion.** A row may also fail a stated
*precondition* — an invalid instrument, a failed assay control, a QC flag — and such a row MUST also
become a `Gap` naming the predicate withheld and stating the precondition as its reason. It MUST NOT
be downgraded to a weaker predicate instead: a weaker assertion hides that a stronger one was
considered and refused, where a gap records it. The reason MUST be the precondition, not the floor,
even when the row also happens to sit under its floor.

### Rule 2 — Mint only what BioLink lacks, and declare it

Every emitted predicate MUST be either a BioLink Model predicate, or a minted predicate in an
implementation-defined CURIE prefix that is registered with a written rationale stating **which
BioLink term was considered and why it overstates or understates the claim**.

An implementation MUST refuse to emit a minted predicate that carries no registered rationale.

> Two predicates are minted by the reference implementation. `evidence_missing_for`, because BioLink
> cannot say "looked for, not found, and here is what would have been visible" — `negated` is a
> different claim and the power number has no slot at all. `tractable_for`, because
> `biolink:target_for` asserts the gene **is** a therapeutic target, where a tractability assessment
> says only that it could become one.

### Rule 3 — Ground or flag, never guess

Every node identifier MUST be a CURIE. An identifier that could not be resolved to an ontology MUST
be emitted under the implementation's own prefix carrying `id_grounded: false`, and MUST NOT be
dropped.

An implementation MUST NOT assign an ontology CURIE by taking the top hit of a fuzzy text search.
Where a mapping table is hand-written, it SHOULD be cross-checked against the ontology — the table
being the assertion and the ontology the second opinion — and each entry's label MUST match the
ontology term's name or one of its listed synonyms.

> Resolving the label `microglia` against the Cell Ontology returns `CL:4307132`, *microglial cell
> (Mmus)* — a **mouse** term — ahead of `CL:0000129`, *microglial cell*. On a human atlas, top-hit
> grounding changes the species without erroring.

## 3. The Gap record

A `Gap` is a first-class row, not the absence of one. It MUST carry:

| field | meaning |
|---|---|
| `subject` | the entity examined, as a CURIE |
| `object` | the gap type, as a CURIE |
| `predicate` | a minted predicate (§2) — no BioLink term means this |
| `gap_reason` | free text: what was looked for and why nothing was asserted |

and SHOULD carry, where the producer knows them:

| field | meaning |
|---|---|
| `withheld_from` | the predicate Rule 1 refused, empty if the row was always a gap |
| `withheld_object` | the object of the assertion Rule 1 refused, as a CURIE, empty if the row was always a gap |
| `detection_floor` | the effect size that would have been visible |
| `n_required` | the sample size at which the measurement would become decisive |
| `proposal` | the measurement that closes it |
| `kill_condition` | the observation that would retire the gap unresolved |

`withheld_from` is what makes a gap auditable: it names the assertion a less careful pipeline would
have made from the same row.

`withheld_object` is the other half of that assertion. A gap runs from its subject to the gap type,
so the row by itself says which predicate was refused and not what the refused assertion was about.
Where it is written it MUST be the CURIE the association would have carried as its object, resolved
under Rule 3 like any other node, and that node MUST be in `nodes.tsv` (C7). A row that was always a
gap had no assertion withheld from it and leaves the field empty.

### The id names the absence, not the row

A gap's `id` MUST be derived from what the gap is about, and MUST NOT encode the position of the row
in the file. Identity is the subject, the gap type, the context, the predicate withheld, the object
it was withheld about where the producer declares that (below), and any producer-specific column
the producer declares as identifying. What is *measured about* the gap is excluded: the floor,
`n_required`, the reason text and the proposal all move while the gap stays the same gap.

The rule exists because a gap outlives the build that emitted it. A closure, a citation or a memory
record pointing at `…:g000123` is worthless if the next build renumbers, and numbering by emission
order renumbers every gap after any gap that closes. This implementation hashes the identity fields;
any scheme with the same property satisfies the rule.

Two gaps with one identity are one gap written twice, and a producer MUST refuse rather than emit
both. A single file can be checked for duplicate ids (C6); stability across builds cannot be seen
from one file, so it is the producer's own test.

**The withheld object is part of the identity where the producer declares it.** A producer that
withholds at most one assertion per subject and context need not: the object tells none of its gaps
apart. A producer whose withheld rows come one per (subject, object) pair MUST declare
`withheld_object` as identifying. One gene withheld from `biolink:gene_associated_with_condition`
for two diseases is two gaps, and an identity that leaves the object out gives them one id. Seen
2026-10-07 in a graph of gene-to-disease associations: two such rows that differed in another field
were refused as one identity, and two that agreed in every field written were merged into one gap,
so the export held one withheld claim where the source had two.

It is not part of every gap's identity because an id already issued must not move. Declared, it is
part of the identity of a withheld row only. A gap that was always a gap has no withheld object, and
its id is the same whether or not the producer declares one.

`withheld_object` is written whether or not it is declared, so two rows withheld for different
objects always differ in a field. Under an identity that leaves the object out they are therefore
refused, and can no longer be merged.

## 4. Serialisation

Three KGX TSV files: `nodes.tsv`, `edges.tsv`, `gaps.tsv`.

Associations in `edges.tsv` MUST carry the BioLink provenance slots `primary_knowledge_source`,
`knowledge_level` and `agent_type`. Fields this specification adds beyond BioLink MUST be namespaced
by the implementation's prefix, so a strict BioLink consumer can drop them without silently
reinterpreting anything.

Implementations MUST write the empty string, never the literal `nan` or `None`, for an absent value.

A value that is present MUST be written as the producer supplied it, and MUST NOT be reformatted
because it happens to parse as a number. A year is `2018`, not `2018.0`, and an identifier made of
digits keeps its leading zeros: `0012` written as `12.0` is a different identifier, and nothing
errors. The effect size, its standard error and the detection floor are the exception. They are
real-valued by definition, so an implementation MAY read `"0.22"` and `0.22` as one number. No check
on a file can see what the producer supplied, so this is the producer's own test.

**Evidence-derived rows MUST be distinguishable from structural ones.** Every association and gap
derived from an input evidence row MUST carry a non-empty `{prefix}:source_edge_type` naming the
input type it came from. A graph may also contain **structural** edges that no evidence row produced
— an ontology hierarchy, a link from a disease to the axis its severity is scored on — and those MUST
leave that field empty. Without the distinction, conservation (C8) cannot be checked: an
implementation that dropped one evidence row and added one scaffold edge would balance.

**A conforming consumer that keeps only BioLink-native slots loses the floors.** That is why §3
requires them on the gap rows too: the gap file is the copy that survives a lossy reader.

### The id names the claim, not the row

An association's `id` MUST be derived from what the association claims, and MUST NOT encode the
position of the row in the file. Identity is the subject, predicate and object, the three qualifiers
(`subject_aspect_qualifier`, `object_direction_qualifier`, `anatomical_context_qualifier`), the
context, the `primary_knowledge_source`, the source edge type, and any producer-specific column the
producer declares as identifying. What is *measured about* the claim is excluded: the effect size,
its standard error, the floor and `detected` all move while the claim stays the same claim. The
direction qualifier is part of the claim. An edge that said `decreased` and comes to say `increased`
is a different edge and takes a different id, so a citation of the first never comes to point at the
second.

This is §3's rule for gaps, and associations need it for a second reason that a gap file did not
show. A producer usually writes more than one export, and a counter starts at zero in each. Three
exports from the atlas this implementation was extracted from, concatenated, held 2,814 association
rows under 2,021 ids. Each of the 434 ids used more than once named a different subject and object
in each file, and all three exports passed C6. Ids derived from content are unique across a
producer's exports exactly when the claims are.

Two rows with one identity that agree in every field are one claim said twice. A producer MUST write
it once and MUST count it, because without the count C8 cannot tell a merged row from a dropped one.
Two rows that share an identity and differ are two measurements the id cannot tell apart, and a
producer MUST refuse rather than keep either.

A structural edge takes its id the same way; its empty source edge type is one of the fields hashed.

An id MUST NOT be used by both an association and a gap: the two files become edges of one graph when
they are loaded.

## 5. Conformance

An implementation is conformant if, for any input, its output satisfies:

| # | check |
|---|---|
| C1 | no association row carries an effect size that fails to exceed its floor |
| C2 | no row uses `negated` to express an under-floor result |
| C3 | every non-BioLink predicate has a registered rationale |
| C4 | every withheld row names a `withheld_from` predicate, and carries its floor unless it was withheld precisely because no floor exists |
| C5 | every gap carries a non-empty `gap_reason` |
| C6 | every node id is a CURIE; ungrounded ids use the implementation prefix and are flagged; no association id and no gap id is used twice, within either file or across the two |
| C7 | every id referenced by an edge or gap exists in `nodes.tsv`, a gap's `withheld_object` included |
| C8 | evidence-derived associations + gaps + rows merged as exact repeats of one already written account for every input row — nothing is silently dropped. Structural edges (empty `source_edge_type`) are excluded from the count |
| C9 | no field is serialised as `nan`, `None` or `NaN` |
| C10 | rows with no effect size are not diverted to gaps by Rule 1 |

`kgx_gaps.conformance.check(nodes, edges, gaps, mapping)` returns these as pass/fail with the
offending rows. It runs against **any** KGX triple, not only output from this implementation.

Two things no check on one export can show, and which are therefore the producer's own tests: that
an id is stable across builds, and that the ids of two exports meant for one graph do not collide.
