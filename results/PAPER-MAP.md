# Paper → evidence map

Every result-bearing table and figure in the paper, and the bundle it is checked
against. The paper's `scripts/check_results.py` enforces this mapping cell by cell;
`scripts/make_figures.py` reads the same bundles to build the figures. The directory
root contains only mapping and policy files; both tools read claim-bound bundles
under `generated/`.

Rows are keyed by the manuscript's `\label`, which is stable; the number column
follows the compiled PDF and shifts whenever a table is inserted ahead of another.
`scripts/test_check_results.py` in the paper tree asserts this list matches the set
of gated tables exactly, and that each row's number matches the compiled PDF's own
numbering, because both had drifted: five tables behind at one point, and later a
number column reading 20 and 22 for tables the build numbered 19 and 20.

The bundle column names what the gate opens, which is not always what a reader
would guess: the flat tables moved to the `M=16` operating point (`flat16-*`) and
the systems tables to `systems16-*`. The superseded `common-reconstruction-*` and
`systems-hnsw-sift1m` runs are still read, as the `M=12` side of what the move to
`M=16` bought; only `systems-ivf-sift1m` is retained and uncited.

## Tables

| Table | Label | Subject | Bundle | File |
|---|---|---|---|---|
| 1 | `tab:complexity` | Asymptotic costs | — | derived in the text; nothing measured |
| 2 | `tab:datasets` | Public evaluation inputs | `datasets/manifest.toml`, `datasets/derived.toml` | hashes and byte lengths |
| 3 | `tab:common-r10` | Flat MIPS Recall@10, three corpora × four rates | `flat16-sift`, `flat16-gist`, `flat16-dbpedia` | `stdout.txt` |
| 4 | `tab:parity` | Implementation parity against the pinned library | `erab-parity` | `stdout.txt` |
| 5 | `tab:rotationvar` | Margin across rotation draws | `rotation16-gist`, `rotation16-dbpedia` | `stdout.txt` per-comparator blocks; the table reports the closest comparator, not the per-draw field maximum |
| 6 | `tab:ablation-components` | Core component ablations at `M=12` | `appendix-ablations` | `stdout.txt` §1–2b |
| 7 | `tab:tailbite` | Tail-biting against free start, three corpora, five draws | `tail-biting-sweep` | `stdout.txt` summary block |
| 8 | `tab:msweep` | Direction fidelity vs trellis memory | `mechanism-msweep` | `stdout.txt` |
| 9 | `tab:scale` | Byte-identical estimator-scale control | `mechanism-statistics` | `stdout.txt` §3; inputs in `mechanism-inputs` |
| 10 | `tab:coldstart` | Cold-start Recall@10 vs calibration size | `coldstart-sift`, `coldstart-glove` | `stdout.txt` |
| 11 | `tab:pqwidth` | Crossover against PQ factorization, SIFT | `coldstart-pq-width` | `stdout.txt`, SIFT block |
| 12 | `tab:pqwidth-glove` | Crossover against PQ factorization, GloVe | `coldstart-pq-width` | `stdout.txt`, GloVe block |
| 13 | `tab:centerk` | Sample efficiency of the centering mean | `center-sample-efficiency` | `stdout.txt` |
| 14 | `tab:official` | Official E-RaBitQ vs the fitted hybrid | `official-erabitq-bootstrap` | `stdout.txt`; inputs prepared by `official-rabitq-preparation` |
| 15 | `tab:warm` | Warm data-dependent boundary | `dehub-strong-baselines` | `stdout.txt` |
| 16 | `tab:ivf` | SIFT1M IVF recall / throughput / memory | `systems16-ivf-sift1m` | `stdout.txt` |
| 17 | `tab:hnsw` | Controlled SIFT1M HNSW at `ef=64` | `systems16-hnsw-sift1m` | `stdout.txt`, the `ef=64` rows |
| 18 | `tab:mature-systems` | Pinned Faiss and ScaNN reference points | `systems-faiss-sift1m`, `systems-scann-sift1m` | `stdout.txt` |
| 19 | `tab:common-r1r100` | Flat MIPS Recall@1 and Recall@100, paired cells | `flat16-sift`, `flat16-gist`, `flat16-dbpedia` | `stdout.txt`. Per-query rows were retained for the superseded `M=12` campaign only, so `common-reconstruction-*/per-query.csv` records that run rather than this table's |
| 20 | `tab:ratemem` | Trellis memory at one bit, within a byte tier | `rate-memory-sweep` | `stdout.txt` |
| 21 | `tab:encode-cost` | Encode throughput vs memory and rate | `encode-cost` | `stdout.txt` throughput grid; ratios below it |

## Figures

| Figure | Subject | Bundle |
|---|---|---|
| 1 | Cold-start sweep with six-calibration-draw intervals | `coldstart-sift`, `coldstart-glove` |
| 2 | Recall@10 against trellis memory per rate | `appendix-ablations` |

The manuscript prints two figures. An earlier three-panel flat-recall figure was
dropped for the page budget and its generator removed, so no figure reads the
superseded `common-reconstruction-*` bundles any more.

## Prose numbers with dedicated checks

| Claim | Bundle |
|---|---|
| Transfer loss under covariate shift (abstract, §1, §7.2, conclusion) | `shift-sift` |
| Fidelity sweep endpoints and span (abstract, §1, §6.1, conclusion) | `mechanism-msweep` |
| Residual estimator bias at the reported memory (§3.3) | `mechanism-msweep` |
| One-bit paired margins and the GIST separation interval (§5.1) | `flat16-*` |
| IVF recall, throughput, record and resident bytes quoted in §8.1 | `systems16-ivf-sift1m` |
| What state memory costs the graph (§10.6) | `systems16-hnsw-sift1m` against `systems-hnsw-sift1m` |
| Finer centering: per-granularity cells, margins and the global-mean cost (§7.4) | `centering-granularity` |
| Every table's declared trellis memory, checked against its own caption | the bundle behind each table above |
| One-million-vector ordering and per-rate margins (abstract, §6.2) | `flat1m-sift`, compared against `flat16-sift` |
| HNSW drop-in and codes-only figures quoted in §8.2 | `systems16-hnsw-sift1m` |
| Estimator-scale reductions (§6.3 and conclusion) | `mechanism-statistics` |
| Faiss and ScaNN operating points (§8.3) | `systems-{faiss,scann}-sift1m` |
| Beam-width approximation gap (§3.2 and Appendix B.1) | `appendix-ablations` |
| Rotation-round ablation (Appendix B) | `appendix-ablations` |
| Vector-trellis sweep, quoted as prose rather than tabulated (Appendix B) | `vector-trellis` |
| Shortlist stage-1 scorer A/B (Appendix B) | `shortlist-scorer-ab` |
| Residual control variate (Appendix B) | `residual-control-variate` |

## Bundles that back no manuscript claim

Seven of the thirty-nine retained bundles are not read by a manuscript gate. They
are kept because deleting evidence to tidy a directory is how a superseded result
becomes unreproducible, but nothing in the paper rests on them.

Inputs to a claim rather than a claim of their own:

- `official-rabitq-preparation` — builds and verifies the pinned upstream library;
  its outputs feed `tab:official`.
- `mechanism-inputs` — per-corpus diagnostics consumed by the estimator-scale
  control and the retained supplementary identifiability diagnostic.

Superseded as the reported operating point but still read:

- `common-reconstruction-sift`, `common-reconstruction-gist`,
  `common-reconstruction-embedding` — the earlier `M=12` flat comparison. The
  manuscript quotes what the move to `M=16` bought cell by cell, so a gate reads all
  three against the `flat16-*` bundles.
- `systems-hnsw-sift1m` — the `M=12` graph campaign, read for what state memory costs
  a graph walk (§10.6) against `systems16-hnsw-sift1m`.

Superseded and not read:

- `systems-ivf-sift1m` — replaced by `systems16-ivf-sift1m`.
- `lowrate-sift-m14`, `lowrate-gist-m14`, `lowrate-dbpedia-m14` — the one-bit
  sweep; only the two `lowrate-rotation-m14*` bundles are cited.

Neither, and never cited:

- `smoke-synthetic` — a synthetic self-test for the `smoke` tier.

## Checking it yourself

```sh
python3 scripts/validate_claims.py    # provenance, from this directory
```

This verifies each bundle's provenance metadata, input hashes, and recorded
computation identity.

Manuscript values are checked separately by a gate that lives with the LaTeX
source, which this artifact deliberately does not duplicate. That gate reads the
same bundles under `results/generated/`, cell by cell, so every number in the
paper is reproducible from what is published here: the table above names the
bundle behind each table and figure, and the bundles carry the raw output the
numbers were read from. Verifying a value needs nothing but this repository.
