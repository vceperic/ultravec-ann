# Results and evidence

`generated/` contains the retained claim-bound provenance bundles produced by
`scripts/reproduce.py` and checked by `scripts/validate_claims.py`. They are the
evidence for the paper's results: the highest mean flat MIPS Recall@10 in eleven of
twelve corpus-rate cells and the lowest reconstruction error in all twelve, a 4.2 to
7.7 point lead over the pinned E-RaBitQ library at every rate on SIFT, a million-vector
IVF index that passes the best Recall@10 of both E-RaBitQ and trained Faiss IVFPQ from
four probes at 3.5 times E-RaBitQ's throughput and peaks within 2.7 points of trained
ScaNN, a controlled HNSW study in which UltraVec's lowest-`ef` recall exceeds
E-RaBitQ's highest and codes-only navigation runs at a third of the resident memory
with 14% higher throughput, cold start separated above fitted 256-centroid PQ through
128 SIFT calibration vectors, and no refit under a covariate shift that costs
transferred PQ 25.9 to 41.5% of its recall. The bundles also back a fitted PCA head that exceeds the pinned E-RaBitQ library by 7.8 to 27.9 points for a ten-byte overhead, and trellis memory that raises direction fidelity from 0.876 to 0.970 at a fixed emitted rate. `PAPER-MAP.md` names the bundle behind each one.

Each bundle records the source revision, full command, dataset checksums, random
seeds, compiler and dependency versions, CPU and thread settings, and the captured
stdout/stderr of the run. `claims.toml` binds each paper claim to the bundles that
evidence it, and its `status` field, `validated-clean-evidence`, is enforced by
`validate_claims.py`. `claims.toml` defines the enabled claim
set, and `PAPER-MAP.md` maps the compiled paper directly to those bundles. The root
of this directory contains policy and mapping files only; duplicate convenience
reports are intentionally omitted.

## About `artifact_commit`

Each `manifest.json` carries an `artifact_commit` field naming the revision of the
authoring tree the run was produced from. Those hashes name revisions in the
author's working history, which this repository does not carry, so `git show` does
not resolve them here.

They are provenance labels, and what binds a bundle to the code that produced it is
the recorded *computation hash* over `src/`, `scripts/`, the dataset and baseline
manifests, and the lock files. The two halves are checked in different places:

- **In this repository**, `validate_claims.py` cannot resolve the authoring commit,
  so it retains the recorded hash *structurally* and reports a NOTE. What it does
  verify is everything that travels with the payload: the bundle's own consistency,
  the recorded dataset hashes against the files if present, and the baseline pins,
  `requirements.txt` and the three manifests against the current tree. A bundle
  whose inputs or pins no longer match what ships here is caught; a bundle whose
  *code* drifted is not, because the code it named is not here to compare against.
- **In the authoring monorepo**, `validate_claims.py --strict` recomputes the
  computation hash at the recorded commit and requires a match. That is the check
  that binds evidence to code, it needs the authoring history and the staged
  corpora, and it is gated there by `make check-provenance` before release.

A NOTE in this repository is therefore the expected outcome.
