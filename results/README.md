# Results and evidence

`generated/` contains the retained claim-bound provenance bundles produced by
`scripts/reproduce.py` and checked by `scripts/validate_claims.py`.

Each bundle records the source revision, full command, dataset checksums, random
seeds, compiler and dependency versions, CPU and thread settings, and the captured
stdout/stderr of the run. `claims.toml` binds each paper claim to the bundles that
evidence it, and its `status` field states how far that binding has been
re-validated against the current protocol. `claims.toml` defines the enabled claim
set, and `PAPER-MAP.md` maps the compiled paper directly to those bundles. The root
of this directory contains policy and mapping files only; duplicate convenience
reports are intentionally omitted.

## About `artifact_commit`

Each `manifest.json` carries an `artifact_commit` field naming the revision of the
authoring tree the run was produced from. Those hashes belong to the author's
working history and are **not resolvable in this repository** — do not expect
`git show` to find them.

They are provenance labels, and what binds a bundle to the code that produced it is
the recorded *computation hash* over `src/`, `scripts/`, the dataset and baseline
manifests, and the lock files. Be precise about which half of that is checkable
where, because the two are easy to conflate:

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

So a NOTE in this repository is the expected outcome, not a degraded one — but read
it as "the commit binding was verified upstream", not as "this run verified it".
