# Retained clean-campaign evidence

These bundles were produced by completed clean evidence runs. Each experiment
manifest records the computation-tree hash, dataset hashes, baseline revisions,
Python lock, toolchain, CPU, thread count, controlled seed, command, and pass
status. Bundles need not share an authoring commit; each records its own clean
computation hash and commit.

- `smoke-synthetic/` is a synthetic self-test, not a `full`-tier step. It backs no
  manuscript claim. It was re-run on 2026-08-28 during the history re-stamp --- the
  only bundle whose computation tree no commit on the new history carried, so it was
  regenerated rather than re-pointed --- and now records a normal provenance block
  like every other bundle.
- `mechanism-msweep/` is the fixed-emission-rate direction-fidelity memory sweep. It
  now runs through `M=16`, the memory the flat comparison reports, so the mechanism
  table covers the operating point instead of stopping four steps below it. Its
  summary header states the number of queries the diagnostic *samples* (100) against
  a seeded 1,000-vector database subsample, which is what the manuscript's table
  caption states; the earlier bundle printed the number loaded (200) instead.
- `systems-scann-sift1m/` is a targeted seeded ScaNN reference. Its driver sets
  partitioning, asymmetric-hash clustering, and sampling seeds and verifies two
  independent builds return identical neighbors at every sweep setting.

Portable validation checks bundle structure, clean-run invariants, pins, locks,
declared hashes, controlled seeds, and pass status without requiring the authoring
repository's history or the large datasets. `validate_claims.py --strict` also
requires every dataset hash and a locally resolvable evidence commit.

Before publication, absolute copies of the artifact root in retained text output
were replaced with `.` and the randomly named directory used for the independent
official E-RaBitQ recheck was replaced with `<temporary-directory>`. This
release-only path normalization does not alter numerical output or any provenance
manifest. The recheck substitution is applied by
`scripts/prepare_official_rabitq.py` at the point of write and is enforced by
`scripts/test_bundle_hygiene.py`.

Run the enabled-claim gate from the repository root:

```sh
python3 scripts/validate_claims.py
```

Raw datasets, derived vectors, downloaded baseline source, wheels, binaries, and
indexes are excluded from the repository and must be recreated through the
checksum-enforcing workflow.
