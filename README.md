# UltraVec: Trellis-Coded Quantization for Calibration-Free Low-Bit Vector Search

UltraVec is a calibration-free trellis-coded quantizer: a free-start Viterbi encoder
over a computed Gaussian codebook, scored by RaBitQ's rescaled inner-product estimator.
The plain codec has no fitted state, so it needs no training corpus and no calibration pass.

Against seven data-oblivious comparators on SIFT, GIST and DBpedia-OpenAI it has the
highest mean flat MIPS Recall@10 in eleven of twelve corpus-rate cells with paired
intervals excluding zero against every comparator, and the highest Recall@100 and lowest
reconstruction error in all twelve, 19 to 59% below the next-lowest codec's; no comparator separates above it in any cell. The twelfth cell, two-bit GIST, ties at the reported rotation draw and has the higher five-draw mean. On
a shared footing with the pinned upstream E-RaBitQ library (both codecs centered) it
leads by 4.2 to 7.7 points at every rate on SIFT, every paired interval excluding zero. On SIFT's full 1,000,000-vector base all 25 pairwise intervals
exclude zero, and on a million-vector IVF index the codes-only scan passes the best Recall@10 of both E-RaBitQ (`.4510`) and trained Faiss IVFPQ (`.4370`) from four probes (`.4810`), at 3.5 times the throughput E-RaBitQ needs for its best, peaks 14.8 points above E-RaBitQ (`.5990`), and comes within 2.7 points of trained ScaNN's best (`.6255`). In a controlled HNSW study it leads E-RaBitQ by 8.0 Recall@10 points, its lowest-`ef` recall (`.4680`) above E-RaBitQ's highest (`.4615`), and scoring that graph from the codes alone cuts its resident memory to a third (258 against 770 bytes/vector) at near-identical recall (`.5225` against `.5240`) and 14% higher throughput. From 2 to 16 state bits at a fixed emitted rate, direction fidelity rises from 0.876 to 0.970, and memories 9 through 16 serialize to the same bytes. With six calibration draws, plain UltraVec is separated above fitted 256-centroid PQ through 128 SIFT calibration vectors, and it needs no refit under a controlled covariate shift that costs transferred PQ 25.9 to 41.5% of its recall; with a fitted PCA head it exceeds the pinned E-RaBitQ library by 7.8 to 27.9 points for a ten-byte overhead. `results/PAPER-MAP.md` names the hash-validated evidence bundle behind each of
these.

This directory is the paper-specific artifact for that work. It contains the Rust
implementation, experiment drivers and retained evidence. The paper itself is
published separately; this repository is the code and evidence it cites.

The Rust dependency graph is locked, the smoke tier is self-contained, and every
evidence-producing step writes a provenance bundle under `results/generated/`.
Dataset checksums, dataset-license notes, and official-baseline revisions have been
curated. Sole ownership of the UltraVec implementation and artifact-specific scripts
is stated explicitly in `NOTICE`. Run `python3 scripts/reproduce.py doctor` before
attempting an experiment.

`results/claims.toml` binds each claim to its evidence bundles under the status
`validated-clean-evidence`, which `scripts/validate_claims.py` enforces. `results/PAPER-MAP.md` maps every table and figure
in the paper to its evidence bundle, providing a direct path to any reported value.
Each bundle records its authoring commit and aggregate computation hash.

## Reproduction tiers

| Tier | Command | Intended evidence | Current gate |
|---|---|---|---|
| Check | `python3 scripts/reproduce.py check` | Locked Rust build and unit tests | Self-contained after dependency cache is present |
| Smoke | `python3 scripts/reproduce.py smoke` | All-codec deterministic synthetic run and output parsing | Self-contained; not paper evidence |
| Core | `python3 scripts/reproduce.py core` | Flat SIFT recall and cold start on SIFT/GloVe | Requires public corpora and full Python environment |
| Systems | `python3 scripts/reproduce.py systems` | IVF/HNSW recall–QPS on SIFT1M | Requires SIFT1M and a controlled CPU/thread setup |
| Mechanism | `python3 scripts/reproduce.py mechanism` | Fidelity/MSE identifiability diagnostic and estimator control | Requires all four public corpora |
| Msweep | `python3 scripts/reproduce.py msweep` | Direction fidelity versus trellis memory at a fixed emission rate | Requires SIFT; evidence for the central mechanism claim |
| ScaNN | `python3 scripts/reproduce.py scann` | Seeded ScaNN reference point and its determinism check | Requires SIFT1M and the pinned ScaNN wheel |
| Prepare | `python3 scripts/reproduce.py prepare` | Verified inputs, pinned baselines, and upstream RaBitQ-Library/E-RaBitQ outputs | Requires a C++ toolchain, cmake and the staged corpora |
| Scale | `python3 scripts/reproduce.py flat1m` | The flat comparison on SIFT at one million base vectors | Requires SIFT1M; roughly 16 GB resident |
| Full | `python3 scripts/reproduce.py full` | Every ANN table and figure, including GIST and DBpedia | Runs `prepare` first, then every tier that produces a gated number |

## Associated paper

*UltraVec: Trellis-Coded Quantization for Calibration-Free Low-Bit Vector Search*
(Vladimir Čeperić). The paper is not distributed in this repository; its DOI is added
here on publication. `results/PAPER-MAP.md` maps each of its tables to the evidence
bundles below.

Use `--dry-run` to inspect commands. Every declared corpus is required by the full
tier. `python3 scripts/validate_claims.py` performs the portable retained-evidence
gate without requiring datasets or authoring Git history. `--strict` additionally
requires every recorded dataset and resolvable authoring commits. `--schema-only`
checks only the artifact configuration. `python3 scripts/restamp_bundles.py --check`
isolates the commit-reachability half of `--strict` and needs no datasets; it is
meaningful only where the authoring history exists, and it exists because a rebase
before promotion orphans a bundle's recorded revision while portable validation
stays green. Repair with `--map OLD=NEW --apply`, which refuses any mapping whose
two revisions do not carry byte-identical computation trees.

## Environment

The container is the portable environment:

```sh
docker build -t ultravec-ann .
docker run --rm -v "$PWD/data:/artifact/data" ultravec-ann \
  python3 scripts/reproduce.py doctor
```

**The plain image runs `doctor`, `check` and `smoke`; evidence tiers need the checkout.** `.dockerignore` excludes
`.git`, and every evidence tier refuses to run without a committed snapshot to
record in its provenance bundle, so `core`, `systems`, `mechanism`, `msweep`,
`scann`, `prepare` and `full` stop at that check inside a plain image. To run those in the
container, bind-mount the checkout itself so the repository travels with it:

```sh
docker run --rm -v "$PWD:/artifact" ultravec-ann \
  python3 scripts/reproduce.py core
```

Note also that the runtime stage carries the compiled binary and the cargo registry
but not `target/`, so the first cargo-based step in a fresh container rebuilds the
crate.

For a host build, install the Rust toolchain from `rust-toolchain.toml`, Python
3.12, a C/C++ build toolchain, and dependencies from `requirements.txt` into a
dedicated environment. `requirements.txt` is compiled for 3.12 with hash pins, so
an older interpreter cannot resolve it. Invoke the runner with that interpreter, for example
`data/venv/bin/python scripts/reproduce.py full`; every Python child step is pinned
to the same active interpreter. `doctor` reports a full/prepare tier as blocked if
the exact Faiss and ScaNN distributions are not present. The default build is
portable; CPU-native flags must only be enabled for explicitly labelled throughput
runs. Record CPU model, core count, thread count, Rust version, and container digest
with every timing result.

The codecs read thirty-seven `ULTRAVEC_*` environment variables, and they change results.
`scripts/reproduce.py` therefore refuses to inherit one it has not declared: each tier
states its own operating point under `[tiers.<name>.env]`, defaulting to
`ULTRAVEC_TRELLIS_MEM=12`, and the tiers behind the reported flat and systems tables
declare `16`. Every `ULTRAVEC_*` value in effect is
recorded in each bundle's `manifest.json` under `ultravec_env`; read it before
comparing against a bundle. Enumerate the current set with:

```sh
grep -rhoE 'ULTRAVEC_[A-Z0-9_]+' src/ scripts/ | sort -u
```

They fall into four groups. **Harness**: `ULTRAVEC_BIN`, `ULTRAVEC_RESULTS_DIR`,
`ULTRAVEC_ERAB_DIR`. **Trellis**: `_MEM` (state bits; the one the tiers set),
`_BEAM`, `_V`, `_CODE` (`1mad` selects the ablation codebook), `_CODES_ONLY`,
`_FIXEDSTART`, `_TAILBITE` (constrain the path to end in its start state, which
removes the stored start field; measured on three corpora in `tail-biting-sweep`:
free on GIST and DBpedia, 4.8 points on SIFT), `_TAILBITE_K`, `_RESID`, `_ANISO`,
`_SHORTLIST`, `_SHORTLIST_ASYM`, `_RESCALE_BITS`, `_START_BITS`. **Research arm**:
`ULTRAVEC_COSET_TRELLIS=1` adds a trellis-coded vector quantizer over E8 cosets to
the flat comparison (`src/coset_trellis.rs`); off by default so every retained
bundle reproduces byte-identically. **Per-codec**:
`ULTRAVEC_PVQ_BLOCK`, `ULTRAVEC_PVQ_K`, `ULTRAVEC_BLOCKQUANT_P`,
`ULTRAVEC_BLOCKQUANT_EST`, `ULTRAVEC_DEHUB_{R,RMAX,HEAD_BITS,ITERS,COMPLEMENT}`,
`ULTRAVEC_IVF_NLIST`, `ULTRAVEC_ROTATION` (`pca`/`itq`; a no-op for the oblivious
default), `ULTRAVEC_ROTATION_ROUNDS`, `ULTRAVEC_ITQ_ITERS`,
`ULTRAVEC_PQ_CODE_BITS` (bits per subquantizer; lower means more subspaces at the
same emitted rate --- the cold-start factorization control). **Estimator-axis
control**: `ULTRAVEC_BIASED_ESTIMATOR` and its per-codec overrides
`ULTRAVEC_EDEN_BIASED`, `ULTRAVEC_RABITQ_BIASED`, plus `ULTRAVEC_CENTER` and, when
centering is on, `ULTRAVEC_CENTER_K` (estimate the mean from a K-vector draw
instead of the whole base) and `ULTRAVEC_CENTER_SEED` (which draw).
**Shared across every codec**: `ULTRAVEC_ROTATION_SEED` selects the oblivious
rotation draw; it is read once in `lib.rs` so a sweep moves the whole field
together, because a per-codec seed would silently compare across transforms. The `check`
tier deliberately strips every one of them so unit tests cannot inherit a research
knob from the shell.

## Datasets

`datasets/manifest.toml` is the source of truth. List it and verify files staged
through the checksum-enforcing downloader with:

```sh
python3 scripts/download_datasets.py list            # dataset ids and status
python3 scripts/download_datasets.py fetch --dataset sift    # download + verify
python3 scripts/download_datasets.py verify --dataset sift   # re-check staged files
```

`fetch` is the command that actually downloads; `verify` only re-checks what is
already staged. Pass an id exactly as `list` prints it.

Fetching is only the first of four staging steps, and the later ones are not
optional for the core/systems/mechanism/full tiers:

```sh
python3 scripts/download_datasets.py fetch --dataset sift   # ... and gist, glove, dbpedia-*
python3 scripts/prepare_ann_benchmarks.py                   # HDF5 -> .fvecs, incl. the SIFTsmall prefix view
python3 scripts/prepare_glove.py                            # GloVe angular split -> .fvecs
python3 scripts/fetch_dbpedia_openai.py                     # pinned parquet shards -> .fvecs
pip download -r requirements.txt -d data/wheelhouse         # the pinned faiss/scann wheels
git clone https://github.com/VectorDB-NTU/RaBitQ-Library data/baselines/RaBitQ-Library
python3 scripts/verify_public_data.py                       # hashes everything, writes the verified-inputs stamp
```

`verify_public_data.py` is the bootstrap the `full` and `prepare` tiers expect to
have been run once; `full` runs it itself as `prepare`'s first step.

The downloader refuses placeholders, insecure URLs, checksum mismatches, and unsafe
archive paths. Raw and derived corpora are never committed. See
`datasets/LICENSES.md` and each dataset's upstream terms before use or
redistribution.

## Evidence and interpretation

- Recall runs use external held-out queries, exact in-harness ground truth for the
  stated metric, seed 42 unless otherwise recorded, and report Recall@1/10/100.
- The main flat comparison evaluates 1,000 queries against TurboQuant's MSE and
  inner-product arms, canonical RaBitQ at one bit, the artifact-native implementation
  of the published E-RaBitQ construction at two through four bits, UltraVec, PVQ,
  EDEN, BlockQuant, and an artifact-native E8 lattice baseline. The E8 baseline
  assigns each rotated 8-D block to the nearest point in a deterministic lowest-norm
  E8 codebook and uses the rescaled estimator; it is not presented as an upstream ANN
  implementation. It is tabulated at one and two bits; the artifact also implements
  three bits (a 2^24-point codebook resolved by closed-form nearest-point search,
  exact against brute force by test), where E8 lands fourth on GIST and third on
  DBpedia and displaces no reported margin. Four bits would need 2^32 points and is
  beyond enumeration. The comparison retains
  per-query recall and reports deterministic 10,000-sample paired-bootstrap
  intervals for UltraVec minus every comparator.
- Recall and encoded output are host-independent: rustc keeps `fp-contract=off`, so
  no FMA fusion changes f32 rounding, and the byte-identity tests pin it. Throughput
  is host-specific: the retained timings come from a `-C target-cpu=native` build (a
  flag no manifest field records) on the AVX-512 host named in each manifest's `cpu`
  field, and this repository builds portable by default, so validate wall-clock
  ordering and tolerance on the documented machine class rather than comparing
  absolute QPS across hosts.
- `trellis_recon` uses a resident fp32 reconstruction cache and is a decode-cost
  diagnostic, not a compressed-resident deployment; `trellis_codes` is the
  RAM-compressed mode. They must remain separate in tables and plots.
- The codec-system comparison controls for optimization level: every codec in the IVF
  table but one scores through the same runtime-detected AVX2/FMA kernels in
  artifact-native Rust, TurboQuant being the exception, left scalar as the frozen
  control. The trellis row is
  understated in the other direction: it divides by its stored rescale per candidate
  where the scalar codecs multiply by a precomputed reciprocal, so its throughput
  column is a floor. Recall decides that table in any case: E-RaBitQ's peak Recall@10
  over the whole probe sweep is `.4510` against UltraVec's `.5990`. Codes-only UltraVec emits its window stream directly from the
  packed codes -- the trellis state is the last `M` emitted bits, so a window is a field at a
  known offset rather than a value carried from the previous step -- and then gathers through a
  runtime-detected AVX2/FMA dense dot.
- Memory reports separate serialized codes, shared model state, index structures,
  and decoded caches. Compare deployments using total resident bytes, not code bytes alone.
- The PCA-head residual hybrid is data-dependent and batch fitted. Its legacy CLI
  and Rust backend name is `dehub`; new library users should pass `DehubConfig`
  explicitly, while environment variables remain only for legacy CLI runs.
- The official E-RaBitQ cross-check uses an exact Apache-2.0 RaBitQ-Library commit
  with its native estimator and search path. A separately hashed one-line patch in
  an ignored detached worktree replaces the rotator's unseeded
  `std::random_device` draw with seed 42 while leaving quantization, indexing,
  estimation, and search unchanged; neither the checkout nor datasets are
  redistributed here.
- In that cross-check, E-RaBitQ uses the upstream inner-product path and the fitted
  PCA-head side uses exact cosine over normalized reconstructions. Its byte rows
  count emission payload and, for the hybrid, the head and free-start state; they
  do not assert a common serialized scorer layout.
- Long steps stream stdout and stderr while retaining identical per-experiment
  logs in the provenance bundle, keeping multi-hour inner-loop progress visible.
- The checked-in evidence logs normalize the artifact root and ephemeral
  second-run directory to `.` and `<temporary-directory>`; numerical output and
  provenance manifests are unchanged. Tiers that write their own logs rather than
  going through `record_bundle` apply the same substitution at the point of write.
- Shared `.fvecs`, normalization, reconstruction, and q8 helpers live in the
  domain-neutral `scripts/vector_io.py`; the ANN drivers depend on nothing outside
  this repository.

## Repository hygiene

This is the public repository <https://github.com/vceperic/ultravec-ann>; release
`v0.1.0` is the submission snapshot, and an archival DOI is added after acceptance.
Each evidence bundle records the authoring commit it ran at. Three source additions
postdate some bundles: an encoder guard for degenerate rows and a record-level API
used by a downstream consumer, both of which leave every encoded bit unchanged
(pinned by the byte-identity tests in `src/trellis.rs`), and TurboQuant's QJL scale
for orthonormal sketch rows (`residual::qjl_scale`), which touches only the
sketched-residual arms, whose rows come from the `tqip` and `resid` tiers recorded
after it. CI scans every
push for absolute home-directory paths, private keys, and credential-shaped strings
and fails the build on a hit. The target journal uses single-blind review, so author
and citation metadata identify Vladimir Čeperić by design; see `NOTICE` for
ownership. The owner handoff and post-publication steps are in `RELEASE.md`.

## License

The original software and artifact-specific material is licensed under the
[MIT license](LICENSE), matching the `license = "MIT"` field in `Cargo.toml`.
Third-party baselines keep their own terms — see `baselines/LICENSES.md` — and the
evaluated corpora are neither redistributed nor relicensed here; see
`datasets/LICENSES.md`.

## Security and citation

Do not run downloaded archives or untrusted `.fvecs` files outside the container.
Citation metadata is in `CITATION.cff`. For anything else, including a suspected
vulnerability, open an issue or contact the author at the address recorded there.
