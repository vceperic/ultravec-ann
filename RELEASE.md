# Public-release handoff

This directory is the complete repository payload for
<https://github.com/vceperic/ultravec-ann>. It deliberately contains no nested
`.git` directory. The author can copy its contents to a new repository root and
perform the GitHub initialization, commit, tag, and release there.

## Before the first public push

0. **From the authoring repository, before copying anything**, confirm every bundle
   still names a reachable commit:

   ```sh
   python3 scripts/restamp_bundles.py --check
   python3 scripts/validate_claims.py --strict
   ```

   This step cannot be deferred. Strict validation recomputes each bundle's
   computation-tree hash *at its recorded revision*, which needs the authoring
   history; the fresh repository created below does not have it, and there every
   bundle degrades to the portable check. It also re-hashes every recorded dataset
   file, so run it on the campaign machine with the corpora staged under `data/` --
   a missing corpus is a hard error in strict mode and only a NOTE without it.
   The paper tree wires both into one target: `make check-provenance`. A rebase before promotion orphans the
   recorded revision while leaving portable validation green, so this is the only
   place the breakage is visible. If `--check` fails, repair it with
   `restamp_bundles.py --map <old>=<new> --apply` — it refuses any mapping whose two
   revisions do not carry byte-identical computation trees.

1. Copy the directory contents, excluding ignored local state such as `data/`,
   `target/`, virtual environments, and caches. Initialize a fresh repository at
   the new root rather than carrying over any prior working history — the payload
   is self-contained and needs none of it.
2. Confirm that `paper/ultravec-ann-vldbj.pdf` is the final submission build and
   that the root contains `LICENSE`, `NOTICE`, `CITATION.cff`, and this file.
3. Note for anyone comparing throughput: the authoring campaigns were built inside
   the monorepo, whose `.cargo/config.toml` sets `-C target-cpu=native` on an
   AVX-512 host, and no manifest field records a build flag. Recall and encode
   output are unaffected --- rustc keeps `fp-contract=off`, so no FMA fusion
   changes f32 rounding, which the byte-identity tests pin --- but wall-clock and
   QPS are host-specific, as `README.md` already says. This repository builds
   portable by default, which is the right default and not the campaign's
   configuration.
4. Run the release gates from the repository root:

   ```sh
   python3 scripts/validate_claims.py
   python3 -m unittest discover -s scripts -p 'test_*.py' -v
   cargo fmt --all -- --check
   cargo clippy --locked --all-targets -- -D warnings
   cargo test --locked --all-targets
   ```

   These commands validate retained results; they do not rerun the paper
   experiments.
5. Inspect the first commit before pushing. In particular, `data/`, `target/`,
   `.git/`, credentials, private keys, and authoring-machine paths must not be in
   the payload.
6. Push to `vceperic/ultravec-ann` and wait for both GitHub Actions jobs to pass.

## Tag and archive sequence

Create the public tag and GitHub release only after the first commit passes CI.
The current software version is `0.1.0`; use a matching tag such as `v0.1.0` unless
the version is changed consistently first. Do not add a release date before the
release exists.

The manuscript is intentionally set up for an archival identifier to be added
after initial submission. Once a repository archive/DOI exists:

- add the real DOI and release date to `CITATION.cff`;
- add the DOI to the manuscript's code-availability statement;
- rebuild the manuscript and refresh `paper/ultravec-ann-vldbj.pdf`; and
- make a follow-up tag if those metadata changes occur after `v0.1.0`.
