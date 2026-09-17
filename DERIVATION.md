# Artifact derivation and scope

This repository holds the ANN-relevant portion of the author's UltraVec work. It is
deliberately a self-contained copy rather than a submodule of a larger tree: each
paper in the UltraVec series receives an independent repository and release record,
so this one can be cloned, built, and verified on its own.

Included material comprises the Rust quantizers and ANN index harnesses, public ANN
dataset staging/analysis scripts, the official RaBitQ comparison harness, retained
evidence, and an author-generated PDF of the associated paper. The paper's LaTeX
source is maintained separately. KV-cache, genomics, protein/RNA, single-cell,
AlphaEarth, internal
product integration, local datasets, build products, and research roadmaps are out
of scope.

New evidence records a clean authoring commit, computation hash, command,
environment, and input hashes. CI additionally scans every push for absolute
home-directory paths, private keys, and credential-shaped strings, and fails the
build on a hit.

All implementation source and artifact-specific scripts in this repository are
original work owned solely by Vladimir Čeperić. No third-party or copyleft
implementation source was incorporated.
External dependencies and independently fetched baselines remain separate items, as
recorded in the manifests and `NOTICE`.

The official E-RaBitQ comparison targets the upstream RaBitQ-Library, licensed
Apache-2.0. The setup script pins an exact upstream commit and verifies its license;
upstream implementation files are fetched into ignored local data rather than
redistributed as UltraVec source.
