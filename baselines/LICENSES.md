# Baseline provenance and licenses

The artifact does not redistribute baseline source trees or wheels. It records and
verifies the exact external revisions used by the public-data campaign in
`baselines/manifest.toml`.

* **RaBitQ-Library** is fetched from the official VectorDB-NTU repository at commit
  `540242ea0a68926f1b827bf1f9add844f07a427b` under Apache-2.0. The upstream
  `LICENSE` SHA-256 is pinned in the manifest. `official_harness/erab_ivf_query10.cpp`
  changes only result serialization and the requested top-k. The upstream FHT
  rotator draws entropy from `std::random_device` and exposes no seed. For
  reproducible comparisons, the recorded
  `official_harness/rabitq-fixed-rotation-seed.patch` changes only that entropy
  source to the artifact seed 42 in an ignored detached worktree. The official
  comparison uses canonical RaBitQ at one bit and E-RaBitQ at multi-bit rates. The manifest
  pins the patch SHA-256; quantization, indexing, scoring, and search logic remain
  upstream code.
* **Faiss** is installed as the official `faiss-cpu==1.14.2` wheel, verified by its
  SHA-256. The release corresponds to source tag `v1.14.2`, peeled to commit
  `8f038486672564190a78307e4fb5ce2da6c43d45`, and declares the MIT license.
* **ScaNN** is installed as the official `scann==1.4.2` CPython 3.12 wheel, verified
  by its SHA-256, and declares Apache-2.0. The wheel is used as the executable pin
  because the Google Research monorepo does not provide a matching release tag.

The package lock in `requirements.txt` pins all transitive Python dependencies with
hashes. Baseline names and licenses remain the property of their respective owners.
