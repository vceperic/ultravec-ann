# Pinned upstream E-RaBitQ comparison harness

The adapters in this directory are built against the pinned official
RaBitQ-Library revision listed in `baselines/manifest.toml`.

The upstream FHT rotation uses `std::random_device`, so two otherwise identical
index builds do not reproduce the same values. The artifact applies
`rabitq-fixed-rotation-seed.patch` to an ignored detached worktree and fixes that
rotation seed to 42. The patch changes no quantization, indexing, estimator, or
search logic, and its SHA-256 is pinned in the baseline manifest. The original
upstream checkout remains clean and is verified before each build.

`erab_ivf_query10.cpp` is the output adapter used by
`scripts/prepare_official_rabitq.py`; it changes result serialization and the
requested top-k only. The preparation script builds the upstream IVF indexer,
produces its top-10 outputs at one through four bits, and verifies byte-identical
outputs across two independent runs. Multi-bit rows use the upstream E-RaBitQ
construction and native search path.
