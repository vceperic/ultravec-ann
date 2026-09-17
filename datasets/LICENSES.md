# Dataset provenance and use conditions

The artifact downloads source objects for local reproduction but does not redistribute
any raw vectors, source text, or derived `fvecs` files. SHA-256 and byte lengths are
recorded in `manifest.toml`; the URLs below were reviewed on 2026-07-22.

## SIFT-128 and GIST-960

The exact HDF5 objects are the public ANN-Benchmarks packages of the Texmex SIFT1M
and GIST1M research benchmarks. ANN-Benchmarks documents the splits and top-100
ground truth but does not state a separate license for the underlying vector data.
Accordingly, the repository does not redistribute them and makes no broader license
claim. Users should cite ANN-Benchmarks and the original benchmark publications.

- https://github.com/erikbern/ann-benchmarks#data-sets
- https://doi.org/10.1016/j.is.2019.02.006
- https://doi.org/10.1109/TPAMI.2010.57

## GloVe-25

ANN-Benchmarks supplies the deterministic HDF5 train/test package. Stanford states
that the underlying pretrained GloVe vectors are available under the Open Data
Commons Public Domain Dedication and License 1.0.

- https://nlp.stanford.edu/projects/glove/
- https://opendatacommons.org/licenses/pddl/1-0/

## DBpedia-OpenAI

The Hugging Face dataset card reports MIT for the embedding package. It says the
records derive from BEIR DBpedia-Entity; that source is marked CC BY-SA 4.0, while
DBpedia identifies CC BY-SA/GFDL terms for its data. The artifact therefore treats
the upstream content terms as controlling, provides attribution, and does not
redistribute either the text or embeddings. Three parquet shards are pinned at
snapshot `af9b8869cc2d8debbd254d77737865bb09a2067f`.

- https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M
- https://huggingface.co/datasets/BeIR/dbpedia-entity
- https://www.dbpedia.org/imprint/

This inventory records upstream representations; it is not legal advice and does not
grant rights beyond the cited terms.
