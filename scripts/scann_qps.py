#!/usr/bin/env python3
"""ScaNN (tree + asymmetric hashing) recall@10 vs QPS on SIFT-1M.

This produces the second pinned mature-system reference, alongside Faiss IVFPQ.
Configuration: 1024 leaves (= IVF nlist=1024), AH with 2 dims/block @4-bit = 32 B/vec
(the 2-bit/dim analogue of the oblivious 36-40 B codecs), 20 threads, SIFT-1M,
exact-cosine (dot on L2-normalized) top-10 gold, num_leaves_to_search sweep.

Run: python3 scripts/scann_qps.py --seed 42 --verify-determinism
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIFT = ROOT / "data" / "sift"
NLEAVES = 1024
DIMS_PER_BLOCK = 2      # 64 blocks on 128-d; AH 4-bit codebook => 64*4bit = 32 B/vec
NLS = [1, 2, 4, 8, 16, 32, 64, 128]   # num_leaves_to_search (nprobe analogue)
THREADS = 20
TRAINING_THREADS = 1
K = 10
QUERY_MAX = 200
GOLD_BLOCK = 50000
MIN_TIMING_SECONDS = 0.5


def read_fvecs(p):
    a = np.fromfile(p, dtype=np.int32)
    d = a[0]
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))


def build_searcher(scann, db, seed):
    """Build through ScaNN's config proto so every training RNG is controlled."""
    from google.protobuf import text_format
    from scann.proto import scann_pb2

    config_text = (
        scann.scann_ops_pybind.builder(db, K, "dot_product")
        .tree(
            num_leaves=NLEAVES,
            num_leaves_to_search=max(NLS),
            training_sample_size=250000,
        )
        .score_ah(DIMS_PER_BLOCK, anisotropic_quantization_threshold=0.2)
        .create_config()
    )
    config = scann_pb2.ScannConfig()
    text_format.Parse(config_text, config)
    config.partitioning.clustering_seed = seed
    config.partitioning.single_machine_center_initialization = 0
    config.partitioning.num_cpus = TRAINING_THREADS
    config.hash.asymmetric_hash.clustering_seed = seed
    config.hash.asymmetric_hash.sampling_seed = seed
    config.hash.asymmetric_hash.num_cpus = TRAINING_THREADS
    return scann.scann_ops_pybind.create_searcher(
        db, text_format.MessageToString(config), training_threads=TRAINING_THREADS
    )


def neighbor_fingerprints(searcher, queries):
    fingerprints = {}
    for nls in NLS:
        ids, _ = searcher.search_batched(
            queries, leaves_to_search=nls, final_num_neighbors=K
        )
        fingerprints[str(nls)] = hashlib.sha256(
            np.ascontiguousarray(ids, dtype=np.int64).tobytes()
        ).hexdigest()
    return fingerprints


def verify_determinism(first, queries, seed):
    first_fingerprints = neighbor_fingerprints(first, queries)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--seed",
            str(seed),
            "--fingerprint-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "independent ScaNN build failed:\n" + completed.stderr[-4000:]
        )
    second_fingerprints = json.loads(completed.stdout)
    if first_fingerprints != second_fingerprints:
        differing = sorted(
            nls
            for nls in first_fingerprints
            if first_fingerprints[nls] != second_fingerprints.get(nls)
        )
        raise RuntimeError(
            "seeded ScaNN processes differ at leaves_to_search="
            + ",".join(differing)
        )
    print(
        "determinism: PASS (two independent processes returned identical neighbors "
        "for every searched-leaf setting)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="build twice and require identical neighbors at every sweep point",
    )
    parser.add_argument("--fingerprint-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    np.random.seed(args.seed)
    import scann

    DB = read_fvecs(f"{SIFT}/sift_base.fvecs").astype(np.float32)
    Q = read_fvecs(f"{SIFT}/sift_query.fvecs")[:QUERY_MAX].astype(np.float32)
    DB /= np.linalg.norm(DB, axis=1, keepdims=True) + 1e-12
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    n, d = DB.shape
    if args.fingerprint_only:
        searcher = build_searcher(scann, DB, args.seed)
        print(json.dumps(neighbor_fingerprints(searcher, Q), sort_keys=True))
        return

    print(
        f"SIFT-1M: {n} base, {Q.shape[0]} queries, dim {d}, "
        f"{THREADS} search threads, {TRAINING_THREADS} training thread, "
        f"seed {args.seed}",
        flush=True,
    )

    # exact-cosine gold (brute force on normalized => dot)
    print("computing exact gold ...", flush=True)
    best_scores = np.full((len(Q), K), -np.inf, dtype=np.float32)
    gold = np.full((len(Q), K), -1, dtype=np.int64)
    for start in range(0, len(DB), GOLD_BLOCK):
        scores = Q @ DB[start:start + GOLD_BLOCK].T
        ids = np.broadcast_to(
            np.arange(start, start + scores.shape[1], dtype=np.int64), scores.shape
        )
        all_scores = np.concatenate((best_scores, scores), axis=1)
        all_ids = np.concatenate((gold, ids), axis=1)
        keep = np.argpartition(-all_scores, K - 1, axis=1)[:, :K]
        best_scores = np.take_along_axis(all_scores, keep, axis=1)
        gold = np.take_along_axis(all_ids, keep, axis=1)

    print("building ScaNN (tree + AH) ...", flush=True)
    t0 = time.perf_counter()
    searcher = build_searcher(scann, DB, args.seed)
    build = time.perf_counter() - t0
    bytes_vec = (d // DIMS_PER_BLOCK) * 4 // 8   # blocks * 4 bit / 8 = bytes
    print(f"built in {build:.1f}s, ~{bytes_vec} B/vec (AH {DIMS_PER_BLOCK}dims/block, 4-bit)")
    if args.verify_determinism:
        print("building independent-process ScaNN determinism check ...", flush=True)
        verify_determinism(searcher, Q, args.seed)

    def recall(I):
        return sum(len(set(I[i]) & set(gold[i])) for i in range(len(I))) / (len(I) * K)

    print("\n=== ScaNN tree+AH — recall@10 vs QPS ===")
    print("  nls    R@10      QPS")
    for nls in NLS:
        searcher.search_batched(Q[:100], leaves_to_search=nls)  # warmup
        t0 = time.perf_counter()
        runs = 0
        while True:
            I, _ = searcher.search_batched(Q, leaves_to_search=nls, final_num_neighbors=K)
            runs += 1
            dt = time.perf_counter() - t0
            if dt >= MIN_TIMING_SECONDS:
                break
        print(f"  {nls:>4}   {recall(I):.4f}   {runs*len(Q)/dt:9.0f}", flush=True)
    print("\nDONE scann_qps")

if __name__ == "__main__":
    main()
