"""Normalize machine-specific paths out of retained evidence, at the point of write.

Evidence logs are committed and published, so any absolute path in them describes
the authoring machine rather than the experiment. `reproduce.py` rewrites the
artifact root for the stdout it captures, but three drivers write their own logs
and each has to do the same thing:

* `generate_mechanism_inputs.py` and `mechanism_msweep.py` send the benchmark's
  side-effect summaries to a scratch directory, whose randomly generated name the
  binary then echoes into the log;
* `prepare_official_rabitq.py` runs its determinism recheck in a second temporary
  directory and echoes the commands it runs there.

One shared helper keeps the normalization rules identical across all three drivers;
`test_bundle_hygiene.py` enforces them in CI.
"""
from __future__ import annotations

from pathlib import Path


def redact(text: str, root: Path, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """Rewrite `root` to "." and each `extra` literal to its placeholder.

    Longest literal first, so a nested path cannot be partially rewritten by its
    own parent.
    """
    rules = [(str(root), "."), *extra]
    for literal, placeholder in sorted(rules, key=lambda pair: -len(pair[0])):
        if literal:
            text = text.replace(literal, placeholder)
    return text


SCRATCH = "<scratch-directory>"
TEMPORARY = "<temporary-directory>"
