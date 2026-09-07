"""
Project-folder auto-detection.

Two things this project already has on disk that the app should not make
someone re-supply by hand every session:

    - the ten defect-free reference boards, in data/raw/PCB_USED/
    - the trained weights for the team's winning configuration, somewhere
      under runs/

Both are optional - the app works without either, falling back to manual
upload - but when the project folder layout is present, using it
automatically is what "the winner is incorporated into the program" means
in practice, not just which preprocessing technique is selected by default.
"""

import re
from pathlib import Path

# The run folder name to prefer when more than one candidate is found under
# runs/. This is the confirmed winning configuration - see
# core/preprocessing.py's DEFAULT_TECHNIQUE for the matching image
# processing side of the same decision. Update this constant, not the
# search logic below, if a later result supersedes it.
WINNING_RUN_NAME = "A3B2"

_SEED_SUFFIX = re.compile(r"_s\d+$")


def find_project_root(start=None):
    """Walk upward from `start` (default: the current directory) until a
    folder containing "data" is found.

    Mirrors the PROJECT_ROOT search used throughout the team's notebooks,
    so this behaves the same way regardless of whether the app is launched
    from the app/ folder itself or from the project root.

    Returns None - rather than searching indefinitely - if nothing is
    found within a few levels. A missing root is a normal, handled case
    here (the caller falls back to manual upload for everything), not an
    error worth raising.
    """
    current = Path(start) if start else Path.cwd()
    for _ in range(6):
        if (current / "data").is_dir():
            return current
        if current == current.parent:
            break
        current = current.parent
    return None


def find_matching_reference(filename, pcb_used_dir):
    """Find the defect-free reference board matching an uploaded filename.

    Follows this project's naming convention: "<board_id>_<defect>_<n>.jpg"
    -> a file named "<board_id>.<ext>" in PCB_USED. Returns None if the
    filename does not fit that pattern or no matching file exists; the
    caller should fall back to a manually supplied reference in that case,
    not treat this as an error.
    """
    if pcb_used_dir is None or not Path(pcb_used_dir).is_dir():
        return None

    board_id = Path(filename).stem.split("_")[0]
    for extension in (".JPG", ".jpg", ".jpeg", ".png"):
        candidate = Path(pcb_used_dir) / f"{board_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def discover_trained_weights(project_root, preferred_name=WINNING_RUN_NAME):
    """Find candidate best.pt files under runs/, best guess first.

    A run folder whose name contains `preferred_name` is ranked ahead of
    every other run, since that is the confirmed winning configuration.
    Among those, a plain run (e.g. "A3B2") is preferred over a
    seed-suffixed one (e.g. "A3B2_s0"), because the plain run is the one
    that appeared in the main comparison table - the seed-suffixed runs
    exist only to verify that result, not to be the reported model.

    Returns a list of (label, Path) tuples, sorted best guess first. Empty
    if runs/ does not exist or contains no weights at all - the caller
    should fall back to manual upload in that case.
    """
    if project_root is None:
        return []

    runs_dir = Path(project_root) / "runs"
    if not runs_dir.is_dir():
        return []

    candidates = []
    for weights_path in runs_dir.glob("**/weights/best.pt"):
        run_name = weights_path.parent.parent.name
        candidates.append((run_name, weights_path))

    def sort_key(item):
        run_name, _ = item
        not_preferred = 0 if preferred_name.lower() in run_name.lower() else 1
        has_seed_suffix = 1 if _SEED_SUFFIX.search(run_name) else 0
        return (not_preferred, has_seed_suffix, run_name)

    candidates.sort(key=sort_key)

    return [(f"{run_name} ({weights_path.relative_to(runs_dir)})", weights_path)
            for run_name, weights_path in candidates]
