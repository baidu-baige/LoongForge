#!/usr/bin/env bash
# pack.sh -- Package lightning_indexer_bwd for transfer to a remote GPU test machine.
#
# What is included:
#   - All source files tracked by git (csrc/, lightning_indexer_bwd/, tests/, setup.py, etc.)
#   - Key untracked files: CLAUDE.md, .gitignore, install.sh, pack.sh, pack_deps.sh
#   - Active workspace topics: latest step file from each workspace/<topic>/ that has
#     a MANIFEST file, placed at the main library path declared in MANIFEST.
#     This makes the archive self-contained -- no manual file moves needed on the test machine.
#
# MANIFEST format (workspace/<topic>/MANIFEST):
#   # comments start with #
#   step_latest -> <destination relative to repo root>   # "step_latest" = highest step_NN_* file
#   step_03_foo.cuh -> csrc/apis/some_file.hpp           # or an explicit filename
#
# What is excluded:
#   - vendor/          (auto-generated at build time on target)
#   - third_party/     (local dev reference only)
#   - workspace/       (sandbox; active topics are overlaid via MANIFEST, archive is local only)
#   - CLAUDE.zh.md     (local Chinese reading copy)
#   - build/ dist/ *.egg-info/ __pycache__/ *.pyc *.so *.o *.a  .git/
#
# Usage:
#   bash pack.sh                      # produces ../lightning_indexer_bwd_<date>.tar.gz
#   bash pack.sh -o /tmp/my.tar.gz    # custom output path
#   bash pack.sh --dry-run            # list files (including workspace overlays) without creating archive

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
OUTPUT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUTPUT="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$SCRIPT_DIR")"
DATE="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="${SCRIPT_DIR}/../${REPO_NAME}_${DATE}.tar.gz"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
OUTPUT="$(cd "$(dirname "$OUTPUT")" 2>/dev/null && pwd)/$(basename "$OUTPUT")"

# ---------------------------------------------------------------------------
# Build file list + workspace overlays via Python
# (single Python process handles both to share the overlay map)
# ---------------------------------------------------------------------------
cd "$SCRIPT_DIR"

FILELIST=$(mktemp)
trap 'rm -f "$FILELIST"' EXIT

# 1. Git-tracked files
if git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    git ls-files >> "$FILELIST"
fi

# 2. Important untracked files that travel with the repo
for f in CLAUDE.md .gitignore install.sh pack.sh pack_deps.sh .claude/settings.json; do
    if [[ -f "$f" ]] && ! grep -qxF "$f" "$FILELIST" 2>/dev/null; then
        echo "$f" >> "$FILELIST"
    fi
done

sort -u "$FILELIST" -o "$FILELIST"

# ---------------------------------------------------------------------------
# Dry-run or real pack -- both handled by the same Python script
# ---------------------------------------------------------------------------
python3 - "$SCRIPT_DIR" "$REPO_NAME" "$OUTPUT" "$FILELIST" "$DRY_RUN" <<'PYEOF'
import sys, os, tarfile, glob

src_dir   = sys.argv[1]
repo_name = sys.argv[2]
out_path  = sys.argv[3]
list_file = sys.argv[4]
dry_run   = sys.argv[5] == "1"

# ------------------------------------------------------------------
# Load main file list
# ------------------------------------------------------------------
with open(list_file) as f:
    main_files = [l.rstrip('\n') for l in f if l.strip()]

# ------------------------------------------------------------------
# Scan workspace/ for active topics with a MANIFEST
# overlays: { destination_rel_path -> abs_source_path }
# ------------------------------------------------------------------
overlays = {}   # dest_rel -> abs_src
workspace_dir = os.path.join(src_dir, 'workspace')

if os.path.isdir(workspace_dir):
    for entry in sorted(os.listdir(workspace_dir)):
        topic_path = os.path.join(workspace_dir, entry)
        # skip archive file/dir and non-directories
        if not os.path.isdir(topic_path) or entry in ('archive',):
            continue
        manifest_path = os.path.join(topic_path, 'MANIFEST')
        if not os.path.exists(manifest_path):
            print(f"  [workspace] {entry}: no MANIFEST, skipping", flush=True)
            continue

        with open(manifest_path) as mf:
            rules = [l.strip() for l in mf if l.strip() and not l.startswith('#')]

        for rule in rules:
            if '->' not in rule:
                print(f"  [workspace] {entry}: bad MANIFEST line: {rule!r}", flush=True)
                continue
            src_spec, dest = [x.strip() for x in rule.split('->', 1)]

            # Resolve source: "step_latest" = highest step_NN_* file
            if src_spec == 'step_latest':
                candidates = sorted(
                    glob.glob(os.path.join(topic_path, 'step_[0-9][0-9]_*'))
                )
                if not candidates:
                    print(f"  [workspace] {entry}: no step_NN_* files found", flush=True)
                    continue
                abs_src = candidates[-1]
            else:
                abs_src = os.path.join(topic_path, src_spec)

            if not os.path.isfile(abs_src):
                print(f"  [workspace] {entry}: source not found: {abs_src}", flush=True)
                continue

            overlays[dest] = abs_src
            step_name = os.path.basename(abs_src)
            print(f"  [workspace] {entry}: {step_name} -> {dest}", flush=True)

# ------------------------------------------------------------------
# Dry-run: just print
# ------------------------------------------------------------------
if dry_run:
    print("\n=== Dry run: files that would be packed ===")
    print("\n-- Main library --")
    for rel in main_files:
        marker = " (overridden by workspace)" if rel in overlays else ""
        print(f"  {rel}{marker}")
    if overlays:
        print("\n-- Workspace overlays (packed at main library paths) --")
        for dest, src in overlays.items():
            if dest not in main_files:
                print(f"  {dest}  [new, from workspace]")
    total = len(set(list(main_files) + list(overlays.keys())))
    print(f"\nTotal  : {total} files")
    print(f"Output : {out_path}")
    sys.exit(0)

# ------------------------------------------------------------------
# Real pack: main files + workspace overlays
# ------------------------------------------------------------------
# Build effective file map: dest -> abs_src (overlays win over main files)
file_map = {}
for rel in main_files:
    file_map[rel] = os.path.join(src_dir, rel)
for dest, abs_src in overlays.items():
    file_map[dest] = abs_src   # overlay wins

with tarfile.open(out_path, 'w:gz') as tar:
    for rel, abs_path in sorted(file_map.items()):
        if not os.path.isfile(abs_path):
            print(f"  [skip] {rel}  (not found)", flush=True)
            continue
        arcname = os.path.join(repo_name, rel)
        tar.add(abs_path, arcname=arcname)

print(f"  Files packed: {len(file_map)}", flush=True)
PYEOF

# ---------------------------------------------------------------------------
# Summary (only for real pack)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 0 ]]; then
    SIZE=$(du -sh "$OUTPUT" | cut -f1)
    echo ""
    echo "Done."
    printf "  %-14s %s\n" "Archive size:" "$SIZE"
    printf "  %-14s %s\n" "Output:"       "$OUTPUT"
    echo ""
    echo "Transfer and unpack on target machine:"
    echo "  scp ${OUTPUT} <user>@<host>:<remote_dir>/"
    echo "  tar -xzf $(basename "$OUTPUT") -C <remote_dir>/"
    echo "  cd <remote_dir>/${REPO_NAME} && bash install.sh"
fi
