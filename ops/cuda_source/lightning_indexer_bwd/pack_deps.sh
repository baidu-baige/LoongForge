#!/usr/bin/env bash
# pack_deps.sh -- Package third-party dependencies for offline transfer to a GPU test machine.
#
# What is packed:
#   third_party/DeepGEMM/  (excluding .git directories)
#
# How to use on the target machine:
#   1. Unpack:  tar -xzf deps_<date>.tar.gz -C <remote_dir>/
#   2. Move:    mv <remote_dir>/deps/DeepGEMM  <lightning_indexer_bwd_dir>/vendor/
#   3. Build:   cd <lightning_indexer_bwd_dir> && bash install.sh
#      (SKIP_DEEP_GEMM_CLONE defaults to 1, so the build reads from vendor/DeepGEMM directly)
#
# Usage:
#   bash pack_deps.sh                      # produces ../deps_<date>.tar.gz
#   bash pack_deps.sh -o /tmp/deps.tar.gz  # custom output path
#   bash pack_deps.sh --dry-run            # show what would be packed + estimated size

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
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATE="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="${SCRIPT_DIR}/../deps_${DATE}.tar.gz"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
OUTPUT="$(cd "$(dirname "$OUTPUT")" 2>/dev/null && pwd)/$(basename "$OUTPUT")"

DEEPGEMM_SRC="${SCRIPT_DIR}/third_party/DeepGEMM"

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if [[ ! -d "$DEEPGEMM_SRC" ]]; then
    echo "ERROR: third_party/DeepGEMM not found." >&2
    echo "Run:  cd third_party && git clone --recurse-submodules https://github.com/deepseek-ai/DeepGEMM.git" >&2
    exit 1
fi

# Check key submodules are populated
for sub in third-party/cutlass third-party/fmt; do
    if [[ ! -d "${DEEPGEMM_SRC}/${sub}" ]] || [[ -z "$(ls -A "${DEEPGEMM_SRC}/${sub}" 2>/dev/null)" ]]; then
        echo "ERROR: submodule ${sub} is empty. Run:" >&2
        echo "  cd third_party/DeepGEMM && git submodule update --init --recursive" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Dry-run: show size estimate
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== Dry run: what would be packed ==="
    echo ""
    echo "Source : $DEEPGEMM_SRC"
    echo "Target : deps/DeepGEMM/  (inside archive)"
    echo "Output : $OUTPUT"
    echo ""
    echo "--- Size breakdown (excluding .git) ---"
    python3 -c "
import os

def dir_size(path, exclude=('.git',)):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

src = '$DEEPGEMM_SRC'
total = dir_size(src)
subdirs = [
    ('DeepGEMM core', src, ['third-party']),
    ('third-party/cutlass', os.path.join(src, 'third-party', 'cutlass'), []),
    ('third-party/fmt',     os.path.join(src, 'third-party', 'fmt'),     []),
]
for label, path, skip in subdirs:
    sz = dir_size(path, exclude=tuple(['.git'] + skip))
    print(f'  {label:<25s}  {sz/1024/1024:6.1f} MB')
print(f'  {\"Total (uncompressed)\":<25s}  {total/1024/1024:6.1f} MB')
print()
print('  Compressed (.tar.gz) will typically be 30-50% of uncompressed size.')
"
    exit 0
fi

# ---------------------------------------------------------------------------
# Create archive via Python tarfile (portable, skips .git directories)
# ---------------------------------------------------------------------------
echo "Packing DeepGEMM -> ${OUTPUT} ..."
echo "(Excluding .git directories -- archive is source-only, not a git repo)"
echo ""

python3 - "$DEEPGEMM_SRC" "$OUTPUT" <<'PYEOF'
import sys, os, tarfile

src_dir  = sys.argv[1]   # third_party/DeepGEMM absolute path
out_path = sys.argv[2]

def should_exclude(path):
    """Return True for any path component named .git"""
    return any(part == '.git' for part in path.replace('\\', '/').split('/'))

file_count = 0
with tarfile.open(out_path, 'w:gz') as tar:
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Prune .git directories in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d != '.git']

        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            # Archive path: deps/DeepGEMM/<relative>
            rel      = os.path.relpath(abs_path, os.path.dirname(src_dir))
            arcname  = os.path.join('deps', rel)
            tar.add(abs_path, arcname=arcname, recursive=False)
            file_count += 1

print(f'  Files packed: {file_count}', flush=True)
PYEOF

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
SIZE=$(du -sh "$OUTPUT" | cut -f1)

echo ""
echo "Done."
printf "  %-14s %s\n" "Archive size:" "$SIZE"
printf "  %-14s %s\n" "Output:"       "$OUTPUT"
echo ""
echo "On the target machine:"
echo "  # Step 1: transfer"
echo "  scp ${OUTPUT} <user>@<host>:<remote_dir>/"
echo ""
echo "  # Step 2: unpack and place under vendor/"
echo "  tar -xzf $(basename "$OUTPUT") -C <lightning_indexer_bwd_dir>/"
echo "  mv <lightning_indexer_bwd_dir>/deps/DeepGEMM  <lightning_indexer_bwd_dir>/vendor/"
echo "  rmdir <lightning_indexer_bwd_dir>/deps"
echo ""
echo "  # Step 3: build (SKIP_DEEP_GEMM_CLONE=1 is the default, reads from vendor/DeepGEMM)"
echo "  cd <lightning_indexer_bwd_dir> && bash install.sh"
