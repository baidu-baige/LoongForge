#!/bin/bash
# apply_repo_patches.sh - Apply patches generated from repository comparison
# This script applies patches (A relative to B) to the baseline repository B

set -euo pipefail # Exit on error, undefined variable, pipe fail

# Configuration
PATCH_DIR="${1:-patches}"
TARGET_REPO="${2:-./target-repo}"
NON_INTERACTIVE="${NON_INTERACTIVE:-false}"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Repository Patch Application Tool ===${NC}\n"

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Patch Directory: $PATCH_DIR"
echo "  Target Repository: $TARGET_REPO"
echo ""

# Check if patch directory exists
if [ ! -d "$PATCH_DIR" ]; then
    echo -e "${RED}Error: Patch directory does not exist: $PATCH_DIR${NC}"
    echo "Usage: $0 [patch_dir] [target_repo_path]"
    exit 1
fi

# Check if repository exists
if [ ! -d "$TARGET_REPO" ]; then
    echo -e "${RED}Error: Target repository does not exist: $TARGET_REPO${NC}"
    echo "Please ensure the repository is cloned locally"
    exit 1
fi

# Check if it's a git repository
if [ ! -d "$TARGET_REPO/.git" ]; then
    echo -e "${RED}Error: $TARGET_REPO is not a Git repository${NC}"
    exit 1
fi

# Convert PATCH_DIR to absolute path before cd into TARGET_REPO
if [[ "$PATCH_DIR" != /* ]]; then
    PATCH_DIR="$(cd "$PATCH_DIR" && pwd)"
fi

cd "$TARGET_REPO"
echo -e "${GREEN}Using repository: $(pwd)${NC}"

# Display current repository status
CURRENT_BRANCH=$(git branch --show-current)
CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo -e "Current branch: ${YELLOW}$CURRENT_BRANCH${NC}"
echo -e "Current commit: ${YELLOW}$CURRENT_COMMIT${NC}"

# Check if working directory is clean
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo -e "${YELLOW}Warning: Working directory has uncommitted changes${NC}"
    if [ "$NON_INTERACTIVE" = "true" ]; then
        echo "NON_INTERACTIVE=true, continuing..."
    else
        read -p "Continue applying patches? (y/n): " confirm < /dev/tty
        if [ "$confirm" != "y" ]; then
            echo "Operation cancelled"
            exit 0
        fi
    fi
fi

# Apply all patches
echo -e "\n${GREEN}Starting to apply patches...${NC}\n"

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
FAILED_PATCHES=()

# Collect patch files into array (recursively search subdirectories)

# Use mapfile for bash 4.0+, fallback to while-read for older versions
BASH_MAJOR_VERSION="${BASH_VERSINFO[0]}"
if [ "$BASH_MAJOR_VERSION" -ge 4 ]; then
    # bash 4.0+ supports mapfile (more efficient)
    mapfile -t PATCH_FILES < <(find "$PATCH_DIR" -name "*.patch" -type f | sort)
else
    # bash 3.x fallback (compatible with macOS default bash)
    PATCH_FILES=()
    while IFS= read -r line; do
        PATCH_FILES+=("$line")
    done < <(find "$PATCH_DIR" -name "*.patch" -type f | sort)
fi

# Check if there are any patch files
if [ ${#PATCH_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}No .patch files found${NC}"
    exit 0
fi

TOTAL_PATCHES=${#PATCH_FILES[@]}
CURRENT=0

# Iterate through all patch files
for patch in "${PATCH_FILES[@]}"; do
    ((CURRENT=CURRENT+1))
    
    if [ ! -f "$patch" ]; then
        continue
    fi
    
    patch_name=$(basename "$patch")
    
    # Check if patch file is empty
    if [ ! -s "$patch" ]; then
        echo -e "${YELLOW}⊘${NC} [$CURRENT/$TOTAL_PATCHES] $patch_name (empty file, skipped)"
        ((SKIP_COUNT=SKIP_COUNT+1))
        continue
    fi
    
    echo -n "[$CURRENT/$TOTAL_PATCHES] Applying $patch_name ... "
    
    # Temporarily disable pipefail as git apply warnings cause non-zero exit codes
    set +o pipefail
    
    # Check if patch can be applied
    if git apply --check "$patch" >/dev/null 2>&1; then
        # Apply patch, even with whitespace warnings continue
        # Use || true to ensure we don't exit due to warnings
        git apply --whitespace=fix "$patch" >/dev/null 2>&1 || {
            # If it fails, check if it really failed (not just warnings)
            if git diff --quiet HEAD; then
                # No changes, it really failed
                echo -e "${RED}✗${NC}"
                ((FAIL_COUNT=FAIL_COUNT+1))
                FAILED_PATCHES+=("$patch_name")
                set -o pipefail
                continue
            fi
        }
        echo -e "${GREEN}✓${NC}"
        ((SUCCESS_COUNT=SUCCESS_COUNT+1))
    else
        echo -e "${RED}✗${NC}"
        ((FAIL_COUNT=FAIL_COUNT+1))
        FAILED_PATCHES+=("$patch_name")
    fi
    
    # Restore pipefail
    set -o pipefail
done

# Output statistics
echo -e "\n${GREEN}=== Application Complete ===${NC}"
echo -e "Success: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
[ $SKIP_COUNT -gt 0 ] && echo -e "Skipped: ${YELLOW}$SKIP_COUNT${NC}"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "\n${RED}Failed patches:${NC}"
    for failed in "${FAILED_PATCHES[@]}"; do
        echo "  - $failed"
    done
    echo -e "\n${YELLOW}Tips for handling failed patches:${NC}"
    echo "  1. Check your git diff path in $PATCH_DIR"
    echo "  2. Manually edit and apply"
    exit 1
fi

echo -e "\n${GREEN}All patches applied successfully!${NC}"
echo -e "Repository location: ${BLUE}$(pwd)${NC}"
echo ""