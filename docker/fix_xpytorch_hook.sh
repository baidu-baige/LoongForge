#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Fix xpytorch_import_hook.py and torch_xray_import_hook.py
# Auto-detect pip isolated build environment and skip hook

set -e

echo "Patching xpytorch and xray import hooks..."

python3 << 'EOFPY'
import os
import re
import site
import sys

patched_count = 0

# ======================== Fix xpytorch_import_hook.py ========================
hook_files = []
for path in site.getsitepackages():
    candidate = os.path.join(path, "xpytorch_import_hook.py")
    if os.path.exists(candidate):
        hook_files.append(candidate)

if not hook_files:
    print("xpytorch_import_hook.py not found, skipping patch")
else:
    for hook_file in hook_files:
        with open(hook_file, "r") as f:
            content = f.read()
        
        if "# patched: auto-detect pip isolated env" in content:
            print(f"[xpytorch] Already patched: {hook_file}")
            patched_count += 1
            continue
        
        # Fix 1: Add pip isolated environment detection at the beginning of _custom_import
        old_func_start = r"def _custom_import\(module_name, globals=None, locals=None, fromlist=\(\), level=0\):\n( *)global SYMBOL_REWRITE_REGISTER"
        
        def make_func_patch(m):
            indent = m.group(1)
            detection_code = (
                f"{indent}# patched: auto-detect pip isolated build environment\n"
                f"{indent}if any('pip-' in p or 'pip_' in p for p in sys.path):\n"
                f"{indent}    return builtins.__origin__import__(module_name, globals, locals, fromlist, level)\n"
            )
            return (
                "def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):\n"
                + detection_code +
                f"{indent}global SYMBOL_REWRITE_REGISTER"
            )
        
        content, count1 = re.subn(old_func_start, make_func_patch, content)
        
        # Fix 1 is critical, fail if not matched
        if count1 == 0:
            print(f"ERROR: Function pattern not found in {hook_file}", file=sys.stderr)
            print("This is a critical patch. Build cannot continue.", file=sys.stderr)
            sys.exit(1)
        
        # Fix 2: torch_version = version('torch') with fallback
        old_pattern2 = r"^( *)torch_version = version\('torch'\)"
        
        def make_patch2(m):
            indent = m.group(1)
            return (
                f"{indent}try:\n"
                f"{indent}    torch_version = version('torch')\n"
                f"{indent}except Exception:\n"
                f"{indent}    try:\n"
                f"{indent}        torch_version = __import__('torch').__version__  # patched\n"
                f"{indent}    except ImportError:\n"
                f"{indent}        torch_version = \"0.0.0\"  # patched fallback"
            )
        
        content, count2 = re.subn(old_pattern2, make_patch2, content, flags=re.MULTILINE)
        
        with open(hook_file, "w") as f:
            f.write(content)
        
        patched_count += 1
        print(f"[xpytorch] Patched: {hook_file} (func={count1}, version={count2})")

    # Check if at least one file was patched
    if patched_count == 0:
        print("ERROR: No xpytorch_import_hook.py files were successfully patched", file=sys.stderr)
        sys.exit(1)

# ======================== Fix torch_xray_import_hook.py ========================
xray_files = []
for path in site.getsitepackages():
    candidate = os.path.join(path, "torch_xray_import_hook.py")
    if os.path.exists(candidate):
        xray_files.append(candidate)

xray_patched_count = 0

if not xray_files:
    print("torch_xray_import_hook.py not found, skipping patch")
else:
    for xray_file in xray_files:
        with open(xray_file, "r") as f:
            content = f.read()
        
        if "if spec is not None and spec.loader is not None:" in content:
            print(f"[xray] Already patched: {xray_file}")
            xray_patched_count += 1
            continue
        
        old_pattern = r"^( *)spec\.loader = XrayMetaPathLoader\(spec\.loader\)"
        
        def make_xray_patch(m):
            indent = m.group(1)
            return (
                f"{indent}if spec is not None and spec.loader is not None:\n"
                f"{indent}    spec.loader = XrayMetaPathLoader(spec.loader)"
            )
        
        content, count = re.subn(old_pattern, make_xray_patch, content, flags=re.MULTILINE)
        
        if count == 0:
            print(f"ERROR: Pattern not found in {xray_file}", file=sys.stderr)
            print("This is a critical patch. Build cannot continue.", file=sys.stderr)
            sys.exit(1)
        
        with open(xray_file, "w") as f:
            f.write(content)
        
        xray_patched_count += 1
        print(f"[xray] Patched: {xray_file}")

    # Check if at least one xray file was patched
    if xray_patched_count == 0:
        print("ERROR: No torch_xray_import_hook.py files were successfully patched", file=sys.stderr)
        sys.exit(1)

print("All import hook fixes applied successfully!")
EOFPY
