"""Setup BaigeOmni environment.

Dependency management strategy:
  - Megatron-LM : managed as a git submodule (third_party/Baige-Megatron),
                  version is locked by the submodule commit pointer — no patching needed.
  - TransformerEngine : cloned from upstream NVIDIA repo, checked out at the
                        specified tag, then patched from patches/TransformerEngine_<tag>/.
"""
import os
import subprocess
import sys
import argparse


def run_command(command, cwd=None, shell=True, check=True, env=None):
    """Run a shell command."""
    print(f"Running: {command} in {cwd if cwd else os.getcwd()}")
    try:
        subprocess.check_call(command, cwd=cwd, shell=shell, env=env)
        return True
    except subprocess.CalledProcessError:
        if check:
            print(f"Error executing command: {command}")
            sys.exit(1)
        return False


def main():
    """main process"""
    parser = argparse.ArgumentParser(
        description="Setup BaigeOmni development environment.")
    parser.add_argument("--te-tag", required=True,
                        help="Tag for TransformerEngine (e.g., v2.9)")
    parser.add_argument("--workspace", default=os.getcwd(),
                        help="Workspace root directory")

    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    te_path = os.path.join(workspace, "TransformerEngine")
    omni_path = os.path.join(workspace, "BaigeOmni")

    # Adjust omni_path if script is running inside BaigeOmni
    if os.path.basename(workspace) == "BaigeOmni":
        omni_path = workspace
        workspace = os.path.dirname(workspace)
        te_path = os.path.join(workspace, "TransformerEngine")

    # Megatron-LM lives as a submodule inside the main repo
    megatron_path = os.path.join(omni_path, "third_party", "Baige-Megatron")

    print(f"Workspace              : {workspace}")
    print(f"Megatron Path       : {megatron_path}  (submodule)")
    print(f"TransformerEngine Path : {te_path}")
    print(f"BaigeOmni Path         : {omni_path}")

    # 1. Initialize Megatron-LM submodule
    # Version is locked by the submodule commit pointer in BaigeOmni — no patching needed.
    print("\n[1/5] Initializing Megatron submodule...")
    run_command("git submodule update --init third_party/Baige-Megatron", cwd=omni_path)

    # 2. Clone TransformerEngine from upstream
    print("\n[2/5] Setting up TransformerEngine...")
    if not os.path.exists(te_path):
        run_command(
            f"git clone https://github.com/NVIDIA/TransformerEngine.git {te_path}")
    else:
        print(f"TransformerEngine already exists at {te_path}")

    # 3. Checkout TransformerEngine tag
    print(f"Checking out TransformerEngine tag: {args.te_tag}")
    run_command("git fetch --all --tags", cwd=te_path)

    branch_name = f"baige_{args.te_tag}"
    if run_command(f"git rev-parse --verify {branch_name}", cwd=te_path, shell=True, check=False):
        print(f"Branch {branch_name} already exists, checking it out.")
        run_command(f"git checkout {branch_name}", cwd=te_path)
    else:
        print(f"Creating branch {branch_name} from tag {args.te_tag}...")
        run_command(f"git checkout {args.te_tag} -b {branch_name}", cwd=te_path)

    run_command("git restore .", cwd=te_path)

    # 4. Apply patches to TransformerEngine
    # Patch directory is named after the base tag, e.g. patches/TransformerEngine_v2.9/
    print("\n[3/5] Applying patches to TransformerEngine...")
    apply_script = os.path.join(omni_path, "patches/apply_patches.sh")
    if not os.path.exists(apply_script):
        print(f"Error: Patch script not found at {apply_script}")
        sys.exit(1)

    te_patch_dir = os.path.join(omni_path, f"patches/TransformerEngine_{args.te_tag}")
    if not os.path.isdir(te_patch_dir):
        print(f"Error: TE patch directory not found: {te_patch_dir}")
        sys.exit(1)

    run_command(f"bash {apply_script} {te_patch_dir} {te_path}")

    # 5. Build and install TransformerEngine
    print("\n[4/5] Building and installing TransformerEngine...")
    run_command("git submodule update --init --recursive", cwd=te_path)

    env = os.environ.copy()
    env["NVTE_FRAMEWORK"] = "pytorch"
    run_command("pip3 install --no-build-isolation .", cwd=te_path, env=env)

    # 6. Install BaigeOmni dependencies
    print("\n[5/5] Installing BaigeOmni dependencies...")
    run_command("pip install -r requirements.txt", cwd=omni_path)

    print("\nSetup completed successfully!")
    print(f"  Megatron : {megatron_path}")
    print(f"  TransformerEngine : {te_path}  (tag {args.te_tag}, branch {branch_name})")


if __name__ == "__main__":
    main()
