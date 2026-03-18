"""Setup BaigeOmni environment."""
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
    except subprocess.CalledProcessError as e:
        if check:
            print(f"Error executing command: {command}")
            sys.exit(1)
        return False


def main():
    """main process"""
    parser = argparse.ArgumentParser(
        description="Setup BaigeOmni development environment.")
    parser.add_argument("--megatron-tag", required=True,
                        help="Tag for Megatron-LM (e.g., core_v0.15.0)")
    parser.add_argument("--te-tag", required=True,
                        help="Tag for TransformerEngine (e.g., v2.9)")
    parser.add_argument("--workspace", default=os.getcwd(),
                        help="Workspace root directory")

    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    megatron_path = os.path.join(workspace, "Megatron-LM")
    te_path = os.path.join(workspace, "TransformerEngine")
    # Assuming this script is run from outside or we adjust paths
    omni_path = os.path.join(workspace, "BaigeOmni")

    # Adjust omni_path if script is running inside BaigeOmni
    if os.path.basename(workspace) == "BaigeOmni":
        omni_path = workspace
        workspace = os.path.dirname(workspace)
        megatron_path = os.path.join(workspace, "Megatron-LM")
        te_path = os.path.join(workspace, "TransformerEngine")

    print(f"Workspace: {workspace}")
    print(f"Megatron-LM Path: {megatron_path}")
    print(f"TransformerEngine Path: {te_path}")
    print(f"BaigeOmni Path: {omni_path}")

    # 1. Clone Repositories
    if not os.path.exists(megatron_path):
        run_command(
            f"git clone https://github.com/NVIDIA/Megatron-LM.git {megatron_path}")
    else:
        print(f"Megatron-LM already exists at {megatron_path}")

    if not os.path.exists(te_path):
        run_command(
            f"git clone https://github.com/NVIDIA/TransformerEngine.git {te_path}")
    else:
        print(f"TransformerEngine already exists at {te_path}")

    # 2. Checkout Tags
    print(f"Checking out Megatron-LM tag: {args.megatron_tag}")
    run_command("git fetch --all --tags", cwd=megatron_path)

    # Check if branch baige_{tag} exists
    if run_command(f"git rev-parse --verify baige_{args.megatron_tag}", cwd=megatron_path, shell=True, check=False):
        print(
            f"Branch baige_{args.megatron_tag} already exists, checking it out.")
        run_command(
            f"git checkout baige_{args.megatron_tag}", cwd=megatron_path)
    else:
        print(
            f"Creating branch baige_{args.megatron_tag} from {args.megatron_tag}...")
        run_command(
            f"git checkout {args.megatron_tag} -b baige_{args.megatron_tag}", cwd=megatron_path)

    run_command("git restore .", cwd=megatron_path)

    print(f"Checking out TransformerEngine tag: {args.te_tag}")
    run_command("git fetch --all --tags", cwd=te_path)

    if run_command(f"git rev-parse --verify baige_{args.te_tag}", cwd=te_path, shell=True, check=False):
        print(f"Branch baige_{args.te_tag} already exists, checking it out.")
        run_command(f"git checkout baige_{args.te_tag}", cwd=te_path)
    else:
        print(f"Creating branch baige_{args.te_tag} from {args.te_tag}...")
        run_command(
            f"git checkout {args.te_tag} -b baige_{args.te_tag}", cwd=te_path)

    run_command("git restore .", cwd=te_path)

    # 3. Apply Patches
    apply_script = os.path.join(
        omni_path, "tools/apply_patches/apply_two_repo.sh")
    if not os.path.exists(apply_script):
        print(f"Error: Patch script not found at {apply_script}")
        sys.exit(1)

    megatron_patch_dir = os.path.join(omni_path, "patches/Megatron-LM")
    te_patch_dir = os.path.join(omni_path, "patches/TransformerEngine")

    print("Applying patches to Megatron-LM...")
    run_command(f"bash {apply_script} {megatron_patch_dir} {megatron_path}")

    print("Applying patches to TransformerEngine...")
    run_command(f"bash {apply_script} {te_patch_dir} {te_path}")

    # 4. Build and Install TransformerEngine
    print("Building and Installing TransformerEngine...")
    run_command("git submodule update --init --recursive", cwd=te_path)

    # Set environment variable for build
    env = os.environ.copy()
    env["NVTE_FRAMEWORK"] = "pytorch"

    # Using pip install .
    run_command("pip3 install --no-build-isolation .", cwd=te_path, env=env)

    # 5. Install BaigeOmni dependencies
    print("Installing BaigeOmni dependencies...")
    run_command("pip install -r requirements.txt", cwd=omni_path)

    print("\nSetup completed successfully!")


if __name__ == "__main__":
    main()
