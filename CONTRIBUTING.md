# Contribute
👍🎉 First off, thanks for taking the time to contribute! 🎉👍

Please check out the [Apache Code of Conduct](https://www.apache.org/foundation/policies/conduct.html) first.

We welcome community contributors to LoongForge. Feel free to share your ideas or submit code—help us make LoongForge even better!

Before getting started, please read the following open-source contribution guidelines and adhere to the relevant agreements.

## How to Contribute
We welcome and encourage contributions from the community. Whether it's fixing bugs, adding new features, improving documentation, or sharing ideas, all contributions help make LoongForge better.

## Issues
We use GitHub Issues to track bugs, feature requests, and other public discussions.

### Search Existing Issues First
Before opening a new issue, please search through existing issues to check whether a similar bug report or feature request already exists. This helps avoid duplicates and keeps discussions focused.

### Reporting New Issues
When opening a new issue, please provide as much information as possible, such as:
* A clear and detailed problem description
* Relevant logs or error messages
* Code snippets, screenshots, or videos if applicable

The more context you provide, the easier it will be for maintainers to diagnose and resolve the issue.

## Pull Requests
We strongly welcome pull requests to help improve LoongForge.

All pull requests will be reviewed by the maintainers. Automated checks and tests will be run as part of the review process. Once all checks pass and the review is approved, the pull request will be accepted. Please note that merging into the `main` branch may not happen immediately and could be subject to scheduling.

### Repository Structure

LoongForge manages its dependencies using two strategies:

| Repository | Strategy | Where changes land |
|---|---|---|
| **LoongForge** (this repo) | fork → PR | `baidu-baige/LoongForge` |
| **Loong-Megatron** | fork → PR | `baidu-baige/Loong-Megatron` |
| **TransformerEngine** | patch files | `patches/TransformerEngine_<tag>/` in LoongForge |

### Step 0 — Fork the repositories

Fork the repositories you intend to modify on GitHub:

```
baidu-baige/LoongForge        →  your-name/LoongForge
baidu-baige/Loong-Megatron   →  your-name/Loong-Megatron   # only if modifying Megatron
```

### Step 1 — Clone your fork and initialize submodules

```bash
git clone --recurse-submodules https://github.com/your-name/LoongForge.git
cd LoongForge

# Add the official repo as upstream
git remote add upstream https://github.com/baidu-baige/LoongForge.git
```

### Step 2 — Configure Loong-Megatron remotes (only if modifying Megatron)

```bash
cd third_party/Loong-Megatron
# origin already points to baidu-baige/Loong-Megatron via the submodule config
git remote add my-fork https://github.com/your-name/Loong-Megatron.git

# Verify
git remote -v
# origin    https://github.com/baidu-baige/Loong-Megatron.git (fetch)
# my-fork   https://github.com/your-name/Loong-Megatron.git   (fetch)
cd ../..
```

### Step 3 — Create a development branch

```bash
# LoongForge
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# Loong-Megatron (only if modifying Megatron)
cd third_party/Loong-Megatron
git checkout loong-main/core_v0.15.0
git pull origin loong-main/core_v0.15.0
git checkout -b feature/your-feature-name
cd ../..
```

### Step 4 — Develop and commit your changes

Make your changes, then commit:

```bash
git add .
git commit -m "feat: add your commit message"
```

### Step 5 — Sync with upstream and push to your fork

```bash
# (Optional but recommended) Rebase on the latest upstream main before pushing
git pull --rebase upstream main
git push -u origin feature/your-feature-name
```

For Loong-Megatron changes, push to your Megatron fork instead:

```bash
cd third_party/Loong-Megatron
git pull --rebase origin loong-main/core_v0.15.0
git push -u my-fork feature/your-feature-name
```

### Step 6 — Create a Pull Request

Open a PR on GitHub from your feature branch to the target upstream branch:

- **LoongForge changes**: `your-name/LoongForge:feature/xxx` → `baidu-baige/LoongForge:main`
- **Megatron changes**: `your-name/Loong-Megatron:feature/xxx` → `baidu-baige/Loong-Megatron:loong-main/core_v0.15.0`
- **TE changes**: commit the patch file to LoongForge, then open a PR as in the LoongForge flow above

---

### Pre-Submission Checklist

Before submitting a pull request, please make sure that:

1. You create your branch from the correct base branch (`main` for LoongForge, `loong-main/core_v0.15.0` for Loong-Megatron).
2. You update relevant code comments or documentation if APIs are changed.
3. You add the appropriate copyright and license notice to the top of any new source files when applicable, and preserve upstream notices for third-party derived files.
4. For original source files, prefer using the SPDX-based Apache-2.0 header described in the project guidelines.
5. Your code passes linting and style checks.
6. Your changes are fully tested.
7. You submit the pull request against the correct development branch as required.

## License
By contributing to LoongForge, you agree that your original contributions will be licensed under the [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE).

Please note that some files in this repository include or are derived from third-party open-source projects. For such files, contributors must retain the original copyright, license, and attribution notices required by the upstream project, and add modification notices where appropriate. See the corresponding file headers for additional details.

For practical file header templates and examples, please refer to our **[License and File Header Guidelines](https://loongforge.readthedocs.io/en/latest/HEADER_GUIDELINES.html)**.
