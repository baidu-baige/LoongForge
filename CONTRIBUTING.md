# Contribute
👍🎉 First off, thanks for taking the time to contribute! 🎉👍

Please check out the [Apache Code of Conduct](https://www.apache.org/foundation/policies/conduct.html) first.

We welcome community contributors to OmniTraining. Feel free to share your ideas or submit code—help us make OmniTraining even better!

Before getting started, please read the following open-source contribution guidelines and adhere to the relevant agreements.

## How to Contribute
We welcome and encourage contributions from the community. Whether it's fixing bugs, adding new features, improving documentation, or sharing ideas, all contributions help make OmniTraining better.

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
We strongly welcome pull requests to help improve OmniTraining. 

All pull requests will be reviewed by the maintainers. Automated checks and tests will be run as part of the review process. Once all checks pass and the review is approved, the pull request will be accepted. Please note that merging into the `main` branch may not happen immediately and could be subject to scheduling.

### Standard GitHub Workflow

1. **Fork** the OmniTraining repository to your own GitHub account.
2. **Clone** your fork to your local machine:
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/OmniTraining.git
   cd OmniTraining
   ```
3. **Create a new branch** from `main` for your changes:
   ```bash
   git checkout -b dev_your_feature_branch
   ```
4. **Add and commit** your changes:
   ```bash
   git add .
   git commit -m "feat: add your commit message"
   ```
5. **Sync with upstream and push** to your fork:
   ```bash
   # (Optional but recommended) Pull latest from upstream main before pushing
   git pull --rebase [https://github.com/baidu-baige/OmniTraining.git](https://github.com/baidu-baige/OmniTraining.git) main
   git push -u origin dev_your_feature_branch
   ```
6. **Create a Pull Request** on GitHub from your branch to the original OmniTraining `main` branch.

### Pre-Submission Checklist

Before submitting a pull request, please make sure that:

1. You create your branch from `main`.
2. You update relevant code comments or documentation if APIs are changed.
3. You add the appropriate copyright and license notice to the top of any new source files when applicable, and preserve upstream notices for third-party derived files.
4. For original source files, prefer using the SPDX-based Apache-2.0 header described in the project guidelines.
5. Your code passes linting and style checks.
6. Your changes are fully tested.
7. You submit the pull request against the correct development branch as required.

## License
By contributing to OmniTraining, you agree that your original contributions will be licensed under the [Apache License 2.0](https://github.com/baidu/OmniTraining/blob/master/LICENSE).

Please note that some files in this repository include or are derived from third-party open-source projects. For such files, contributors must retain the original copyright, license, and attribution notices required by the upstream project, and add modification notices where appropriate. See the corresponding file headers for additional details.

For practical file header templates and examples, please refer to our **[License and File Header Guidelines](docs/source/get_started/license_guidelines.md)**.