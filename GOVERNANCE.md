# Governance

LoongForge is maintained by the Baidu AI Cloud Baige team with contributions
from the broader community. This document describes technical ownership,
decision-making, releases, and maintenance expectations.

## Roles and ownership

### Maintainers

The named core maintainers are [@nullnonenilNULL](https://github.com/nullnonenilNULL)
and [@Zachary-wW](https://github.com/Zachary-wW). The
`@baidu-baige/loongforge-team` GitHub team is the broader repository review
group, and the core maintainers, team, and current owners are listed together
in [CODEOWNERS](.github/CODEOWNERS).

Maintainers review and merge changes, maintain CI, triage issues and security
reports, and coordinate releases.

### Appointment and retirement

Contributors interested in ongoing maintenance can contact the project team.
Current maintainers agree on appointments and removals by consensus. A
maintainer may retire by notifying the team. The `baidu-baige` organization
updates GitHub team membership when a maintainer joins or leaves.

### Contributors

Anyone may open issues, propose changes, improve documentation, or report
regressions. The contribution workflow and testing expectations are described
in [CONTRIBUTING.md](CONTRIBUTING.md).

## Technical decisions

Proposals start as issues or pull requests. For changes that affect
compatibility or training behavior, describe the problem, design, impact,
test results, and documentation updates. Maintainers review changes through
the pull request process, discuss trade-offs in the issue or pull request,
and approve merges to the protected default branch after required checks pass.

## Releases and maintenance

Maintainers review the release notes and validation results, confirm the
version and tag match, and coordinate publication. The release process is
documented in the [workflow guide](.github/workflows/README.md#releases).

Maintainers review issues and pull requests, update supported-model and
compatibility documentation, and publish fixes through the release process.
LoongForge commits to at least 12 months of active maintenance and to validating
the CPU compatibility of its Python package against the latest patch release in
each of the two newest stable PyTorch release series, for the Python versions
supported by each PyTorch release. Full accelerator and training validation is
provided separately by the documented Docker and hardware CI paths.

## Hardware validation status

LoongForge supports NVIDIA GPU and Kunlun XPU workflows. GitHub CI runs CPU
checks and can run GPU regressions on self-hosted runners. CI has no Kunlun
XPU runner, so the repository's XPU paths are outside automated regression
coverage. Hardware-specific validation results are welcome in pull requests.

## Related policies

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)
- [Contributing Guidelines](CONTRIBUTING.md)
