# Governance

LoongForge is maintained by the Baidu AI Cloud Baige team with contributions from
the broader community. This document defines the project's technical ownership,
decision process, and release responsibilities.

## Roles

### Maintainers

The `@baidu-baige/loongforge-maintainers` GitHub team owns the LoongForge
repository. Maintainers are responsible for:

- reviewing and merging changes;
- maintaining repository rules and required checks;
- triaging issues and security reports;
- approving releases;
- keeping release and compatibility documentation current.

The project keeps at least two active maintainers. Maintainers act by consensus
on GitHub. If consensus cannot be reached, the Baidu AI Cloud Baige team lead
makes the final decision and records the rationale in the relevant issue or pull
request.

### Contributors

Anyone may open issues, propose changes, improve documentation, or report
regressions. Contribution workflow and testing expectations are described in
[CONTRIBUTING.md](CONTRIBUTING.md).

### Maintainer changes

Existing maintainers add or remove maintainers by consensus. Membership is
managed through the `@baidu-baige/loongforge-maintainers` GitHub team so ownership
remains visible and auditable. Retired committers are listed below.

| GitHub handle | Role | Status |
|---|---|---|
| _None recorded_ | Retired committer | No retired committers have been recorded yet |

## Technical decisions

Proposals start as issues or pull requests. Non-trivial changes explain the
problem, design, compatibility impact, testing, and documentation updates. A
change may be merged after:

1. the required checks pass;
2. accelerator-sensitive changes receive the applicable GPU validation;
3. unresolved review threads are addressed; and
4. at least one maintainer approves the change.

Self-approval does not satisfy maintainer review. The default branch is protected
and requires an approving review before merge.

## Release governance

Releases use immutable `vX.Y.Z` tags. The release owner ensures that the package
version, tag, documentation, and validation evidence agree before tagging. See
the release section in [README.md](README.md) and
[GitHub Actions documentation](.github/workflows/README.md) for the release
methodology, cadence, and criteria.

## Support commitment

Maintainers intend to support LoongForge for at least the next 12 months.
Compatibility work is planned and tracked publicly in GitHub issues, including
validation against the PyTorch runtimes used by the supported NVIDIA and Kunlun
XPU environments.
