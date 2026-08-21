# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Rule definitions for the sensitive-information scanner.

Kept as a Python module (rather than JSON/TOML) so that patterns can use raw
strings without double-escaping, and so that each rule can carry an inline
rationale. Consumed by ``ci/sensitive_scan.py``.

Severity contract:
    error - blocks CI. Directly identifies internal infrastructure, a person,
            or credential material.
    warn  - reported but does not block. Usually a naming or path-hygiene
            issue: real, but high-volume and not individually disclosive.

To silence a single line, append a trailing comment:
    sensitive-scan: allow                  (all rules on that line)
    sensitive-scan: allow[rule-id,rule-id]  (specific rules)

To silence a whole path or pattern, add an entry to ALLOWLIST below with a
``reason``. Entries without a reason are rejected by the scanner.
"""

CONFIG = {
    # Paths never scanned. Matched with fnmatch against the repo-relative path.
    "exclude_paths": [
        ".git/*",
        "third_party/*",
        "output/*",
        "outputs/*",
        # Model/training data: Chinese SFT corpora legitimately contain words
        # such as 机密 and long digit strings that would swamp the report.
        "tests/llm_vlm/datasets/*",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.svg",
        "*.ico",
        "*.mp4",
        "*.pdf",
        "*.whl",
        "*.tar.gz",
        "*.pt",
        "*.bin",
        "*.safetensors",
        "*.DS_Store",
    ],
    # Cap findings printed per rule so a high-volume warn rule cannot drown the
    # report. Full counts are always shown in the summary.
    "max_findings_per_rule": 20,
}

RULES = [
    # ---------------------------------------------------------------- error --
    {
        "id": "internal-domain",
        "severity": "error",
        "title": "Internal-only hostname",
        "why": "Corporate intranet hosts are unreachable externally and expose "
               "internal tooling topology.",
        "pattern": r"(?i)\b[\w.-]*\.baidu-int\.com\b"
                   r"|\biregistry\.[\w.-]+\b"
                   r"|\bicode\.baidu\.com\b"
                   r"|\b(?:ku|wiki|note|agile|icafe|newicafe|noah)\.baidu(?:-int)?\.com\b",
        "hint": "Point at public documentation, or drop the reference entirely.",
    },
    {
        "id": "internal-package-mirror",
        "severity": "error",
        "title": "Internal package registry or mirror",
        "why": "Hardcoded internal mirrors make builds fail for external users "
               "and leak the internal build chain.",
        "pattern": r"(?i)\b(?:registry|mirrors)\.baidubce\.com\b"
                   r"|\bpip\.baidu(?:-int)?\.com\b"
                   r"|--index-url\s+\S*baidu\S*",
        "hint": "Use the public index (pypi.org / nvcr.io / docker.io), or make "
                "it an overridable build arg.",
    },
    {
        "id": "private-object-storage",
        "severity": "error",
        "title": "Private object-storage location",
        "why": "Private bucket prefixes are not publicly readable and disclose "
               "the internal artifact-distribution layout.",
        "pattern": r"(?i)\bbos:/\S+"
                   r"|\baihc-private-[\w-]+"
                   r"|\baiak[_-]share\b",
        "hint": "Replace with a public URL or a documented placeholder such as "
                "bos:/path/to/<artifact>/.",
    },
    {
        "id": "corp-email",
        "severity": "error",
        "title": "Corporate email address",
        "why": "Directly identifies an employee.",
        "pattern": r"(?i)\b[a-z0-9._%+-]+@(?:baidu|baidu-int|baidubce)\.com\b",
        "hint": "Remove, or replace with a role address / 'The LoongForge Authors'.",
        "redact": True,
    },
    {
        "id": "developer-home-path",
        "severity": "error",
        "title": "Developer home directory",
        "why": "A /home/<login>/ path names an individual and never resolves on "
               "another machine.",
        "pattern": r"/home/(?!opt/|user/|users/|ubuntu/|runner/|admin/|work/)"
                   r"[a-z][a-z0-9_.-]{2,}/",
        "hint": "Use a generic root such as /workspace/ or an env-var override.",
    },
    {
        "id": "secret-material",
        "severity": "error",
        "title": "Key material or provider token",
        "why": "Live credential material.",
        "pattern": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
                   r"|\bhf_[A-Za-z0-9]{20,}\b"
                   r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
                   r"|\bglpat-[A-Za-z0-9_-]{20,}\b"
                   r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
                   r"|\bsk-[A-Za-z0-9]{20,}\b"
                   r"|\bAKIA[0-9A-Z]{16}\b",
        "hint": "Revoke the credential, then read it from the environment or a "
                "secret store.",
        "redact": True,
    },
    {
        "id": "credential-assignment",
        "severity": "error",
        "title": "Credential assigned to a literal value",
        "why": "AK/SK/token/password bound to an inline literal.",
        # Requires the value to be a real literal: either quoted, or a
        # space-free shell assignment. The lookaheads demand at least one digit
        # and one letter, which excludes code expressions (`sk = index_k.size(0)`,
        # `token = os.environ.get(...)`) and obvious placeholders (`xxx`,
        # `your-token`).
        "pattern": r"(?i)\b(?:ak|sk|access_?key(?:_id)?|secret_?(?:access_)?key"
                   r"|api_?key|passwo?rd|passwd|auth_?token|token)\b"
                   r"(?:"
                   r"\s*[:=]\s*\"(?=[^\"]*\d)(?=[^\"]*[A-Za-z])[A-Za-z0-9/+=_.-]{12,}\""
                   r"|\s*[:=]\s*'(?=[^']*\d)(?=[^']*[A-Za-z])[A-Za-z0-9/+=_.-]{12,}'"
                   r"|=(?=[^\s\"']*\d)(?=[^\s\"']*[A-Za-z])[A-Za-z0-9/+=_-]{12,}\b"
                   r")",
        "hint": "Read from the environment instead of committing the value.",
        "redact": True,
    },
    {
        "id": "internal-network-address",
        "severity": "error",
        "title": "RFC1918 address",
        "why": "A private-range address identifies an internal host and is "
               "useless (or actively misleading) outside the corporate network.",
        "pattern": r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))"
                   r"\.\d{1,3}\.\d{1,3}\.\d{1,3}\b(?::\d{1,5})?"
                   r"|\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))"
                   r"\.\d{1,3}\.\d{1,3}\b:\d{1,5}",
        "hint": "Use localhost, a DNS name, or an env-var override.",
    },
    {
        "id": "hardcoded-proxy",
        "severity": "error",
        "title": "Hardcoded proxy endpoint",
        "why": "Proxy addresses reveal internal egress infrastructure.",
        "pattern": r"(?i)\b(?:https?_proxy|all_proxy)\s*=\s*[\"']?"
                   r"(?:https?://)?(?!\$|\{|<|your-|127\.0\.0\.1|localhost)"
                   r"[a-z0-9][\w.-]*(?::\d+)?",
        "hint": "Expose it as ${http_proxy} and let the caller supply the value.",
    },
    {
        "id": "confidentiality-marker",
        "severity": "error",
        "title": "Confidentiality / do-not-publish marker",
        "why": "Explicit internal-only markings must not survive into a public "
               "repository.",
        "pattern": r"(?i)Private\s*::\s*Do Not Upload"
                   r"|\bConfidential\b"
                   r"|\bInternal Use Only\b"
                   r"|\bProprietary and Confidential\b"
                   r"|机密|绝密|内部资料|不打算开源",
        "hint": "Remove the marker and confirm the file is cleared for release.",
    },
    # ----------------------------------------------------------------- warn --
    {
        "id": "user-scoped-path",
        "severity": "warn",
        "title": "Per-user directory in a path",
        "why": "A /users/<name>/ segment names a person or team account.",
        "pattern": r"(?i)/users?/[a-z][a-z0-9_.-]{2,}/",
        "hint": "Drop the user segment or move it behind an env-var override.",
    },
    {
        "id": "internal-cluster-path",
        "severity": "warn",
        "title": "Internal cluster filesystem path",
        "why": "Collectively these describe the internal cluster layout. Usually "
               "harmless ${VAR:-default} values, but they should not be the "
               "shipped defaults.",
        "pattern": r"(?i)(?:^|[\s\"'=:(\[])/(?:mnt/(?:cluster|rapidfs|cfs[\w-]*|data)"
                   r"|ssd\d)(?:/|\b)",
        "hint": "Prefer /workspace/... or a documented placeholder as the default.",
    },
    {
        "id": "legacy-internal-name",
        "severity": "warn",
        "title": "Pre-rename internal project name",
        "why": "Leftover internal names leak the rename history and, when used "
               "as a path or symbol default, break external users.",
        "pattern": r"(?i)\bAIAK[-_](?:Training[-_](?:Omni|LLM)|Megatron)\b"
                   r"|\bBaigeOmni\b"
                   r"|\baiak_training_omni\b"
                   r"|\bset_aiak_\w+|\binitialize_baige_\w+|\bUSE_AIAK_\w+"
                   r"|\baiak-ckpt\b",
        "hint": "Rename to the public equivalent (LoongForge / Loong-Megatron).",
    },
    {
        "id": "internal-doc-reference",
        "severity": "warn",
        "title": "Reference to an internal-only document or library",
        "why": "Dangling pointers to internal docs are unactionable externally.",
        "pattern": r"(?i)<\s*Baige[^>]*>"
                   r"|internal Baige\b"
                   r"|\[internal use only\]"
                   r"|内部文档|内网文档|详见\s*wiki",
        "hint": "Inline the needed information or remove the pointer.",
    },
    {
        "id": "internal-codename",
        "severity": "warn",
        "title": "Internal hardware or product codename",
        "why": "Non-public chip/cluster/product codenames disclose the internal "
               "hardware fleet and roadmap.",
        "pattern": r"(?i)\bBZZ\d?\b|\bP6K\b|\bDECK_STD\w*\b|\bqianfan\b|\bwenxin\b",
        "hint": "Use a public model designation, or parameterize it.",
    },
    {
        "id": "attributed-todo",
        "severity": "warn",
        "title": "TODO/FIXME attributed to a named handle",
        "why": "Handles may be corporate logins. Upstream handles inherited from "
               "vendored code are fine and belong in the allowlist.",
        "pattern": r"\b(?:TODO|FIXME|XXX|HACK)\(\s*([A-Za-z][\w.-]{2,})\s*\)",
        "hint": "Drop the handle, or allowlist it if it came from upstream code.",
    },
    {
        "id": "internal-ticket-id",
        "severity": "warn",
        "title": "Internal issue-tracker identifier",
        "why": "Ticket IDs disclose the internal tracker and numbering scheme.",
        "pattern": r"(?i)\baiak-train-\d+\b|\bicafe/\S+|\bhac-aiacc\b",
        "hint": "Remove, or restate the change without the ticket reference.",
    },
]

# Each entry needs: rule, path (fnmatch glob), reason.
# Optional: match (regex applied to the matched text; when absent, every match
# under `path` is allowed).
ALLOWLIST = [
    {
        "rule": "legacy-internal-name",
        "path": "README*.md",
        "match": r"AIAK-Training-LLM",
        "reason": "Deliberate public attribution of the former product name.",
    },
    {
        "rule": "legacy-internal-name",
        "path": "docs/source*/get_started/README.md",
        "match": r"AIAK-Training-LLM",
        "reason": "Same public attribution, mirrored into the docs site.",
    },
    {
        "rule": "internal-codename",
        "path": "*",
        "match": r"(?i)Qianfan-VL",
        "reason": "Link to the public Qianfan-VL model release on GitHub.",
    },
    {
        "rule": "private-object-storage",
        "path": "*",
        "match": r"bos:/path/to/",
        "reason": "Documented placeholder, not a real bucket.",
    },
    {
        "rule": "private-object-storage",
        "path": ".github/workflows/*.yml",
        "match": r"bos:/\$\{?BOS_BUCKET",
        "reason": "Bucket name comes from a repository variable, not from the source.",
    },
    {
        "rule": "*",
        "path": "ci/sensitive_rules.py",
        "reason": "Rule patterns necessarily contain the strings they match.",
    },
    {
        "rule": "*",
        "path": "skills/sensitive-scan/SKILL.md",
        "reason": "Skill documentation quotes example findings.",
    },
]
