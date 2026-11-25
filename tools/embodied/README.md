# Embodied Architecture Tools

Two command-line tools used while porting a legacy world model onto the
five-layer architecture under `loongforge/embodied/model/arch/`.

Both are read-only and neither is imported by training code. Run them from the
repository root with `PYTHONPATH=.`.

| Tool | Answers |
| --- | --- |
| `arch_probe` | What does this spec actually resolve to? |
| `code_checks` | Does this source tree still satisfy the three hand-written rules? |

## `arch_probe` — read a spec out loud

The providers are real, with their widths and geometry shrunk, so a probe runs
in a few seconds on CPU. Topology is **not** shrunk: the condition table, the
visibility rules, the head table, and the clean prefix are whatever the
provider declares.

```bash
# every state_dict key with its shape, plus the geometry and the tables
PYTHONPATH=. python -m tools.embodied.arch_probe keys vpp2

# the shapes the entry stage produces, per stream and per context slot
PYTHONPATH=. python -m tools.embodied.arch_probe entry fastwam

# the per-token timestep, next to the two legacy formulas it reproduces
PYTHONPATH=. python -m tools.embodied.arch_probe timestep fastwam
```

Pick a subcommand by the question you have:

- **`keys`** — reach for this first when `load_state_dict(strict=True)` reports
  a missing or unexpected key. It prints what the new side *wants* to be given,
  which is the other half of the comparison the error message gives you.
- **`entry`** — hidden, modulation, timestep embedding, RoPE table, and the
  projected cross-attention keys. Use it when a parity harness disagrees at the
  entry rather than inside the stack.
- **`timestep`** — the per-token tensor a stream expands a `[B]` draw into.
  Both legacy formulas are reproduced inline, so this runs without either
  legacy checkout on the path.

Not every declared head has an implementation yet. `internw0_delta`'s
`track_align` raises `NotImplementedError` at build time on purpose, so the two
probes that assemble a model have to be told to leave it out:

```bash
PYTHONPATH=. python -m tools.embodied.arch_probe keys internw0_delta \
    --drop-head track_align
```

`--drop-head` is repeatable and removes the head together with the loss terms
that name it, because spec validation requires every `LossSpec.head` to exist.

## `code_checks` — three rules no installed linter enforces

```bash
# all three rules, at the repository's 120-column limit
PYTHONPATH=. python -m tools.embodied.code_checks loongforge/embodied

# one rule, at a subtree's narrower local convention
PYTHONPATH=. python -m tools.embodied.code_checks \
    --rule code_style --limit 79 tools/embodied
```

Directories are expanded to the `.py` files below them. The exit status is the
number of findings, capped at 125, so the module works as a shell gate.

| Rule | Fires on | Deliberate exception |
| --- | --- | --- |
| `comment_language` | Chinese in a comment or docstring | Chinese inside any other string literal, because callers and tests match on error and log text |
| `unused_imports` | An imported name no code node references | A `noqa` import line (kept for a registration side effect) and any name listed in `__all__` |
| `code_style` | Over-long lines, trailing whitespace, missing blank lines above a top-level definition | — |

Two details worth knowing before you trust the output:

- The line limit is an argument, not a constant. `pyproject.toml:282` sets
  `line-length = 120` for the repository; the architecture package and its
  tests are additionally held to 79 columns by
  `tests/embodied/unit/test_arch_style.py`.
- The blank-line rule is computed from the AST, not from a regex over the
  previous line. A definition behind a multi-line decorator sits directly under
  that decorator's closing bracket, which every textual form of the rule
  reports as a violation.

## Enforced, not remembered

`tests/embodied/unit/test_arch_style.py` points `code_checks` at the
architecture package, its tests, and this directory, and asserts zero findings.
`tests/embodied/unit/test_arch_probe.py` runs all three probes against all
three providers. Both are part of:

```bash
PYTHONPATH=. python -m pytest $(ls tests/embodied/unit/test_arch_*.py) -q
```

Half the style cases check the *checkers*: each rule is handed a snippet it
must reject, because a rule that has quietly stopped firing passes every file
and looks exactly like a clean tree.

