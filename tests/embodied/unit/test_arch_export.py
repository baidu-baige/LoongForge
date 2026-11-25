# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The exporters' inlined PhyAI field lists, checked against PhyAI.

``legacy_phyai_bundle.zh-CN.md`` §6.3 singles out G3 and G9 as the two
gates the whole bundle scheme rests on, because they guard the class of
failure that does not raise. That failure is concrete:
``PretrainedConfig.from_dict`` filters its input down to declared field
names and every field carries a default
(``models/configuration.py:97-134``), so a geometry key that was
forgotten or misspelled on the export side does not fail the load -- it
silently takes the default, and the policy runs on numbers nobody chose.

The exporters answer that by writing every field explicitly from an
inlined copy of the names PhyAI declares, which only works for as long
as the copy stays in sync. Three comment blocks promise that *this file*
is what keeps it so: ``config_bridge.py:23``,
``export_dreamzero.py:56``, ``export_pi05.py:77``.

Nothing here imports PhyAI. The boundary rule (§8, J1) is that neither
repository imports the other, so the upstream dataclasses are read with
``ast``. Reading a class body is exact rather than approximate here:
``PretrainedConfig`` declares no dataclass fields of its own
(``nested_sources`` is a ``ClassVar``, which ``fields()`` does not
collect) and all nine subclasses inherit it directly, so the annotated
assignments in a class body *are* that class's field list.

The last group of cases covers the advisory/blocking split in
``report.py``, which is what lets G8 be recorded as unmet without
refusing to write the bundle.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from loongforge.embodied.model.dreamzero import export_dreamzero
from loongforge.embodied.model.export import (Capabilities, ConfigSchema,
                                              ExportReport, NativeManifest,
                                              write_bundle)
from loongforge.embodied.model.export.report import ExportError
from loongforge.embodied.model.pi05 import export_pi05


def _phyai_root() -> pathlib.Path:
    """Return the directory that should hold the ``phyai`` package.

    ``$PHYAI_SOURCE_ROOT`` wins outright, so a checkout elsewhere needs no
    edit here and a typo in it surfaces as a skip naming the path rather
    than as a silent fall-through to some other tree. Without it the
    sibling ``phyai/`` tree of the usual working copy is tried, which
    nests the package two levels down (``phyai/phyai/src/phyai``).
    """
    env = os.environ.get("PHYAI_SOURCE_ROOT")
    if env:
        return pathlib.Path(env)
    return (pathlib.Path(__file__).resolve().parents[3].parent / "phyai"
            / "phyai" / "src")


_ROOT = _phyai_root()
if not (_ROOT / "phyai" / "models" / "configuration.py").is_file():
    pytest.skip(f"no phyai package under {_ROOT}; point $PHYAI_SOURCE_ROOT "
                "at the directory holding it to enable the G9 comparison",
                allow_module_level=True)


def _parse(path: pathlib.Path) -> ast.Module:
    """Parse one PhyAI config module.

    Args:
        path: Absolute path to the module.

    Returns:
        The parsed tree.
    """
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


def _class_def(tree: ast.Module, cls: str,
               path: pathlib.Path) -> ast.ClassDef:
    """Return one top-level class, or say which file was searched.

    Only the top level is searched, because a config dataclass nested
    inside a function or another class would not be importable as
    ``phyai.models.<model>.configuration_<model>.<cls>`` either.

    Args:
        tree: Output of :func:`_parse`.
        cls: Class name.
        path: Path the tree came from, used in the message.

    Returns:
        The class definition node.

    Raises:
        LookupError: No such top-level class.
    """
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node
    raise LookupError(f"{cls} is not declared at the top level of {path}")


def _is_class_var(annotation: ast.expr) -> bool:
    """Say whether an annotation is a ``ClassVar``, which is not a field.

    ``PretrainedConfig.nested_sources`` is the live example
    (``models/configuration.py:90``); ``dataclasses.fields`` skips it and
    so must this.

    Args:
        annotation: The annotation expression.

    Returns:
        Whether the outermost name is ``ClassVar``.
    """
    head = annotation.value if isinstance(annotation,
                                          ast.Subscript) else annotation
    if isinstance(head, ast.Attribute):
        return head.attr == "ClassVar"
    return isinstance(head, ast.Name) and head.id == "ClassVar"


def _annotated(node: ast.ClassDef) -> list[ast.AnnAssign]:
    """Return the class body's annotated assignments to plain names.

    Args:
        node: A class definition.

    Returns:
        The statements, in source order.
    """
    return [stmt for stmt in node.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and not stmt.target.id.startswith("_")
            and not _is_class_var(stmt.annotation)]


def declared_fields(path: pathlib.Path, cls: str) -> frozenset[str]:
    """Return one upstream dataclass's field names.

    Args:
        path: PhyAI config module.
        cls: Class name.

    Returns:
        The declared field names.

    Raises:
        LookupError: No such top-level class.
    """
    tree = _parse(path)
    return frozenset(stmt.target.id
                     for stmt in _annotated(_class_def(tree, cls, path)))


def nested_fields(path: pathlib.Path, cls: str) -> frozenset[str]:
    """Return the field names whose type is a sibling config class.

    These are the fields ``from_dict`` rebuilds recursively from a nested
    dict (``models/configuration.py:129-133``), which is why §3.4 says a
    flat payload would leave them at their defaults.

    Args:
        path: PhyAI config module.
        cls: Class name.

    Returns:
        The sub-config field names.

    Raises:
        LookupError: No such top-level class.
    """
    tree = _parse(path)
    siblings = {node.name for node in tree.body
                if isinstance(node, ast.ClassDef)}
    return frozenset(stmt.target.id
                     for stmt in _annotated(_class_def(tree, cls, path))
                     if isinstance(stmt.annotation, ast.Name)
                     and stmt.annotation.id in siblings)


@dataclass(frozen=True)
class Bridge:
    """One exporter's config bridge and the upstream it was copied from.

    Attributes:
        schema: The schema the exporter hands ``finalise_config``.
        source: PhyAI config module, relative to the package directory.
        classes: Schema node name to upstream class name. The empty
            string is the root node.
    """

    schema: ConfigSchema
    source: str
    classes: Mapping[str, str]

    def path(self) -> pathlib.Path:
        """Return the absolute path of the upstream module."""
        return _ROOT / "phyai" / self.source

    def node(self, name: str) -> ConfigSchema:
        """Return one schema node by name; the empty string is the root.

        Args:
            name: Node name.

        Returns:
            The node.
        """
        return self.schema if not name else self.schema.children[name]


BRIDGES = {
    "dreamzero": Bridge(
        schema=export_dreamzero.CONFIG_SCHEMA,
        source="models/dreamzero/configuration_dreamzero.py",
        classes={"": "DreamZeroConfig",
                 "dit": "DreamZeroDiTConfig",
                 "text_encoder": "DreamZeroTextEncoderConfig",
                 "image_encoder": "DreamZeroImageEncoderConfig",
                 "vae": "DreamZeroVAEConfig"},
    ),
    "pi05": Bridge(
        schema=export_pi05.CONFIG_SCHEMA,
        source="models/pi05/configuration_pi05.py",
        classes={"": "PI05Config",
                 "vision": "SiglipVisionConfig",
                 "text": "PaliGemmaTextConfig",
                 "expert": "GemmaExpertConfig"},
    ),
}

FIELD_CASES = [pytest.param(model, node, id=f"{model}-{node or 'root'}")
               for model, bridge in BRIDGES.items()
               for node in sorted(bridge.classes)]


@pytest.mark.parametrize("model,node", FIELD_CASES)
def test_the_inlined_field_list_matches_the_upstream_dataclass(model, node):
    """G9 itself: one case per dataclass, so a failure names which drifted.

    Both directions matter. A field upstream declares but the exporter
    omits is the silent-default failure; a field the exporter would write
    but upstream no longer declares is dropped by ``from_dict``, which is
    the same failure with the blame reversed.
    """
    bridge = BRIDGES[model]
    cls = bridge.classes[node]
    mine = bridge.node(node).fields
    theirs = declared_fields(bridge.path(), cls)
    assert theirs, f"{cls} parsed as having no fields at all"
    assert mine == theirs, (
        f"{cls}: upstream declares and the exporter lacks "
        f"{sorted(theirs - mine)}; the exporter writes and upstream "
        f"lacks {sorted(mine - theirs)}")


@pytest.mark.parametrize("model", sorted(BRIDGES))
def test_the_schema_nests_exactly_what_upstream_rebuilds(model):
    """The nesting is not cosmetic, so it is checked against upstream too.

    ``from_dict`` only recurses into a field whose default is itself a
    config, so a child the schema flattened would load as all-defaults
    and a child the schema invented would be dropped whole.
    """
    bridge = BRIDGES[model]
    root = bridge.classes[""]
    assert set(bridge.schema.children) == nested_fields(bridge.path(), root)
    assert set(bridge.classes) - {""} == set(bridge.schema.children)

# A reader that quietly returned nothing would pass every comparison
# above, so it is handed a class body holding one of each thing it has to
# skip: a ``ClassVar``, a private name, and a bare assignment.
_SNIPPET = """
from typing import ClassVar


class Inner:
    a: int = 1


class Outer:
    nested_sources: ClassVar[dict] = {}
    inner: Inner = field(default_factory=Inner)
    b: int = 2
    _hidden: int = 3
    c = 4
"""


def test_the_field_reader_skips_class_vars_private_and_bare_names(tmp_path):
    """Check the checker, and pin the sub-config rule it infers from a type."""
    path = tmp_path / "configuration_fake.py"
    path.write_text(_SNIPPET, encoding="utf-8")
    assert declared_fields(path, "Outer") == {"inner", "b"}
    assert nested_fields(path, "Outer") == {"inner"}
    assert not nested_fields(path, "Inner")
    with pytest.raises(LookupError, match="Missing"):
        declared_fields(path, "Missing")


def _bundle(dest: pathlib.Path, report: ExportReport) -> dict:
    """Write the smallest legal native bundle and read its report back.

    Args:
        dest: Bundle directory.
        report: Report to write alongside.

    Returns:
        The parsed ``export_report.json``.
    """
    path = write_bundle(dest, tensors={"w": torch.zeros(2)},
                        manifest=NativeManifest(model_type="t", plugin="t"),
                        capabilities=Capabilities(level="EXACT"),
                        config={"a": 1}, report=report)
    return json.loads(report.path(path).read_text(encoding="utf-8"))


def test_an_unmet_advisory_gate_is_written_down_but_does_not_block(tmp_path):
    """The G8 case: unpinned upstream is a note, not a broken bundle."""
    report = ExportReport(model_type="t", plugin="t")
    report.advise("G8_upstream_pinned", False, "0 artefact(s) pinned")
    written = _bundle(tmp_path / "bundle", report)
    assert written["passed"] is True
    assert written["failed"] == []
    assert written["unmet"] == ["G8_upstream_pinned"]
    assert written["checks"][0]["blocking"] is False


def test_a_failed_blocking_gate_refuses_to_write_anything(tmp_path):
    """A bundle whose gates failed must not exist, not even half-written."""
    report = ExportReport(model_type="t", plugin="t")
    report.record("G3_no_key_transform", False, "two keys were renamed")
    dest = tmp_path / "bundle"
    with pytest.raises(ExportError, match="blocking gates"):
        _bundle(dest, report)
    assert not dest.exists()


def test_require_raises_at_its_call_site_and_record_defers_to_the_write():
    """The three severities are three methods, not one flag at the end."""
    report = ExportReport(model_type="t", plugin="t")
    report.advise("advisory", False, "ignored by the write")
    report.record("deferred", False, "caught by the write")
    assert report.failed == ("deferred",)
    assert report.unmet == ("advisory",)
    with pytest.raises(ExportError, match="immediate failed"):
        report.require("immediate", False, "raised right here")

# The four canonical names a minimal pi0.5 export has to carry: the tied
# embedding the plugin reads, the two keys the plugin's own filter
# refuses, and one ordinary parameter. Each appears twice in the live
# state dict, once under each of ``PI05Policy``'s two module aliases.
_PI05_NAMES = (export_pi05.TIED_EMBEDDING_KEY,
               *sorted(export_pi05.CANONICAL_DROP_KEYS),
               "action_in_proj.weight")


def _pi05_state_dict() -> dict:
    """Return a state dict shaped like ``PI05Policy``'s, with 2-element rows.

    Returns:
        Eight entries: four canonical names under each alias.
    """
    out = {}
    for name in _PI05_NAMES:
        out[f"{export_pi05.MODULE_PREFIX}{name}"] = torch.zeros(2)
        out[f"{export_pi05.ALIAS_PREFIX}{name}"] = torch.zeros(2)
    return out


def test_pi05_exports_when_no_upstream_artefact_is_pinned(tmp_path):
    """The default call path, which a blocking G8 gate used to refuse.

    ``upstream`` defaults to empty and the exporters document that empty
    is allowed (``export_dreamzero.py:485-488``), so this is the path the
    first caller takes. Before G8 became advisory it raised
    ``ExportError`` and wrote nothing.
    """
    model = SimpleNamespace(state_dict=_pi05_state_dict)
    model_cfg = SimpleNamespace(action_dim=32, max_action_dim=32,
                                action_horizon=50, max_state_dim=32)
    dest = tmp_path / "bundle"
    report = export_pi05.export(model, model_cfg, str(dest),
                                variant_probe=export_pi05.DEFAULT_GEOMETRY)
    assert report.failed == ()
    assert report.unmet == ("G8_upstream_pinned",)
    assert report.source_keys == 8
    assert report.written_keys == 2
    assert len(report.dropped_keys) == 6
    assert report.config_fields == len(export_pi05.CONFIG_SCHEMA.fields)
    for name in ("config.json", "arch_manifest.json", "capabilities.json"):
        assert (dest / name).is_file(), name
    assert json.loads((dest / "config.json").read_text(
        encoding="utf-8")).keys() == export_pi05.CONFIG_SCHEMA.fields


@pytest.mark.parametrize("module", (export_dreamzero, export_pi05),
                         ids=("dreamzero", "pi05"))
def test_every_exporter_treats_g8_as_advisory(module):
    """Both exporters, including the one with no cheap end-to-end case.

    DreamZero's ``export`` needs a probed DiT geometry to reach the gate
    at all, so the shared invariant is asserted at the source level
    instead: whichever method names G8, it must be the one that does not
    block.
    """
    source = inspect.getsource(module.export)
    assert 'report.advise("G8_upstream_pinned"' in source
    assert 'report.record("G8_upstream_pinned"' not in source
