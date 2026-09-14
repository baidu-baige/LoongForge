# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
ego2robot CLI dispatch entry point.

Each subcommand maps to a module under steps/ with matching
build_arg_parser()/run() functions. See SUBCOMMANDS below for the mapping:
  load         -> steps/loader.py
  align        -> steps/action_alignment.py
  mask         -> steps/hand_mask.py
  inpaint      -> steps/inpaint.py
  depth        -> steps/depth_estimate.py (standalone step; optionally included
                  in run-all with --with_depth)
  retarget     -> steps/robot_retarget.py (dual-arm IK retargeting, rendering,
                  and compositing for 15 morphologies)
  quality      -> steps/quality/quality_curation.py (L1/L2 plus SGLang L3)
  path-b       -> steps/path_b.py (pure video -> hand tracks -> Zarr)
  lerobot      -> steps/lerobot_writer.py
  validate     -> steps/validate.py
  demo         -> steps/demo_grid.py
  run-all      -> orchestration logic in pipeline.py

Usage:
    python cli.py <subcommand> --help   # Show options for a subcommand.
    python cli.py run-all --input_dir <zarr-dir> --output_dir <output-root>
"""
import argparse
import importlib
import sys

SUBCOMMANDS = {
    # Keep imports lazy: ``path-b`` should be usable in a video-only setup
    # without importing MuJoCo/Mink/SAM3 just to parse the CLI.
    "load": "steps.loader",
    "align": "steps.action_alignment",
    "mask": "steps.hand_mask",
    "inpaint": "steps.inpaint",
    "depth": "steps.depth_estimate",
    "retarget": "steps.robot_retarget",
    "lerobot": "steps.lerobot_writer",
    "validate": "steps.validate",
    "demo": "steps.demo_grid",
    "export": "steps.export_hdf5",
    "quality": "steps.quality",
    "path-b": "steps.path_b",
}


def load_subcommand(name):
    return importlib.import_module(SUBCOMMANDS[name])


def build_parser():
    """Build the main CLI parser and attach each subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="EgoVerse -> LeRobot H2R data synthesis pipeline for 15 dual-arm morphologies",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # A video-only installation commonly does not have Mink/MuJoCo/SAM3 yet.
    # Build only the requested Path B parser in that case; the full parser is
    # retained for all other invocations and for the regular pipeline.
    requested = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS else None
    names = ("path-b",) if requested == "path-b" else tuple(SUBCOMMANDS)
    for name in names:
        module = load_subcommand(name)
        step_parser = module.build_arg_parser()
        sub.add_parser(
            name,
            parents=[step_parser],
            add_help=False,
            description=step_parser.description,
        )

    if requested != "path-b":
        import pipeline
        run_all_parser = pipeline.build_arg_parser()
        sub.add_parser(
            "run-all",
            parents=[run_all_parser],
            add_help=False,
            description=run_all_parser.description,
        )

    return parser


def main():
    """Parse a subcommand and dispatch it to the corresponding module's run()."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-all":
        import pipeline
        pipeline.run(args)
        return

    if args.command not in SUBCOMMANDS:
        parser.error(f"unknown command: {args.command}")
        sys.exit(1)
    module = load_subcommand(args.command)

    # Match each module's main(): defaults for the lerobot subcommand's
    # bg_video_dir/tmp_dir depend on output_dir/ik_dir and cannot be represented
    # by argparse default=None, so fill them here as lerobot_writer.main() does.
    if args.command == "lerobot":
        if args.bg_video_dir is None:
            args.bg_video_dir = args.ik_dir
        if args.tmp_dir is None:
            from pathlib import Path
            args.tmp_dir = str(Path(args.output_dir) / "_tmp_av1")

    module.run(args)


if __name__ == "__main__":
    main()
