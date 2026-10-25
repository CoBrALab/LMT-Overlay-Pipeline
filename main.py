#!/usr/bin/env python3
"""
main.py

CLI entrypoint for the LMT overlay pipeline.

Usage:
    python main.py
    python main.py --output-base-dir /data/results --no-progress --log-level DEBUG
    python main.py -i tracking.sqlite -v /data/period1/ -o /data/overlays/period1

With -i/--input and -v/--videos, the database and raw videos come from
the command line and no GUI is shown, so runs can be scripted (e.g. one
run per tracking period in a shell loop). With neither flag, this
launches two file-picker dialogs (via gui_selector.py): first to select
the LMT SQLite database, then to select one or more raw video files.
Either way, files may live anywhere on disk, in any folder structure --
there is no assumption of a fixed project layout.

By default each run writes its output into a fresh, uniquely timestamped
subfolder under --output-base-dir (default: ./output), so repeated runs
never overwrite each other's results. -o/--output-folder writes into
exactly the given folder instead; output videos that already exist there
are skipped unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import database
import video_processor

logger = logging.getLogger(__name__)

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render corrected LMT overlays (from SQLite ground truth) onto raw videos. "
            "Give -i and -v to run without the GUI; give neither to pick files in the GUI."
        ),
    )

    parser.add_argument(
        "-i", "--input", type=Path, metavar="DATABASE",
        help="Path to the LMT SQLite database. Must be given together with -v/--videos.",
    )
    parser.add_argument(
        "-v", "--videos", type=Path, nargs="+", metavar="PATH",
        help=(
            "One or more raw video files and/or folders. Folders are searched "
            f"(non-recursively) for {config.VIDEO_GLOB_PATTERN!r}. "
            "Must be given together with -i/--input."
        ),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", "--output-folder", type=Path,
        help=(
            "Write output videos into exactly this folder (created if needed), "
            "with no timestamped subfolder. Existing output videos are skipped "
            "unless --overwrite is given."
        ),
    )
    output_group.add_argument(
        "--output-base-dir",
        type=Path,
        default=config.OUTPUT_VIDEO_DIR,
        help=(
            "Base directory under which each run creates its own "
            f"timestamped output subfolder (default: {config.OUTPUT_VIDEO_DIR}). "
            "E.g. <base>/2026-07-20_17-34-12/video_t18024.mp4"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace output videos that already exist, instead of skipping them.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bars (useful for non-interactive/log-file runs).",
    )
    parser.add_argument(
        "--log-level", default=config.LOG_LEVEL, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help=f"Logging verbosity (default: {config.LOG_LEVEL})",
    )
    args = parser.parse_args(argv)

    if (args.input is None) != (args.videos is None):
        parser.error("-i/--input and -v/--videos must be given together (or neither, to use the GUI).")
    return args


def _expand_video_paths(paths: list[Path]) -> list[Path]:
    """
    Expand any folders in paths into the raw videos they contain
    (config.VIDEO_GLOB_PATTERN, non-recursive), keeping files as given.
    Folder support matters because PowerShell does not expand '*.mp4'
    itself. Missing paths are logged and skipped, like unusable videos.
    """
    expanded: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            matches = sorted(path.glob(config.VIDEO_GLOB_PATTERN))
            if not matches:
                logger.warning("No videos matched pattern %r under %s", config.VIDEO_GLOB_PATTERN, path)
            expanded.extend(matches)
        elif path.exists():
            expanded.append(path)
        else:
            logger.error("Skipping video path that does not exist: %s", path)
    return expanded


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_unique_output_dir(base_dir: Path, database_path: Path) -> Path:
    """
    Build this run's output directory as <base_dir>/<database_name>_<timestamp>. If that
    exact folder somehow already exists (two runs started within the
    same second -- unlikely but not impossible), append a numeric suffix
    until a free name is found. This is what guarantees "never silently
    overwrite previous outputs": we never write into a folder that
    already exists.
    """
    timestamp = datetime.now().strftime(config.OUTPUT_TIMESTAMP_FORMAT)
    database_name = database_path.stem
    candidate = base_dir / f"{database_name}_{timestamp}"
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = base_dir / f"{database_name}_{timestamp}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _sanity_check_frame_overlap(conn, videos: list[video_processor.VideoInfo]) -> None:
    """
    Warn (but don't abort) if the selected videos' global frame ranges
    don't overlap the DETECTION table's frame range at all -- usually
    means the wrong database was picked for these videos.
    """
    if not videos:
        return

    try:
        db_min, db_max = database.get_global_frame_bounds(conn)
    except ValueError as exc:
        logger.error("Cannot sanity-check frame overlap: %s", exc)
        return

    video_min = min(v.start_frame for v in videos)
    video_max = max(v.end_frame for v in videos)

    if video_max <= db_min or video_min >= db_max:
        logger.warning(
            "Selected videos span global frames [%d, %d) but DETECTION "
            "spans [%d, %d] -- these do not overlap at all. Double-check "
            "you selected the right database for these videos.",
            video_min, video_max, db_min, db_max,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _configure_logging(args.log_level)

    if args.no_progress:
        config.PROGRESS_BAR = False
    config.OVERWRITE_OUTPUT = args.overwrite

    if args.input is not None:
        database_path = args.input.resolve()
        video_paths = _expand_video_paths(args.videos)
    else:
        # gui_selector is imported here, not at module level, so a
        # command-line run never needs Tkinter (e.g. on a headless server).
        import gui_selector

        try:
            database_path, video_paths = gui_selector.select_database_and_videos()
        except ImportError as exc:
            logger.error(
                "Tkinter is not available (%s). On Debian/Ubuntu, install it with: "
                "sudo apt-get install python3-tk. On other platforms, reinstall "
                "Python with Tk support enabled, or pass -i and -v to run without the GUI.", exc,
            )
            return 1
        except gui_selector.SelectionCancelled as exc:
            logger.warning("Selection cancelled: %s Nothing was processed.", exc)
            return 1

    config.DATABASE_PATH = database_path
    if args.output_folder is not None:
        config.OUTPUT_VIDEO_DIR = args.output_folder.resolve()
    else:
        config.OUTPUT_VIDEO_DIR = _make_unique_output_dir(args.output_base_dir.resolve(), database_path)

    try:
        config.validate_config()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Using database: %s", config.DATABASE_PATH)
    logger.info("Output directory for this run: %s", config.OUTPUT_VIDEO_DIR)

    videos = video_processor.discover_videos_from_paths(video_paths)
    if not videos:
        logger.error("None of the selected video files were usable; nothing to do.")
        return 1
    logger.info(
        "Processing %d video(s), global frame range [%d, %d)",
        len(videos), videos[0].start_frame, videos[-1].end_frame,
    )

    start_time = time.monotonic()
    try:
        with database.connect(config.DATABASE_PATH) as conn:
            _sanity_check_frame_overlap(conn, videos)
            video_processor.process_all_videos(conn, videos)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; partially processed videos remain in %s", config.OUTPUT_VIDEO_DIR)
        return 130
    except Exception:
        logger.exception("Unexpected error during processing.")
        return 1

    elapsed = time.monotonic() - start_time
    logger.info("Done. Processed %d video(s) in %.1fs. Output: %s", len(videos), elapsed, config.OUTPUT_VIDEO_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())