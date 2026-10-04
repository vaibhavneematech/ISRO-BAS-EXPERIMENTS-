"""
video_recorder.py
=================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 7: Local Video Storage

Author : SIH26174 Team
Created: 2026-09-21

Purpose
-------
Wraps cv2.VideoWriter to continuously record the annotated experiment
video (BGR frames) to a timestamped MP4 file in the  recordings/  folder.

Usage
-----
    from src.recording.video_recorder import VideoRecorder

    recorder = VideoRecorder(fps=30.0, width=1280, height=720)
    recorder.start()                 # creates the file, opens the writer

    for frame in ...:               # frame = annotated BGR np.ndarray
        recorder.write(frame)

    recorder.stop()                  # flushes + releases the writer
    print(recorder.filepath)         # full path to the saved MP4

Notes
-----
* The MP4 is written with the MJPG codec inside an MP4 container.
  Nearly every media player can open it without extra codecs.
* If the fourcc MJPG is unavailable on a given platform, the recorder
  falls back to the uncompressed DIB codec (larger files, always works).
* One VideoRecorder instance == one MP4 file.  Create a fresh instance
  for each experiment session.
"""

from __future__ import annotations

import pathlib
import datetime
import threading
from typing import Optional

import cv2
import numpy as np


# ── Recordings directory ──────────────────────────────────────────────────────
_THIS_DIR      = pathlib.Path(__file__).resolve().parent
RECORDINGS_DIR = _THIS_DIR.parent.parent / "recordings"


def _make_filepath(prefix: str = "experiment") -> pathlib.Path:
    """Return a unique timestamped path: recordings/experiment_YYYY-MM-DD_HH-MM-SS.mp4"""
    ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{prefix}_{ts}.mp4"
    return RECORDINGS_DIR / name


class VideoRecorder:
    """
    Thread-safe continuous video recorder using cv2.VideoWriter.

    Parameters
    ----------
    fps    : float   – target frame rate (default 30.0).
    width  : int     – frame width  in pixels (default 1280).
    height : int     – frame height in pixels (default 720).
    prefix : str     – filename prefix (default "experiment").
    """

    def __init__(
        self,
        fps    : float = 30.0,
        width  : int   = 640,
        height : int   = 480,
        prefix : str   = "experiment",
        output_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self._fps        = fps
        self._width      = width
        self._height     = height
        self._prefix     = prefix
        self._output_dir = pathlib.Path(output_dir) if output_dir else RECORDINGS_DIR

        self._writer   : Optional[cv2.VideoWriter] = None
        self._filepath : Optional[pathlib.Path]    = None
        self._lock     = threading.Lock()
        self._active   = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True while the recorder is open and accepting frames."""
        return self._active

    @property
    def filepath(self) -> Optional[pathlib.Path]:
        """Full path of the MP4 being written (None before start())."""
        return self._filepath

    @property
    def output_path(self) -> Optional[pathlib.Path]:
        """Alias for filepath."""
        return self._filepath

    @property
    def filename(self) -> str:
        """Basename of the MP4 (empty string before start())."""
        return self._filepath.name if self._filepath else ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open a new MP4 file and begin recording.

        Returns True on success, False if the VideoWriter could not be opened.
        """
        with self._lock:
            if self._active:
                return True   # already running

            self._output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._filepath = self._output_dir / f"{self._prefix}_{ts}.mp4"

            # Try MJPG first (good quality + small files)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                str(self._filepath),
                fourcc,
                self._fps,
                (self._width, self._height),
            )

            # Fallback to uncompressed DIB if MJPG failed
            if not writer.isOpened():
                fourcc  = cv2.VideoWriter_fourcc(*"XVID")
                writer  = cv2.VideoWriter(
                    str(self._filepath),
                    fourcc,
                    self._fps,
                    (self._width, self._height),
                )

            if not writer.isOpened():
                print(f"[VideoRecorder] ⚠ Failed to open writer for {self._filepath}")
                self._filepath = None
                return False

            self._writer = writer
            self._active = True
            print(f"[VideoRecorder] ● Recording → {self._filepath.name}")
            return True

    def write(self, frame: np.ndarray) -> None:
        """
        Write one BGR frame.  Must be called only while is_active == True.
        Frames of incorrect size are resized automatically.
        """
        if not self._active:
            return

        with self._lock:
            if self._writer is None:
                return

            h, w = frame.shape[:2]
            if w != self._width or h != self._height:
                frame = cv2.resize(frame, (self._width, self._height))

            self._writer.write(frame)

    def write_frame(self, frame: np.ndarray) -> None:
        """Alias for write(frame)."""
        self.write(frame)

    def stop(self) -> Optional[pathlib.Path]:
        """
        Flush, release, and close the MP4 file.

        Returns the path to the completed file (or None if never started).
        """
        with self._lock:
            if not self._active:
                return self._filepath

            self._active = False
            if self._writer:
                self._writer.release()
                self._writer = None

            if self._filepath and self._filepath.exists():
                size_mb = self._filepath.stat().st_size / 1_048_576
                print(f"[VideoRecorder] ■ Saved  {self._filepath.name}  ({size_mb:.1f} MB)")
            return self._filepath

    def close(self) -> Optional[pathlib.Path]:
        """Alias for stop()."""
        return self.stop()
