"""
voice_alert.py
==============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Upgraded Offline Voice & Beep Alert System.

Sound Behavior Specifications:
1. Step Successfully Completed (CORRECT):
   - Short clean positive BEEP sound (beep.wav)
   - Plays only once per successful step transition
2. Step Skipped (SKIPPED):
   - assets/audio/skipped (.mp3 / .wav)
   - Plays only once per event with cooldown
3. Wrong Sequence Detected (WRONG_ORDER):
   - assets/audio/wrongsequence (.mp3 / .wav)
   - Plays only once per event with cooldown
4. All Steps Completed (COMPLETED):
   - assets/audio/stepcomplete (.mp3 / .wav)
   - Plays only once upon terminal verification
"""

from __future__ import annotations

import math
import os
import pathlib
import struct
import subprocess
import sys
import threading
import time
import wave
from enum import Enum, auto
from typing import Callable, Optional

# ── Base Directory ────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
AUDIO_DIR = _THIS_DIR.parent.parent / "assets" / "audio"


# ── Alert Types ───────────────────────────────────────────────────────────────

class AlertType(Enum):
    STEP_COMPLETE       = auto()   # Clean positive beep upon successful step completion
    STEP_SKIPPED        = auto()   # assets/audio/skipped
    WRONG_SEQUENCE      = auto()   # assets/audio/wrongsequence
    EXPERIMENT_COMPLETE = auto()   # assets/audio/stepcomplete


# ── File Candidates per Alert ─────────────────────────────────────────────────

_FILE_CANDIDATES: dict[AlertType, list[str]] = {
    AlertType.STEP_COMPLETE       : ["success_beep.wav", "beep.wav", "step_complete.wav"],
    AlertType.STEP_SKIPPED        : ["step_skipped.wav", "skipped.wav", "skipped.mp3"],
    AlertType.WRONG_SEQUENCE      : ["wrong_sequence.wav", "wrongsequence.wav", "wrongsequence.mp3"],
    AlertType.EXPERIMENT_COMPLETE : ["experiment_completed.wav", "stepcomplete.mp3", "stepcompleted.mp3"],
}

_DEFAULT_COOLDOWNS: dict[AlertType, float] = {
    AlertType.STEP_COMPLETE       : 1.5,
    AlertType.STEP_SKIPPED        : 3.5,
    AlertType.WRONG_SEQUENCE      : 3.5,
    AlertType.EXPERIMENT_COMPLETE : 6.0,
}


# ── Clean Positive Beep Generator (Wave Synthesizer) ──────────────────────────

def ensure_beep_exists(audio_dir: pathlib.Path) -> pathlib.Path:
    """
    Synthesize an avionics-grade two-tone clean positive confirmation chime
    (880 Hz -> 1318.5 Hz) if beep.wav does not exist.
    Zero external dependencies, pure standard library.
    """
    beep_file = audio_dir / "beep.wav"
    if beep_file.exists():
        return beep_file

    try:
        audio_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = 44100
        duration_1 = 0.08
        duration_2 = 0.12
        freq_1 = 880.0
        freq_2 = 1318.5

        samples = []
        n1 = int(sample_rate * duration_1)
        for i in range(n1):
            t = i / sample_rate
            env = min(i / (sample_rate * 0.01), 1.0) * min((n1 - i) / (sample_rate * 0.01), 1.0)
            val = 0.45 * env * math.sin(2 * math.pi * freq_1 * t)
            samples.append(int(val * 32767))

        n2 = int(sample_rate * duration_2)
        for i in range(n2):
            t = i / sample_rate
            env = min(i / (sample_rate * 0.01), 1.0) * (1.0 - (i / n2) ** 0.8)
            val = 0.50 * env * math.sin(2 * math.pi * freq_2 * t)
            samples.append(int(val * 32767))

        with wave.open(str(beep_file), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        print(f"[VoiceAlert] Generated clean positive beep at: {beep_file.name}")
    except Exception as exc:
        print(f"[VoiceAlert] Could not synthesize beep.wav: {exc}")

    return beep_file


# ── Native Audio Playback Engine ──────────────────────────────────────────────

def _play_file_blocking(path: pathlib.Path) -> None:
    """
    Play a sound file synchronously. Call strictly from a daemon thread.
    Uses winsound for WAV and PowerShell MediaPlayer for MP3 on Windows.
    """
    if not path.exists():
        print(f"[VoiceAlert] WARNING: file does not exist: {path}")
        return

    p = str(path.resolve())
    suffix = path.suffix.lower()

    if suffix == ".wav" and sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(p, winsound.SND_FILENAME)
            return
        except Exception as exc:
            print(f"[VoiceAlert] winsound error: {exc}")

    if sys.platform == "win32":
        try:
            ps_cmd = (
                f'Add-Type -AssemblyName presentationCore; '
                f'$player = New-Object System.Windows.Media.MediaPlayer; '
                f'$player.Open([System.Uri]"{p}"); '
                f'$player.Play(); '
                f'Start-Sleep -Milliseconds 2400; '
                f'$player.Close()'
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                timeout=6,
            )
        except Exception as exc:
            print(f"[VoiceAlert] Native audio error for {path.name}: {exc}")
    else:
        print(f"[VoiceAlert] [Audio Output] {path.name}")


# ── Voice Alert Manager ───────────────────────────────────────────────────────

class VoiceAlertSystem:
    """
    Thread-safe, non-blocking audio manager with single-event debounce.
    """

    def __init__(
        self,
        audio_dir: pathlib.Path = AUDIO_DIR,
        enabled: bool = True,
        on_play_callback: Optional[Callable[[str], None]] = None,
    ):
        self._audio_dir = pathlib.Path(audio_dir)
        self._enabled = enabled
        self._on_play_callback = on_play_callback
        self._cooldowns = dict(_DEFAULT_COOLDOWNS)
        self._last_played: dict[AlertType, float] = {}
        self._fired_keys: set[str] = set()
        self._lock = threading.Lock()

        # Ensure directory and positive beep exist
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        ensure_beep_exists(self._audio_dir)

    def resolve_audio_file(self, alert: AlertType) -> Optional[pathlib.Path]:
        """Resolve highest priority available file candidate for an alert."""
        candidates = _FILE_CANDIDATES.get(alert, [])
        for name in candidates:
            p = self._audio_dir / name
            if p.exists():
                return p

        # Fallback for beep
        if alert == AlertType.STEP_COMPLETE:
            return ensure_beep_exists(self._audio_dir)

        return None

    def play(self, alert: AlertType) -> bool:
        """
        Play alert asynchronously in a background thread or through GUI callback.
        Guarantees non-blocking execution and enforces cooldown.
        """
        if not self._enabled:
            return False

        now = time.monotonic()
        cooldown = self._cooldowns.get(alert, 2.0)

        with self._lock:
            last = self._last_played.get(alert, 0.0)
            if now - last < cooldown:
                return False  # Cooldown active
            self._last_played[alert] = now

        target_file = self.resolve_audio_file(alert)
        if not target_file:
            print(f"[VoiceAlert] No file found for alert {alert.name}")
            return False

        # If GUI player callback registered, trigger it (lowest latency)
        if self._on_play_callback:
            try:
                self._on_play_callback(str(target_file))
                return True
            except Exception as exc:
                print(f"[VoiceAlert] GUI audio callback failed: {exc}")

        # Dispatch background thread for native audio
        t = threading.Thread(
            target=_play_file_blocking,
            args=(target_file,),
            daemon=True,
            name=f"alert-{alert.name}",
        )
        t.start()
        return True

    def play_alert(self, alert: AlertType) -> bool:
        """Alias for play(alert)."""
        return self.play(alert)

    def play_if_new(self, alert: AlertType, event_key: str) -> bool:
        """
        Play sound exactly ONCE per unique event key (e.g. SKIPPED_REMOVE_RED_BOX).
        Avoids spamming across frames.
        """
        if not self._enabled:
            return False

        with self._lock:
            if event_key in self._fired_keys:
                return False
            self._fired_keys.add(event_key)

        return self.play(alert)

    def reset_fired_keys(self) -> None:
        """Clear single-event history (call on protocol reset)."""
        with self._lock:
            self._fired_keys.clear()
            self._last_played.clear()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self._on_play_callback = callback
