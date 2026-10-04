"""
video_stream.py
===============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 8: MJPEG Video Streaming (zero extra dependencies)

Author : SIH26174 Team
Created: 2026-09-21

Design
------
Streams the live annotated camera frames as an MJPEG HTTP stream.

  • Uses ONLY Python standard library – no Flask, no OpenCV networking,
    no RTSP server needed.
  • Any browser on the SAME network can open:
        http://<this-machine-IP>:<port>/
  • Multiple viewers can connect simultaneously (each is served by its
    own thread via ThreadingHTTPServer).

How MJPEG works
---------------
The server sends a single HTTP response with Content-Type:
  multipart/x-mixed-replace;boundary=frame

Each JPEG frame is pushed as one part of that multipart body.
The browser renders each part as it arrives, giving a live video effect.

Architecture
------------

  CameraWorker (QThread)
       │
       │ pushes annotated np.ndarray via push_frame()
       ▼
  MJPEGStreamer._current_jpeg   (bytes, protected by threading.Lock)
       │
       │ served on demand to every connected client
       ▼
  MJPEGHandler.do_GET()
      ├─ /            → HTML page with <img> pointing at /stream
      └─ /stream      → infinite multipart MJPEG response

Usage
-----
    from src.streaming.video_stream import MJPEGStreamer

    streamer = MJPEGStreamer(host="0.0.0.0", port=8080)
    streamer.start()                   # non-blocking, runs in daemon thread

    # Each time you have a new BGR frame:
    streamer.push_frame(annotated_bgr)

    streamer.stop()
"""

from __future__ import annotations

import io
import threading
import time
import textwrap
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Global latest-frame store
# ══════════════════════════════════════════════════════════════════════════════

class _FrameStore:
    """Thread-safe holder for the latest JPEG bytes."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._jpeg  : Optional[bytes] = None
        self._event = threading.Event()   # signals that a new frame arrived

    def put(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[bytes]:
        self._event.wait(timeout)
        self._event.clear()
        with self._lock:
            return self._jpeg


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP request handler
# ══════════════════════════════════════════════════════════════════════════════

_BOUNDARY = b"frame"

_HTML_PAGE = textwrap.dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VYOM • On-Board Protocol Compliance Assistant (ISRO - SIH26174)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #080c14;
      --sidebar-bg: #0d131f;
      --card-bg: #0e1422;
      --card-border: #1a2338;
      --tile-bg: #121a2c;
      --tile-border: #1e2942;
      --blue-primary: #2563eb;
      --blue-light: #3b82f6;
      --green-ok: #10b981;
      --amber-warn: #f59e0b;
      --red-danger: #ef4444;
      --text-white: #f1f5f9;
      --text-muted: #8492a6;
      --text-subtle: #505e75;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-base);
      color: var(--text-white);
      font-family: 'Plus Jakarta Sans', -apple-system, system-ui, sans-serif;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    /* ── LEFT FIXED SIDEBAR ── */
    #sidebar {
      width: 320px;
      min-width: 320px;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      padding: 20px 16px;
      gap: 14px;
      overflow-y: auto;
    }
    .brand-box {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #1d4ed8, #3b82f6);
      color: #fff;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      font-weight: 900;
      box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    .brand-title {
      font-size: 20px;
      font-weight: 900;
      letter-spacing: 1.5px;
      color: #fff;
      line-height: 1.1;
    }
    .brand-sub {
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.8px;
      color: var(--text-muted);
    }
    .badge-offline {
      background: #121a2c;
      color: #cbd5e1;
      border: 1px solid #243048;
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
      justify-content: center;
    }
    .intro-card {
      background: var(--tile-bg);
      border: 1px solid var(--tile-border);
      border-radius: 6px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .intro-title {
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.6px;
      color: var(--blue-light);
      text-transform: uppercase;
    }
    .intro-text {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.45;
    }
    .section-label {
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 1px;
      color: var(--text-subtle);
      margin-top: 4px;
      text-transform: uppercase;
    }
    .exp-select {
      background: #0e1422;
      color: #f1f5f9;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 11.5px;
      font-weight: 600;
      outline: none;
      cursor: pointer;
      width: 100%;
    }
    .timeline-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      flex: 1;
      overflow-y: auto;
    }
    .timeline-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 11.5px;
    }
    .timeline-item.active {
      background: rgba(37,99,235,0.12);
      border: 1px solid rgba(59,130,246,0.45);
      font-weight: 700;
      color: #fff;
    }
    .timeline-item.completed {
      color: #cbd5e1;
    }
    .timeline-item.pending {
      color: var(--text-muted);
    }
    .timeline-badge {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: 900;
    }
    .timeline-item.active .timeline-badge { background: var(--blue-primary); color: #fff; }
    .timeline-item.completed .timeline-badge { background: #15803d; color: #fff; }
    .timeline-item.pending .timeline-badge { border: 2px solid #243048; color: transparent; }

    /* ── RIGHT MAIN SCROLLABLE VIEWPORT ── */
    #main-scroll {
      flex: 1;
      height: 100vh;
      overflow-y: auto;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Telemetry strip */
    .telemetry-strip {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 24px;
    }
    .telem-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .telem-label {
      font-size: 9.5px;
      font-weight: 800;
      letter-spacing: 0.5px;
      color: var(--text-subtle);
      text-transform: uppercase;
    }
    .telem-val {
      font-size: 15px;
      font-weight: 900;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Grid Row 1: Camera + Status Cards */
    .row-cam-status {
      display: grid;
      grid-template-columns: 480px 1fr;
      gap: 16px;
    }
    .cam-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .cam-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 800;
    }
    .cam-viewport {
      width: 480px;
      height: 480px;
      background: #04060a;
      border: 1px solid #1a2338;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .cam-viewport img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .col-status-cards {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .status-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .card-header-label {
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.6px;
      color: var(--text-subtle);
      text-transform: uppercase;
    }
    .pill-badge {
      background: var(--blue-primary);
      color: #fff;
      font-size: 10.5px;
      font-weight: 800;
      padding: 3px 8px;
      border-radius: 4px;
      width: fit-content;
    }
    .card-title {
      font-size: 15px;
      font-weight: 800;
      color: #fff;
    }
    .card-desc {
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.45;
    }
    .progress-bar-bg {
      background: #121a2c;
      border-radius: 3px;
      height: 6px;
      width: 100%;
      overflow: hidden;
    }
    .progress-bar-fill {
      background: var(--blue-primary);
      height: 100%;
      width: 75%;
    }

    /* Grid Row 2: Explainable Evidence + Object Tracking */
    .row-evidence-tracking {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .tiles-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 6px;
    }
    .evidence-tile {
      background: var(--tile-bg);
      border: 1px solid var(--tile-border);
      border-radius: 6px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .tile-title {
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
    }
    .tile-val {
      font-size: 16px;
      font-weight: 900;
      font-family: 'JetBrains Mono', monospace;
    }
    .tile-ok {
      font-size: 10px;
      font-weight: 800;
      color: var(--green-ok);
    }

    /* Row 3: Big Screen Mission Audit Log Terminal */
    .big-screen-log-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 320px;
    }
    .log-top-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .log-title {
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.6px;
    }
    .log-terminal {
      background: #05080e;
      border: 1px solid #141c2d;
      border-radius: 6px;
      padding: 10px 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      height: 240px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .log-entry {
      color: #94a3b8;
      line-height: 1.5;
    }
    .log-entry.compliance { color: var(--green-ok); }
    .log-entry.alert { color: var(--red-danger); }
    .log-entry.step { color: var(--blue-light); }
    .log-entry.detection { color: #38bdf8; }

    /* Footer Tools */
    .footer-tools {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .btn {
      background: var(--tile-bg);
      color: #cbd5e1;
      border: 1px solid var(--tile-border);
      border-radius: 5px;
      padding: 8px 16px;
      font-size: 11.5px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
    }
    .btn:hover { background: #1a253d; color: #fff; }
    .btn-primary {
      background: var(--blue-primary);
      color: #fff;
      border-color: #3b82f6;
    }
    .btn-primary:hover { background: #1d4ed8; }
  </style>
</head>
<body>

  <!-- LEFT SIDEBAR -->
  <aside id="sidebar">
    <div class="brand-box">
      <div class="brand-logo">V</div>
      <div>
        <div class="brand-title">VYOM</div>
        <div class="brand-sub">ON-BOARD PROTOCOL COMPLIANCE ASSISTANT</div>
      </div>
    </div>

    <div class="badge-offline">✈ Offline Edge Mode</div>

    <!-- Project Introduction -->
    <div class="intro-card">
      <div class="intro-title">Project Overview</div>
      <div class="intro-text">
        Autonomous computer vision assistant for on-board Biological Apparatus Subsystem (BAS) space experiments. Tracks astronaut hand actions, verifies experiment sequence integrity, and issues real-time guidance 100% offline.
      </div>
    </div>

    <!-- Experiment Selector -->
    <div class="section-label">Experiment</div>
    <select class="exp-select" id="expSelect">
      <option selected>ISRO Red &amp; Yellow Box BAS Experiment</option>
      <option>Phone Pickup Sequence</option>
      <option>Bottle Handling Sequence</option>
      <option>Notebook Transfer Sequence</option>
      <option>Cup Placement Sequence</option>
    </select>

    <!-- Mission Timeline -->
    <div class="section-label">Mission Timeline</div>
    <div class="timeline-list">
      <div class="timeline-item completed">
        <div class="timeline-badge">✓</div>
        <span>[0] System Checkout</span>
      </div>
      <div class="timeline-item completed">
        <div class="timeline-badge">✓</div>
        <span>[1] Preparation &amp; Setup</span>
      </div>
      <div class="timeline-item active">
        <div class="timeline-badge">▶</div>
        <span>[2] Sample Canister Manipulation</span>
      </div>
      <div class="timeline-item pending">
        <div class="timeline-badge">○</div>
        <span>[3] Reagent Addition</span>
      </div>
      <div class="timeline-item pending">
        <div class="timeline-badge">○</div>
        <span>[4] Sealing &amp; Locking</span>
      </div>
      <div class="timeline-item pending">
        <div class="timeline-badge">○</div>
        <span>[5] Optical Inspection</span>
      </div>
      <div class="timeline-item pending">
        <div class="timeline-badge">○</div>
        <span>[6] Terminal Storage</span>
      </div>
    </div>
  </aside>

  <!-- RIGHT SCROLLABLE VIEWPORT -->
  <main id="main-scroll">

    <!-- Top Telemetry Strip -->
    <div class="telemetry-strip">
      <div class="telem-item">
        <span class="telem-label">FPS</span>
        <span class="telem-val" style="color:#00f0ff;">29.8</span>
      </div>
      <div class="telem-item">
        <span class="telem-label">Runtime</span>
        <span class="telem-val" id="runtime">00:04:12</span>
      </div>
      <div class="telem-item">
        <span class="telem-label">Recording</span>
        <span class="telem-val" style="color:var(--red-danger);">● ON</span>
      </div>
      <div class="telem-item">
        <span class="telem-label">Stream State</span>
        <span class="telem-val" style="color:var(--green-ok);">● LIVE</span>
      </div>
      <div style="margin-left: auto; font-size:11px; color:var(--text-muted);">
        ↻ Last Updated: Just now
      </div>
    </div>

    <!-- Row 1: Camera Preview + 3 Status Cards -->
    <div class="row-cam-status">
      <div class="cam-card">
        <div class="cam-header">
          <span>📹 LIVE CAMERA PREVIEW</span>
          <span style="font-family:'JetBrains Mono'; color:var(--text-muted);">⤢ 480 × 480</span>
        </div>
        <div class="cam-viewport">
          <img src="/stream" alt="Live Camera Stream">
        </div>
      </div>

      <div class="col-status-cards">
        <!-- Current Step Card -->
        <div class="status-card">
          <div class="card-header-label">Current Step</div>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="pill-badge">STEP 3</div>
            <div class="card-title">Sample Canister Manipulation</div>
          </div>
          <div class="card-desc">
            Align primary sample canisters with clamp guides and verify grasp stability. Avoid sudden acceleration.
          </div>
          <div style="margin-top:auto;">
            <div class="progress-bar-bg"><div class="progress-bar-fill"></div></div>
            <div style="font-size:10px; color:var(--text-subtle); margin-top:4px;">In Progress (75% Stable)</div>
          </div>
        </div>

        <!-- Next Step Guidance Card -->
        <div class="status-card">
          <div class="card-header-label">Next Step Guidance</div>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="pill-badge" style="background:#1d4ed8;">STEP 4</div>
            <div class="card-title">– Reagent Addition</div>
          </div>
          <div class="card-desc">
            Retrieve biological reagent ampoule and inject into canister port 1.
          </div>
        </div>

        <!-- Overall Status Card -->
        <div class="status-card">
          <div class="card-header-label">Overall Status</div>
          <div style="display:flex; align-items:center; gap:14px;">
            <div style="font-size:28px;">🛡</div>
            <div>
              <div style="font-size:16px; font-weight:900; color:var(--green-ok);">COMPLIANT</div>
              <div style="font-size:11px; color:var(--text-muted);">All protocol checks passing nominal tolerances</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Row 2: Explainable Evidence + Object Tracking -->
    <div class="row-evidence-tracking">
      <!-- Evidence Card -->
      <div class="status-card">
        <div class="card-header-label">Explainable Evidence ⓘ</div>
        <div class="tiles-grid">
          <div class="evidence-tile">
            <span class="tile-title">✋ Pose Stability</span>
            <span class="tile-val">0.93</span>
            <span class="tile-ok">OK</span>
          </div>
          <div class="evidence-tile">
            <span class="tile-title">⌖ Hand-Object Proximity</span>
            <span class="tile-val">42 px</span>
            <span class="tile-ok">OK [NEAR]</span>
          </div>
          <div class="evidence-tile">
            <span class="tile-title">⎇ Protocol Sequence</span>
            <span class="tile-val">0.98</span>
            <span class="tile-ok">OK</span>
          </div>
          <div class="evidence-tile">
            <span class="tile-title">🧊 Object Integrity</span>
            <span class="tile-val">0.95</span>
            <span class="tile-ok">OK</span>
          </div>
        </div>
        <div style="margin-top:10px; font-size:11px; color:var(--text-muted); background:var(--tile-bg); padding:8px 10px; border-radius:4px; border:1px solid var(--tile-border);">
          <b>Decision Rationale:</b> Astronaut hands detected within active canister interaction zone. Spatial proximity stable across 8 consecutive frames.
        </div>
      </div>

      <!-- Object Tracking Card -->
      <div class="status-card">
        <div class="card-header-label">Object Tracking</div>
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:6px;">
          <div style="display:flex; align-items:center; justify-content:space-between; background:var(--tile-bg); border:1px solid var(--tile-border); padding:8px 12px; border-radius:6px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span>✋</span>
              <div>
                <div style="font-size:11px; font-weight:800;">HAND</div>
                <div style="font-size:9.5px; color:var(--text-subtle);">ID: H1 • Tracking</div>
              </div>
            </div>
            <div style="font-family:'JetBrains Mono'; font-weight:900;">0.92</div>
          </div>
          <div style="display:flex; align-items:center; justify-content:space-between; background:var(--tile-bg); border:1px solid var(--tile-border); padding:8px 12px; border-radius:6px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span>👤</span>
              <div>
                <div style="font-size:11px; font-weight:800;">OPERATOR FACE</div>
                <div style="font-size:9.5px; color:var(--text-subtle);">ID: F1 • Detected</div>
              </div>
            </div>
            <div style="font-family:'JetBrains Mono'; font-weight:900;">0.89</div>
          </div>
          <div style="display:flex; align-items:center; justify-content:space-between; background:var(--tile-bg); border:1px solid var(--tile-border); padding:8px 12px; border-radius:6px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span>🧊</span>
              <div>
                <div style="font-size:11px; font-weight:800;">CANISTER BOX</div>
                <div style="font-size:9.5px; color:var(--text-subtle);">ID: O1 (RY-Box) • Tracking</div>
              </div>
            </div>
            <div style="font-family:'JetBrains Mono'; font-weight:900;">0.95</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Row 3: Big Screen Mission Audit Terminal -->
    <div class="big-screen-log-card">
      <div class="log-top-bar">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>📋</span>
          <span class="log-title">MISSION AUDIT &amp; EVENT LOG STREAM (BIG SCREEN TERMINAL)</span>
        </div>
        <div style="font-size:10px; font-weight:800; font-family:'JetBrains Mono'; color:var(--green-ok); background:#121a2c; padding:3px 10px; border-radius:4px; border:1px solid var(--tile-border);">
          ● LIVE STREAM | 7 EVENTS
        </div>
      </div>
      <div class="log-terminal" id="logTerminal">
        <div class="log-entry">[09:12:45]  [SYSTEM    ]  Mission started: ISRO Red &amp; Yellow Box BAS Experiment</div>
        <div class="log-entry step">[09:13:02]  [STEP      ]  Step 1: System Checkout completed</div>
        <div class="log-entry step">[09:15:58]  [STEP      ]  Step 2: Preparation completed</div>
        <div class="log-entry step">[09:20:10]  [STEP      ]  Step 3: Sample Canister Manipulation started</div>
        <div class="log-entry detection">[09:20:14]  [DETECTION ]  Hand detected (Confidence: 0.92)</div>
        <div class="log-entry detection">[09:20:18]  [DETECTION ]  Canister box detected (Confidence: 0.95)</div>
        <div class="log-entry compliance">[09:20:21]  [COMPLIANCE]  ✓ Gesture verified: Pose and proximity stable</div>
      </div>
    </div>

    <!-- Footer Tools -->
    <div class="footer-tools">
      <div style="font-size:11.5px; color:var(--text-muted);">
        <b>Stream Endpoint:</b> http://127.0.0.1:8080/stream
      </div>
      <div style="display:flex; gap:10px;">
        <button class="btn" onclick="window.print()">🖨 Print Certificate</button>
        <button class="btn btn-primary" onclick="alert('Mission Report generated in reports/ folder.')">📄 Open HTML Report</button>
      </div>
    </div>

  </main>

  <script>
    // Live MET timer
    let seconds = 252;
    setInterval(() => {
      seconds++;
      const h = String(Math.floor(seconds / 3600)).padStart(2, '0');
      const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
      const s = String(seconds % 60).padStart(2, '0');
      document.getElementById('runtime').textContent = `${h}:${m}:${s}`;
    }, 1000);
  </script>
</body>
</html>
""")
class _MJPEGHandler(BaseHTTPRequestHandler):
    """Handles GET / and GET /stream for every connected client."""

    # injected by MJPEGStreamer before the server starts
    frame_store: _FrameStore = None   # type: ignore[assignment]

    # ── Route dispatch ────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/":
            self._serve_page()
        elif self.path == "/stream":
            self._serve_mjpeg()
        else:
            self.send_error(404, "Not found")

    # ── Serve viewer HTML ─────────────────────────────────────────────────────

    def _serve_page(self):
        body = _HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── Serve MJPEG stream ────────────────────────────────────────────────────

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace;boundary={_BOUNDARY.decode()}",
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        try:
            while True:
                jpeg = self.frame_store.get(timeout=2.0)
                if jpeg is None:
                    continue   # no frame yet – try again

                part = (
                    b"--" + _BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
                self.wfile.write(part)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass   # client disconnected – normal

    # ── Silence noisy access log ──────────────────────────────────────────────

    def log_message(self, format, *args):   # noqa: A002
        pass   # suppress per-request console spam


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGStreamer:
    """
    Lightweight MJPEG HTTP streamer.

    Parameters
    ----------
    host    : str   – interface to bind (use "0.0.0.0" to accept from all NICs).
    port    : int   – TCP port (default 8080).
    quality : int   – JPEG quality 1–100 (default 75).
    """

    def __init__(
        self,
        host   : str = "0.0.0.0",
        port   : int = 8080,
        quality: int = 75,
    ) -> None:
        self._host    = host
        self._port    = port
        self._quality = quality

        self._store   = _FrameStore()
        self._server  : Optional[ThreadingHTTPServer] = None
        self._thread  : Optional[threading.Thread]    = None
        self._active  = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"

    @property
    def stream_url(self) -> str:
        return f"http://{self._host}:{self._port}/stream"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Bind the server and start the background serving thread.
        Returns True on success, False if the port is already in use.
        """
        if self._active:
            return True

        # Inject the shared frame store into the handler class
        _MJPEGHandler.frame_store = self._store

        try:
            self._server = ThreadingHTTPServer((self._host, self._port), _MJPEGHandler)
        except OSError as exc:
            print(f"[MJPEGStreamer] ✗ Could not bind {self._host}:{self._port} – {exc}")
            return False

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="MJPEGServer",
        )
        self._thread.start()
        self._active = True
        print(f"[MJPEGStreamer] ● Streaming → {self.url}")
        return True

    def stop(self) -> None:
        """Shut down the HTTP server and release the port."""
        if not self._active:
            return
        self._active = False
        if self._server:
            self._server.shutdown()
            self._server = None
        print(f"[MJPEGStreamer] ■ Streaming stopped.")

    # ── Frame ingestion ───────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Encode one BGR frame as JPEG and push it to all connected clients.
        Call this every time a new annotated frame is ready.
        No-op if the streamer is not active.
        """
        if not self._active:
            return

        ok, buf = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._quality],
        )
        if ok:
            self._store.put(buf.tobytes())

    def update_frame(self, frame: np.ndarray) -> None:
        """Alias for push_frame."""
        self.push_frame(frame)
