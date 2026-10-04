"""
event_logger.py
===============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 5: Structured JSON Lines event logger.

Author : SIH26174 Team
Created: 2026-09-21

Design
------
* Every important event is written as a single-line JSON object
  (JSON Lines / JSONL format) to a timestamped file in  logs/.
* One log file is created per session (at EventLogger construction).
* Writing is done from the main thread – each write is a single
  file.write() + flush(), which is fast enough for 30 fps without
  any perceptible lag.
* The logger is intentionally stateless (no internal FSM knowledge) –
  callers pass the fields they want logged.

Log file naming
---------------
    logs/session_YYYYMMDD_HHMMSS.jsonl

Log entry schema
----------------
{
  "timestamp"    : "2026-09-21T20:32:00.123456",  // ISO-8601
  "frame"        : 1234,                           // webcam frame index
  "session_sec"  : 42.1,                           // seconds since session start
  "event"        : "STEP_COMPLETE",                // event category string
  "current_step" : "REMOVE_RED_BOX",               // FSM step name
  "next_step"    : "REMOVE_YELLOW_BOX",            // next expected step (or null)
  "status"       : "CORRECT",                      // TransitionStatus.name
  "colours"      : ["red"],                        // list of detected colours
  "hands"        : 1,                              // number of hands detected
  "message"      : "Step completed correctly."     // human-readable summary
}
"""

from __future__ import annotations

import json
import pathlib
import time
import datetime
from typing import Optional


# ── Log directory ─────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR   = _THIS_DIR.parent.parent / "logs"


# ── Event category constants ──────────────────────────────────────────────────
class EventType:
    SESSION_START     = "SESSION_START"
    SESSION_END       = "SESSION_END"
    STEP_COMPLETE     = "STEP_COMPLETE"
    STEP_SKIPPED      = "STEP_SKIPPED"
    WRONG_ORDER       = "WRONG_ORDER"
    PROTOCOL_COMPLETE = "PROTOCOL_COMPLETE"
    PROTOCOL_RESET    = "PROTOCOL_RESET"
    WAITING           = "WAITING"           # logged only when explicitly requested


# ── Logger ────────────────────────────────────────────────────────────────────

class EventLogger:
    """
    Append-only JSON Lines event logger for one experiment session.

    Parameters
    ----------
    log_dir : pathlib.Path, optional
        Directory to write log files. Defaults to  <project_root>/logs/.
    session_name : str, optional
        Short label embedded in the filename (e.g. "live_protocol").
        Defaults to "session".
    enabled : bool
        Set False to suppress all disk I/O (useful for unit tests).

    Attributes
    ----------
    log_path  : pathlib.Path – resolved path of the current log file
    entry_count : int        – total entries written this session
    """

    def __init__(
        self,
        log_dir: pathlib.Path = LOG_DIR,
        session_name: str = "session",
        enabled: bool = True,
    ):
        self._enabled      = enabled
        self._session_name = session_name
        self._log_dir      = pathlib.Path(log_dir)
        self._start_time   = time.monotonic()
        self._entry_count  = 0
        self._file         = None   # type: ignore

        if not enabled:
            import os
            self.log_path = pathlib.Path(os.devnull)
            return

        self._log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self._log_dir / f"{session_name}_{ts}.jsonl"

        self._file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        print(f"[EventLogger] Logging to: {self.log_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        event: str,
        *,
        frame: int = 0,
        current_step: str = "",
        next_step: Optional[str] = None,
        status: str = "",
        colours: list[str] | None = None,
        hands: int = 0,
        message: str = "",
        experiment_name: str = "",
        extra: dict | None = None,
    ) -> None:
        """
        Write one event entry to the JSONL log file.
        """
        if not self._enabled or self._file is None:
            return

        entry: dict = {
            "timestamp"       : datetime.datetime.now().isoformat(timespec="microseconds"),
            "experiment_name" : experiment_name or self._session_name,
            "frame"           : frame,
            "session_sec"     : round(time.monotonic() - self._start_time, 3),
            "event"           : event,
            "current_step"    : current_step,
            "next_step"       : next_step,
            "status"          : status,
            "colours"         : sorted(colours) if colours else [],
            "hands"           : hands,
            "message"         : message,
        }
        if extra:
            entry.update(extra)

        try:
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._entry_count += 1
        except Exception as exc:
            print(f"[EventLogger] Write error: {exc}")

    def log_session_start(self, protocol_name: str = "", total_steps: int = 0) -> None:
        """Convenience: write the SESSION_START marker."""
        self.log(
            EventType.SESSION_START,
            experiment_name=protocol_name,
            message=f"Experiment session started. Protocol: {protocol_name}",
            extra={"protocol": protocol_name, "total_steps": total_steps},
        )

    def log_session_end(self, frame: int, completed: bool, protocol_name: str = "") -> None:
        """Convenience: write the SESSION_END marker."""
        self.log(
            EventType.SESSION_END,
            frame=frame,
            experiment_name=protocol_name,
            message=f"Session ended. Protocol completed: {completed}",
            extra={"completed": completed, "total_entries": self._entry_count},
        )

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        if self._enabled and self.log_path.exists():
            print(
                f"[EventLogger] Session closed. "
                f"{self._entry_count} entries written to {self.log_path.name}"
            )

    @property
    def entry_count(self) -> int:
        return self._entry_count

    def generate_html_report(self, auto_print: bool = False) -> pathlib.Path:
        """
        Generate a standalone, mission-grade HTML audit report from this session's JSONL log.
        """
        return generate_html_report(self.log_path, auto_print=auto_print)

    def open_html_report(self, auto_print: bool = False) -> pathlib.Path:
        """
        Generate and immediately open the HTML report in default browser.
        """
        report_path = self.generate_html_report(auto_print=auto_print)
        import webbrowser
        webbrowser.open(report_path.as_uri())
        return report_path

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"<EventLogger path={self.log_path.name!r} "
            f"entries={self._entry_count}>"
        )


# ── HTML Report Generator ─────────────────────────────────────────────────────

def generate_html_report(
    log_path: pathlib.Path,
    reports_dir: Optional[pathlib.Path] = None,
    auto_print: bool = False,
) -> pathlib.Path:
    """
    Parse a JSONL session log and generate an executive ISRO / VYOM audit report.
    """
    import webbrowser
    r_dir = reports_dir or (log_path.parent.parent / "reports")
    r_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass

    exp_name = "VYOM BAS Experiment"
    total_steps = 7
    session_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = "00:00"
    completed = False

    for e in entries:
        if e.get("event") == EventType.SESSION_START:
            exp_name = e.get("protocol") or e.get("experiment_name", exp_name)
            total_steps = e.get("total_steps", total_steps)
            session_start_time = e.get("timestamp", session_start_time)[:19].replace("T", " ")
        if e.get("event") == EventType.SESSION_END:
            completed = bool(e.get("completed", False))

    if entries:
        last_sec = entries[-1].get("session_sec", 0.0)
        elapsed_str = f"{int(last_sec // 60):02d}:{int(last_sec % 60):02d}"

    # Filter key step milestones
    milestone_entries = [
        e for e in entries
        if e.get("event") in (EventType.STEP_COMPLETE, EventType.STEP_SKIPPED, EventType.WRONG_ORDER, EventType.PROTOCOL_COMPLETE)
    ]
    if not milestone_entries:
        milestone_entries = entries

    correct_count = sum(1 for e in milestone_entries if e.get("status") in ("CORRECT", "COMPLETED"))
    error_count = sum(1 for e in milestone_entries if e.get("status") in ("SKIPPED", "WRONG_ORDER"))
    compliance_score = (correct_count / max(1, (correct_count + error_count))) * 100.0 if (correct_count + error_count) > 0 else 100.0

    print_script = "<script>window.onload = function() { window.print(); };</script>" if auto_print else ""

    # Build rows
    rows_html = []
    for idx, e in enumerate(milestone_entries, 1):
        ts = e.get("timestamp", "")[11:19]
        sec = e.get("session_sec", 0.0)
        step = e.get("current_step", "—")
        nxt = e.get("next_step", "—")
        stat = e.get("status", e.get("event", "INFO"))
        msg = e.get("message", "—")
        objs = ", ".join(e.get("colours", [])) or "—"
        hands = e.get("hands", 0)

        badge_class = "badge-neutral"
        if stat in ("CORRECT", "COMPLETED", "STEP_COMPLETE"):
            badge_class = "badge-success"
        elif stat in ("SKIPPED", "STEP_SKIPPED"):
            badge_class = "badge-danger"
        elif stat in ("WRONG_ORDER", "WRONG_SEQUENCE"):
            badge_class = "badge-warning"

        rows_html.append(f"""
        <tr>
            <td style="font-weight:700; color:#94a3b8;">#{idx:02d}</td>
            <td style="font-family:monospace; color:#38bdf8;">{ts} (+{sec:.1f}s)</td>
            <td style="font-weight:600; color:#f8fafc;">{step}</td>
            <td style="color:#cbd5e1;">{nxt}</td>
            <td><span class="badge {badge_class}">{stat}</span></td>
            <td style="color:#e2e8f0; font-size:13px;">{msg}</td>
            <td style="color:#00f0ff; font-family:monospace;">{objs}</td>
            <td style="text-align:center; color:#94a3b8;">{hands}</td>
        </tr>
        """)

    table_body = "\n".join(rows_html) if rows_html else "<tr><td colspan='8' style='text-align:center; color:#94a3b8;'>No milestones recorded yet.</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VYOM • Mission Compliance Audit Report</title>
    {print_script}
    <style>
        :root {{
            --bg: #0b0e14;
            --surface: #111622;
            --border: #1e293b;
            --cyan: #00f0ff;
            --sky: #0ea5e9;
            --emerald: #10b981;
            --amber: #f59e0b;
            --rose: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            padding: 32px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .title-block h1 {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 1.5px;
            color: var(--cyan);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .title-block .subtitle {{
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .header-actions {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{ background: #334155; }}
        .btn-cyan {{
            background: rgba(0, 240, 255, 0.15);
            color: var(--cyan);
            border-color: rgba(0, 240, 255, 0.4);
        }}
        .btn-cyan:hover {{ background: rgba(0, 240, 255, 0.25); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .card-val {{
            font-size: 20px;
            font-weight: 800;
            color: var(--text-main);
        }}
        .card-val.cyan {{ color: var(--cyan); }}
        .card-val.emerald {{ color: var(--emerald); }}
        .card-val.amber {{ color: var(--amber); }}
        .table-wrap {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 28px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 13.5px;
        }}
        th {{
            background: #0f172a;
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid #1a2234;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-danger  {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-neutral {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; }}
        .footer {{
            border-top: 1px solid var(--border);
            padding-top: 16px;
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 12px;
        }}
        @media print {{
            body {{ background: #ffffff !important; color: #000000 !important; padding: 10px !important; }}
            .header-actions {{ display: none !important; }}
            .card {{ background: #f8fafc !important; border-color: #cbd5e1 !important; color: #000 !important; }}
            .card-label {{ color: #475569 !important; }}
            .card-val {{ color: #0f172a !important; }}
            .table-wrap {{ border-color: #cbd5e1 !important; }}
            th {{ background: #f1f5f9 !important; color: #334155 !important; }}
            td {{ border-color: #e2e8f0 !important; color: #0f172a !important; }}
            .title-block h1 {{ color: #0284c7 !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-block">
                <h1>VYOM • Mission Compliance Audit Report</h1>
                <div class="subtitle">ON-BOARD PROTOCOL COMPLIANCE ASSISTANT | ISRO - SIH26174</div>
            </div>
            <div class="header-actions">
                <button class="btn btn-cyan" onclick="window.print()">🖨️ Print / Export PDF</button>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-label">Active Experiment</div>
                <div class="card-val cyan" style="font-size:16px;">{exp_name}</div>
            </div>
            <div class="card">
                <div class="card-label">Session Duration</div>
                <div class="card-val">{elapsed_str}</div>
            </div>
            <div class="card">
                <div class="card-label">Sequence Compliance</div>
                <div class="card-val emerald">{compliance_score:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">Final Outcome</div>
                <div class="card-val {'emerald' if completed else 'amber'}">{'VERIFIED' if completed else 'PARTIAL'}</div>
            </div>
        </div>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Timestamp</th>
                        <th>Step Executed</th>
                        <th>Next Expected</th>
                        <th>Outcome</th>
                        <th>Verification Evidence / Reason</th>
                        <th>Objects</th>
                        <th>Hands</th>
                    </tr>
                </thead>
                <tbody>
                    {table_body}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <div>Log Source: {log_path.name}</div>
            <div>Generated by VYOM Edge Assistant • Flight-Heritage Air-Gapped Core</div>
        </div>
    </div>
</body>
</html>
"""

    report_path = r_dir / f"report_{log_path.stem}.html"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(html_content)

    print(f"[EventLogger] Generated HTML report at: {report_path}")
    return report_path


def open_html_report(log_path: pathlib.Path, auto_print: bool = False) -> pathlib.Path:
    import webbrowser
    report_path = generate_html_report(log_path, auto_print=auto_print)
    webbrowser.open(report_path.as_uri())
    return report_path

