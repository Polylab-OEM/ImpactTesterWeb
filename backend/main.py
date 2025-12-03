#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI backend for U6 / U6-Pro web logger.
Python 3.9 compatible.
- /api/settings   (GET/POST)  load/save JSON settings
- /api/status     (GET)       current acquisition status
- /api/start      (POST)      start acquisition thread
- /api/stop       (POST)      stop acquisition
- /api/live       (GET)       latest decimated samples for live plotting
- /api/stats      (GET)       compute stats from last CSV
- /api/export/report (GET)    generate PDF report with plots + settings
Static/frontend:
- "/" and "/index.html" → frontend/index.html
- "/settings.html"      → frontend/settings.html
- "/static/*"           → frontend/static/*
"""

import os
import json
import threading
import csv
import math
import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .acquisition import (
    AppSettings,
    ChannelConfig,
    TriggerConfig,
    StreamConfig,
    dictify,
    U6Streamer,
)

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

# --- Paths ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
SETTINGS_PATH = os.path.join(BASE_DIR, "u6_web_settings.json")
CLEANUP_AFTER_EXPORT = True  # set True if you want to delete run data after export
app = FastAPI(title="U6 Web Logger")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev; tighten if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mount for CSS, etc.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def load_settings() -> AppSettings:
    if not os.path.exists(SETTINGS_PATH):
        return AppSettings()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return AppSettings()

    channels = [ChannelConfig(**ch) for ch in data.get("channels", [])]
    trig = TriggerConfig(**data.get("trigger", {}))
    stream = StreamConfig(**data.get("stream", {}))

    return AppSettings(
        client=data.get("client", ""),
        item=data.get("item", ""),
        notes=data.get("notes", ""),
        channels=channels or AppSettings().channels,
        trigger=trig,
        stream=stream,
        output_dir=data.get("output_dir", os.path.abspath(".")),
        export_only_after_trigger=bool(data.get("export_only_after_trigger", True)),
    )


def save_settings(settings: AppSettings):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(dictify(settings), f, indent=2)


# Global state for worker + live buffer
settings_lock = threading.Lock()
current_settings: AppSettings = load_settings()

acq_thread: Optional[threading.Thread] = None
acq_worker: Optional[U6Streamer] = None
acq_lock = threading.Lock()
acq_status: str = "idle"
last_csv_path: Optional[str] = None

live_lock = threading.Lock()
live_t: List[float] = []
live_y: List[List[float]] = []  # per channel


def _status_cb(msg: str):
    global acq_status
    acq_status = msg

def _related_paths(csvp: str):
    base, _ = os.path.splitext(csvp)
    return {
        "csv": csvp,
        "events": base + "_events.csv",
        "history": base + "_history.csv",
        "errors": base + "_errors.csv",
    }
def _sample_cb(batch_t, batch_scaled):
    global live_t, live_y
    with live_lock:
        live_t.extend(batch_t)
        n_ch = len(batch_scaled[0]) if batch_scaled else 0
        if not live_y or len(live_y) != n_ch:
            live_y = [[] for _ in range(n_ch)]
        for row in batch_scaled:
            for i in range(n_ch):
                live_y[i].append(row[i])
        MAX_POINTS = 10000
        if len(live_t) > MAX_POINTS:
            trim = len(live_t) - MAX_POINTS
            live_t = live_t[trim:]
            for i in range(len(live_y)):
                live_y[i] = live_y[i][trim:]


# ---------- Frontend routes ----------
def _frontend_file(path_rel: str) -> FileResponse:
    full = os.path.join(FRONTEND_DIR, path_rel)
    if not os.path.exists(full):
        return FileResponse(full, status_code=404)
    return FileResponse(full)


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>U6 Web Logger</h1><p>index.html missing.</p>", status_code=500)
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


@app.get("/index.html", response_class=HTMLResponse)
def index_html():
    return root()


@app.get("/settings.html", response_class=HTMLResponse)
def settings_html():
    path = os.path.join(FRONTEND_DIR, "settings.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>Settings page missing.</h1>", status_code=500)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


# ---------- API: Settings ----------
@app.get("/api/settings")
def get_settings():
    global current_settings
    with settings_lock:
        return dictify(current_settings)


@app.post("/api/settings")
def post_settings(payload: Dict[str, Any]):
    global current_settings
    try:
        ch_list = [
            ChannelConfig(
                ain=int(ch.get("ain", 0)),
                label=str(ch.get("label", "")),
                multiplier=float(ch.get("multiplier", 1.0)),
                enabled=bool(ch.get("enabled", True)),
            )
            for ch in payload.get("channels", [])
        ]

        trig_data = payload.get("trigger", {})
        trig = TriggerConfig(
            enable=bool(trig_data.get("enable", True)),
            trigger_ain=int(trig_data.get("trigger_ain", 0)),
            threshold_volts=float(trig_data.get("threshold_volts", 1.0)),
            rising=bool(trig_data.get("rising", True)),
            use_digital=bool(trig_data.get("use_digital", False)),
            digital_line=int(trig_data.get("digital_line", 0)),
            digital_rising=bool(trig_data.get("digital_rising", True)),
            digital_level_mode=bool(trig_data.get("digital_level_mode", False)),
            digital_active_high=bool(trig_data.get("digital_active_high", True)),
            digital_debounce_ms=int(trig_data.get("digital_debounce_ms", 0)),
            pre_samples=int(trig_data.get("pre_samples", 4000)),
            post_samples=int(trig_data.get("post_samples", 4000)),
            safety_timeout_sec=float(trig_data.get("safety_timeout_sec", 15.0)),
        )

        stream_data = payload.get("stream", {})
        stream = StreamConfig(
            scan_frequency=int(stream_data.get("scan_frequency", 50000)),
            resolution_index=int(stream_data.get("resolution_index", 0)),
            settling_factor=int(stream_data.get("settling_factor", 0)),
            duration_sec=float(stream_data.get("duration_sec", 5.0)),
            ui_plot_hz=int(stream_data.get("ui_plot_hz", 100)),
        )

        new_settings = AppSettings(
            client=str(payload.get("client", "")),
            item=str(payload.get("item", "")),
            notes=str(payload.get("notes", "")),
            channels=ch_list or AppSettings().channels,
            trigger=trig,
            stream=stream,
            output_dir=os.path.abspath(payload.get("output_dir", os.path.abspath("."))),
            export_only_after_trigger=bool(payload.get("export_only_after_trigger", True)),
        )

        with settings_lock:
            current_settings = new_settings
            save_settings(current_settings)

        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------- API: Status / control / live ----------
@app.get("/api/status")
def get_status():
    global acq_status, acq_thread, acq_worker, last_csv_path
    running = acq_thread is not None and acq_thread.is_alive()
    return {
        "running": running,
        "status": acq_status,
        "csv_path": last_csv_path,
    }


@app.post("/api/start")
def start_acquisition():
    global acq_thread, acq_worker, acq_status, last_csv_path

    with acq_lock:
        if acq_thread is not None and acq_thread.is_alive():
            raise HTTPException(status_code=400, detail="Acquisition already running.")

        with settings_lock:
            cfg = current_settings

        if not any(ch.enabled for ch in cfg.channels):
            raise HTTPException(status_code=400, detail="No channels enabled.")

        if cfg.trigger.enable and not cfg.trigger.use_digital:
            enabled_ains = [c.ain for c in cfg.channels if c.enabled]
            if cfg.trigger.trigger_ain not in enabled_ains:
                raise HTTPException(
                    status_code=400,
                    detail="Trigger AIN must be one of the enabled channels (for analog trigger).",
                )

        global live_t, live_y
        with live_lock:
            live_t = []
            live_y = []

        acq_worker = U6Streamer(cfg, status_callback=_status_cb, sample_callback=_sample_cb)

        def runner():
            global last_csv_path
            acq_worker.run()
            last_csv_path = acq_worker.csv_path

        acq_status = "starting..."
        acq_thread = threading.Thread(target=runner, daemon=True)
        acq_thread.start()
        acq_status = "running"

    return {"ok": True}


@app.post("/api/stop")
def stop_acquisition():
    global acq_worker, acq_status
    with acq_lock:
        if acq_worker is None:
            return {"ok": True}
        acq_worker.stop_flag = True
        acq_status = "stopping..."
    return {"ok": True}


@app.get("/api/live")
def get_live():
    with live_lock:
        return {
            "t": list(live_t),
            "y": [list(row) for row in live_y],
        }


# ---------- Stats & Report helpers ----------
def _first_trigger_time(csv_path: str) -> Optional[float]:
    base, _ = os.path.splitext(csv_path)
    ev_path = base + "_events.csv"
    if not os.path.exists(ev_path):
        return None
    try:
        with open(ev_path, "r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            next(rdr, None)
            row = next(rdr, None)
            if row and len(row) >= 3:
                return float(row[2])
    except Exception:
        return None
    return None


def _load_timeseries(csv_path: str, cfg: AppSettings):
    """
    Load time series from CSV and (optionally) crop to a window around the trigger.
    Window is defined via pre_samples / post_samples and scan_frequency.
    """
    if not os.path.exists(csv_path):
        raise RuntimeError("CSV not found.")

    times: List[float] = []
    series: Dict[str, List[float]] = {}
    labels: List[str] = []

    # Window config
    only_after_trigger = bool(cfg.export_only_after_trigger)
    fs = float(cfg.stream.scan_frequency or 0.0)
    pre_samples = int(getattr(cfg.trigger, "pre_samples", 0) or 0)
    post_samples = int(getattr(cfg.trigger, "post_samples", 0) or 0)

    pre_win = pre_samples / fs if fs > 0 else None
    post_win = post_samples / fs if fs > 0 else None
    trig_t = _first_trigger_time(csv_path) if only_after_trigger else None

    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header or "t_seconds" not in header:
            raise RuntimeError("CSV missing t_seconds header.")

        t_idx = header.index("t_seconds")
        scaled_idxs = [(i, name) for i, name in enumerate(header) if name.endswith("_scaled")]
        labels = [name.replace("_scaled", "") for _, name in scaled_idxs]
        for _, name in scaled_idxs:
            series[name] = []

        for row in rdr:
            try:
                t = float(row[t_idx])
            except Exception:
                continue

            # Apply trigger-based window if enabled
            if only_after_trigger and trig_t is not None:
                if pre_win is not None and post_win is not None:
                    if t < (trig_t - pre_win) or t > (trig_t + post_win):
                        continue
                else:
                    # Fallback: just drop pre-trigger samples
                    if t < trig_t:
                        continue

            times.append(t)
            for i, name in scaled_idxs:
                try:
                    series[name].append(float(row[i]))
                except Exception:
                    series[name].append(float("nan"))

    return times, series, labels


def _compute_stats(times, series: Dict[str, List[float]]):
    headers = ["Channel", "N", "Min", "Max", "Max @ t(s)", "Mean", "RMS"]
    rows = []
    for name, y in series.items():
        label = name.replace("_scaled", "")
        finite = [(idx, v) for idx, v in enumerate(y) if math.isfinite(v)]
        n = len(finite)
        if n == 0:
            rows.append([label, 0, "", "", "", "", ""])
            continue
        vals = [v for _, v in finite]
        idx_min = min(finite, key=lambda kv: kv[1])[0]
        idx_max = max(finite, key=lambda kv: kv[1])[0]
        vmin = y[idx_min]
        vmax = y[idx_max]
        t_at_max = times[idx_max] if idx_max < len(times) else ""
        mean = sum(vals) / len(vals)
        rms = math.sqrt(sum(v * v for v in vals) / len(vals))
        rows.append(
            [
                label,
                n,
                "%.6g" % vmin,
                "%.6g" % vmax,
                "%.6g" % t_at_max,
                "%.6g" % mean,
                "%.6g" % rms,
            ]
        )
    return headers, rows


@app.get("/api/stats")
def get_stats():
    global last_csv_path
    if not last_csv_path:
        raise HTTPException(status_code=400, detail="No capture yet.")
    try:
        with settings_lock:
            cfg = current_settings

        times, series, labels = _load_timeseries(
            last_csv_path, cfg
        )
        headers, rows = _compute_stats(times, series)

        # Optional: return window info for the UI
        window_info = {
            "export_only_after_trigger": bool(cfg.export_only_after_trigger),
            "pre_samples": int(getattr(cfg.trigger, "pre_samples", 0) or 0),
            "post_samples": int(getattr(cfg.trigger, "post_samples", 0) or 0),
            "scan_frequency": int(cfg.stream.scan_frequency or 0),
        }

        return {"headers": headers, "rows": rows, "window": window_info}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


from fastapi import Body
import tempfile
import zipfile

def _export_report_impl(include_data: bool = False, overrides: Dict[str, Any] | None = None):
    """
    Core implementation for export report.
    include_data=True → returns a ZIP with PDF + CSV (+events/history/errors if present).
    overrides can contain: client, item, notes, datetime (all optional).
    """
    global last_csv_path
    if not last_csv_path:
        raise HTTPException(status_code=400, detail="No capture yet.")

    csvp = last_csv_path
    overrides = overrides or {}

    with settings_lock:
        cfg = current_settings

    try:
        times, series, labels = _load_timeseries(csvp, cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not times or not series:
        raise HTTPException(status_code=500, detail="No time-series data found in CSV.")

    out_dir = os.path.dirname(csvp)
    base = os.path.splitext(os.path.basename(csvp))[0]
    pdf_path = os.path.join(out_dir, base + "_report.pdf")

    headers, rows = _compute_stats(times, series)

    # --- Describe window (trigger cropping) ---
    only_after = bool(cfg.export_only_after_trigger)
    fs = float(cfg.stream.scan_frequency or 0.0)
    pre_samples = int(getattr(cfg.trigger, "pre_samples", 0) or 0)
    post_samples = int(getattr(cfg.trigger, "post_samples", 0) or 0)
    pre_win = pre_samples / fs if fs > 0 else None
    post_win = post_samples / fs if fs > 0 else None

    if only_after:
        if pre_win is not None and post_win is not None:
            window_str = f"Data window: trigger ± ({pre_win:.4g} s pre, {post_win:.4g} s post)"
        else:
            window_str = "Data window: only samples after trigger"
    else:
        window_str = "Data window: full record (no trigger cropping)"

    # --- Build stats bullet list ---
    lines = []
    for ch, n, vmin, vmax, tmax, mean, rms in rows:
        lines.append(
            f"• {ch}: max {vmax or '-'} @ {tmax or '-'} s, mean {mean or '-'}, RMS {rms or '-'}"
        )
    stats_block = "\n".join(lines) if lines else "(no stats)"

    from datetime import datetime as _dt
    import textwrap as _tw

    # Use overrides if provided, otherwise fall back to cfg / now
    dtstr = overrides.get("datetime") or _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    client = overrides.get("client") if overrides.get("client") is not None else (cfg.client or "-")
    item = overrides.get("item") if overrides.get("item") is not None else (cfg.item or "-")
    notes_raw = overrides.get("notes") if overrides.get("notes") is not None else (cfg.notes or "")
    notes = (notes_raw or "").strip()
    wrapped_notes = "\n".join(_tw.wrap(notes, width=95)) if notes else ""

    # --- Create PDF ---
    with PdfPages(pdf_path) as pdf:
        # PAGE 1 – Header + stats + window info
        fig1 = Figure(figsize=(8.27, 11.69), dpi=100)  # A4 portrait
        ax1 = fig1.add_subplot(111)
        ax1.axis("off")
        text_lines = [
            "Impact Test Report",
            "",
            f"Date/Time: {dtstr}",
            f"Client: {client or '-'}",
            f"Item: {item or '-'}",
            "",
            "Notes:",
            wrapped_notes,
            "",
            f"CSV: {os.path.basename(csvp)}",
            f"Samples (used): {len(times)}",
            "Channels: %s" % (", ".join(labels) if labels else "-"),
            "",
            window_str,
            "",
            "Channel Stats:",
            stats_block,
        ]
        ax1.text(0.04, 0.96, "\n".join(text_lines), va="top", ha="left")
        pdf.savefig(fig1)

        # PAGE 2 – All channels overlay vs time
        fig2 = Figure(figsize=(8.27, 5.0), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.set_title("Scaled Signals vs Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Scaled units")
        for name, y in series.items():
            label = name.replace("_scaled", "")
            ax2.plot(times, y, label=label)
        if series:
            ax2.legend(loc="best")
        pdf.savefig(fig2)

        # PAGE 3+ – One page per channel
        for name, y in series.items():
            label = name.replace("_scaled", "")
            fig_ch = Figure(figsize=(8.27, 5.0), dpi=100)
            ax_ch = fig_ch.add_subplot(111)
            ax_ch.set_title(f"{label} vs Time")
            ax_ch.set_xlabel("Time (s)")
            ax_ch.set_ylabel(label)
            ax_ch.plot(times, y)
            pdf.savefig(fig_ch)

        # LAST PAGE – Settings snapshot as JSON
        fig3 = Figure(figsize=(8.27, 11.69), dpi=100)
        ax3 = fig3.add_subplot(111)
        ax3.axis("off")
        settings_json = json.dumps(dictify(cfg), indent=2)
        ax3.text(
            0.04,
            0.96,
            "Appendix A — Settings Snapshot\n\n" + settings_json,
            va="top",
            ha="left",
            family="monospace",
        )
        pdf.savefig(fig3)

    # If user only wants PDF, return it now
    if not include_data:
        resp = FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(pdf_path),
        )
    else:
        # Bundle PDF + CSV + related files into a temp ZIP
        paths = _related_paths(csvp)
        tmp_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(tmp_fd)
        with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            # Always include PDF
            z.write(pdf_path, arcname=os.path.basename(pdf_path))
            # Include data files if present
            for key, p in paths.items():
                if os.path.exists(p):
                    z.write(p, arcname=os.path.basename(p))

        resp = FileResponse(
            tmp_zip_path,
            media_type="application/zip",
            filename=base + "_report_and_data.zip",
        )

    # Optional cleanup (temp-style behaviour)
    if CLEANUP_AFTER_EXPORT:
        try:
            paths = _related_paths(csvp)
            for p in paths.values():
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass  # best-effort

    return resp


@app.get("/api/export/report")
def export_report_get():
    # Old behaviour: just PDF, using saved settings
    return _export_report_impl(include_data=False, overrides={})


@app.post("/api/export/report")
def export_report_post(body: Dict[str, Any] = Body(...)):
    """
    Body can contain:
      - client: str
      - item: str
      - notes: str
      - datetime: str (e.g. "2025-12-03 17:12:16")
      - include_data: bool (default True)
    """
    include_data = bool(body.get("include_data", True))
    overrides = {
        "client": body.get("client"),
        "item": body.get("item"),
        "notes": body.get("notes"),
        "datetime": body.get("datetime"),
    }
    return _export_report_impl(include_data=include_data, overrides=overrides)
