# U6 / U6-Pro Impact Logger – Web UI (Python 3.9)

This project turns a LabJack U6 / U6-Pro into a web‑controlled impact logger.

- **Backend:** FastAPI (Python 3.9 compatible) + LabJack `u6` driver.
- **Frontend:** Plain HTML + JS (Chart.js) served by Uvicorn/Starlette.
- **Features:**
  - Multi‑channel analog capture at high sample rates (e.g. 50 kHz).
  - **Analog or digital trigger** with pre/post samples.
  - Ultra‑light despiking (clamps isolated jumps &gt; 5 V).
  - Full‑rate CSV logging with:
    - main data (`u6log_*.csv`)
    - `_events.csv` (trigger events)
    - `_errors.csv` (packet errors & missed samples)
    - `trigger_history.csv` (rolling history)
  - Sub‑sampled **live chart** for smooth UI.
  - **Stats endpoint** and **PDF report export** (summary page + plots + settings snapshot).
  - Settings stored in `u6_web_settings.json`.

## Layout

```text
.
├─ backend/
│  ├─ acquisition.py   # U6Streamer worker (thread) + dataclasses
│  └─ main.py          # FastAPI app
├─ frontend/
│  ├─ index.html       # Live view (chart, start/stop, stats, report)
│  ├─ settings.html    # Settings page with input fields
│  └─ static/
│     └─ css/
│        └─ styles.css
├─ requirements.txt
└─ u6_web_settings.json (auto‑created on first save)
```

## Requirements

- **Python 3.9**
- LabJack U6 / U6‑Pro with the LabJack `u6` driver installed.
- Recommended to use a virtual environment.

Install dependencies:

```bash
cd backend-root
python -m venv .venv
. .venv/Scripts/activate      # Windows
# or
source .venv/bin/activate     # Linux / macOS

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The requirements file contains the `LabJackPython` / `u6` dependency.
> Make sure the LabJack drivers are installed for your OS.

## Running the server

From the project root (the folder that contains `backend/` and `frontend/`):

```bash
uvicorn backend.main:app --reload
```

Uvicorn will start on `http://127.0.0.1:8000` by default.

### Serving the frontend

For local dev you can use Uvicorn's static file mounting, e.g.:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000/index.html` – **Live view**
- `http://127.0.0.1:8000/settings.html` – **Settings page**

If you prefer, you can also serve `frontend/` via a simple HTTP server
and configure CORS, but the simplest is usually to run from the project root
and let Uvicorn serve the static files (see note below).

### Mounting static files (optional)

If Uvicorn is not already serving `/static` and the HTML directly,
you can add this snippet to `backend/main.py`:

```python
from fastapi.staticfiles import StaticFiles
import os, pathlib

frontend_dir = pathlib.Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
```

In the provided version, this is assumed to be configured by how you run
your server (e.g. via `--root-path` or separate static hosting). If you want
it fully self‑contained, add the snippet above near the top of `main.py`.

## How to use

1. Go to **Settings** (`/settings.html`):
   - Configure **output folder**, client, item, and notes.
   - Set up **streaming parameters**:
     - Scan frequency, resolution index, settling factor, duration if no trigger.
     - UI plot rate (sub‑sampled; does not affect CSV).
   - Configure **trigger**:
     - Analog: enable trigger, pick AIN, threshold, rising/falling, pre/post samples.
     - Digital: enable, choose line (FIO/EIO/CIO/MIO), edge or level,
       active high/low.
   - Define up to 4 channels (enabled, AIN, label, multiplier).
   - Click **Save Settings**.

2. Go to **Live** (`/index.html`):
   - Click **Start Capture** to arm the trigger and begin streaming.
   - Watch the **Live Signals** chart (sub‑sampled).
   - Click **Stop** to manually stop early if needed.

3. After a capture:
   - Click **Show Stats** to view per‑channel min/max/mean/RMS.
   - Click **Export PDF Report** to download a report containing:
     - Summary page (client, item, notes, stats).
     - Overlay plot of all channels.
     - One plot per channel.
     - JSON settings snapshot (appendix).

## Exported files

All files are written into the configured **Output Folder**:

- `u6log_YYYYMMDD_HHMMSS.csv` – main capture (index, t_seconds, raw volts, scaled units).
- `u6log_..._events.csv` – trigger events (time, source, threshold, etc.).
- `u6log_..._errors.csv` – packet errors / missed samples.
- `trigger_history.csv` – rolling history of all trigger events.
- `u6log_..._report.pdf` – exported PDF report (via `/api/export/report`).

## Building a stand‑alone EXE

Because the backend is just Python, you can bundle it using **PyInstaller**:

```bash
pip install pyinstaller
pyinstaller --onefile --name u6_web_logger backend/main.py
```

Then your EXE can:

- Start the FastAPI/uvicorn server (you may want a small launcher script).
- Open the default browser to `http://127.0.0.1:8000/index.html`.

You can also bundle the **frontend** folder next to the EXE and mount it
via `StaticFiles` as shown earlier, so everything lives in one directory.

## Notes

- Live chart is deliberately sub‑sampled so logging stays as fast as possible.
  CSV and report always use full sample rate data.
- The despiker is conservative: if the voltage jumps more than 5 V relative
  to the previous sample, that sample is clamped to the previous value.
  You can change this threshold in `backend/acquisition.py` (`SPIKE_DELTA_V`).
- Digital input is configured as **input** using `BitDirWrite(line, 0)` and
  read with `BitStateRead(line)[0]`, with no debounce or sleeps, matching
  your LabJack expectations.
