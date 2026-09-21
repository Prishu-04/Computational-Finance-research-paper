#!/usr/bin/env python3
"""
Flask backend for the Intraday Direction Ladder website.

Two jobs:
1. Serve the dashboard's data live from whatever is currently on disk in
   data/processed/ and results/tables/ — no pre-baked JSON, no regenerating
   an HTML file by hand. Reuses the exact same build_market_json /
   build_metrics_json functions the standalone generate_dashboard.py script
   uses, so there is one source of truth for "how do we turn pipeline
   output into dashboard data".
2. Let a person upload a new raw_data.csv and run the whole pipeline
   (clean -> features -> windows -> leakage gate -> train Tier 1 ->
   train Tiers 2-5) from the browser, with live progress and logs, instead
   of a terminal.

Run with:  python webapp/app.py
Then open: http://localhost:5000
"""

import io
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.generate_dashboard import (  # noqa: E402
    build_market_json,
    build_metrics_json,
    ensure_full_ladder_table,
)

app = Flask(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

PYTHON = sys.executable  # run pipeline stages with the same interpreter this app is running under

STAGES = [
    {"id": "clean", "label": "Clean data", "cmd": [PYTHON, "src/preprocessing/Datacleaning.py"]},
    {"id": "features", "label": "Engineer features", "cmd": [PYTHON, "src/features/features.py"]},
    {"id": "windows", "label": "Build windows + split", "cmd": [PYTHON, "src/data/windows.py"]},
    {"id": "leakage", "label": "Leakage gate", "cmd": [PYTHON, "src/evaluation/leakage_tests.py"]},
    {"id": "classical", "label": "Train Tier 1 (classical)", "cmd": [PYTHON, "src/models/run_classical_baselines.py"]},
    {"id": "deep", "label": "Train Tiers 2-5 (deep learning)", "cmd": [PYTHON, "src/models/run_deep_learning_tiers.py"]},
]

job_lock = threading.Lock()
job_state = {}


def reset_job_state():
    job_state.update({
        "running": False,
        "stages": [{"id": s["id"], "label": s["label"], "status": "pending"} for s in STAGES],
        "log": [],
        "started_at": None,
        "finished_at": None,
        "error": None,
    })


reset_job_state()


def append_log(line: str):
    ts = datetime.now().strftime("%H:%M:%S")
    job_state["log"].append(f"[{ts}] {line}")
    if len(job_state["log"]) > 4000:
        job_state["log"] = job_state["log"][-4000:]


def run_pipeline():
    job_state["running"] = True
    job_state["started_at"] = datetime.now().isoformat(timespec="seconds")
    job_state["error"] = None
    try:
        for i, stage in enumerate(STAGES):
            job_state["stages"][i]["status"] = "running"
            append_log(f"--- {stage['label']} ---")
            proc = subprocess.run(
                stage["cmd"], cwd=str(PROJECT_ROOT), capture_output=True, text=True
            )
            for line in (proc.stdout or "").splitlines():
                append_log(line)
            if proc.returncode != 0:
                for line in (proc.stderr or "").splitlines():
                    append_log("ERROR: " + line)
                job_state["stages"][i]["status"] = "failed"
                job_state["error"] = f"{stage['label']} failed (exit code {proc.returncode})"
                append_log(f"*** Pipeline stopped: {job_state['error']} ***")
                return
            job_state["stages"][i]["status"] = "done"
            append_log(f"{stage['label']} complete.")

        # Nothing in the pipeline writes the merged ladder table itself
        # (see generate_dashboard.py) -- rebuild it now so /api/metrics is
        # correct the instant the run finishes, no separate step needed.
        ensure_full_ladder_table(RESULTS_DIR)
        append_log("All stages complete. Dashboard is live with the new results.")
    except Exception as exc:  # keep the UI informative instead of hanging
        job_state["error"] = str(exc)
        append_log("*** Unexpected error: " + str(exc) + " ***")
        append_log(traceback.format_exc())
    finally:
        job_state["running"] = False
        job_state["finished_at"] = datetime.now().isoformat(timespec="seconds")


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/pipeline")
def pipeline_page():
    return render_template("pipeline.html")


@app.route("/api/market")
def api_market():
    try:
        return jsonify(build_market_json(DATA_DIR))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/metrics")
def api_metrics():
    try:
        return jsonify(build_metrics_json(RESULTS_DIR))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/meta")
def api_meta():
    def mtime(p: Path):
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else None

    return jsonify({
        "raw_data_updated": mtime(RAW_DIR / "raw_data.csv"),
        "featured_data_updated": mtime(DATA_DIR / "featured_data.csv"),
        "ladder_updated": mtime(RESULTS_DIR / "tables" / "full_model_ladder_results.csv"),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if job_state["running"]:
        return jsonify({"error": "A pipeline run is already in progress."}), 409

    f = request.files.get("dataset")
    if f is None or f.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400

    raw_bytes = f.read()
    import pandas as pd  # local import: keep it out of the hot path for every other route

    try:
        preview = pd.read_csv(io.BytesIO(raw_bytes), nrows=5)
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(preview.columns)
    if missing:
        return jsonify({"error": f"CSV is missing required column(s): {sorted(missing)}"}), 400

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "raw_data.csv"
    if target.exists():
        backup = RAW_DIR / f"raw_data.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        target.replace(backup)
        append_log(f"(previous raw_data.csv backed up to {backup.name})")

    with open(target, "wb") as out:
        out.write(raw_bytes)

    return jsonify({"ok": True, "columns": list(preview.columns), "preview_rows": len(preview)})


@app.route("/api/run", methods=["POST"])
def api_run():
    with job_lock:
        if job_state["running"]:
            return jsonify({"error": "A pipeline run is already in progress."}), 409
        if not (RAW_DIR / "raw_data.csv").exists():
            return jsonify({"error": "No dataset uploaded yet. Upload a CSV first."}), 400
        reset_job_state()
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(job_state)


if __name__ == "__main__":
    print(f"[webapp] project root : {PROJECT_ROOT}")
    print("[webapp] dashboard    : http://localhost:5000/")
    print("[webapp] pipeline UI  : http://localhost:5000/pipeline")
    app.run(host="0.0.0.0", port=5000, debug=False)
