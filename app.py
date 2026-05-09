import os
import re
import shutil
import subprocess
from datetime import datetime
from collections import defaultdict
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify

app = Flask(__name__)

UPLOAD_FOLDER   = "uploads"
OUTPUT_FOLDER   = "outputs"
EVIDENCE_FOLDER = "evidence"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, EVIDENCE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Holds the filename across requests (single-user local app)
current_video = {"input": None, "output": None}


# ─── HOME: upload page ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── UPLOAD: save video, go to process page ───────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(file.filename)
    filename = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    output_filename = "output_" + filename
    current_video["input"]  = filename
    current_video["output"] = output_filename

    return redirect(url_for("process"))


# ─── PROCESS PAGE: shows start button ─────────────────────────────────────────

@app.route("/process")
def process():
    if not current_video["input"]:
        return redirect(url_for("index"))
    return render_template("processing.html", filename=current_video["input"])


# ─── RUN: executes main.py detection pipeline ─────────────────────────────────

@app.route("/run", methods=["POST"])
def run():
    input_path  = os.path.join(UPLOAD_FOLDER,  current_video["input"])
    output_path = os.path.join(OUTPUT_FOLDER,  current_video["output"])

    # Build the command; main.py reads INPUT_VIDEO / OUTPUT_VIDEO / EVIDENCE_DIR
    # We pass them as env vars so main.py doesn't need editing
    env = os.environ.copy()
    env["INPUT_VIDEO"]  = input_path
    env["OUTPUT_VIDEO"] = output_path
    env["EVIDENCE_DIR"] = EVIDENCE_FOLDER

    # Run synchronously (blocking) — page will wait until done
    subprocess.run(["python", "main.py"], env=env)

    return redirect(url_for("results"))


# ─── RESULTS: show output video + evidence grouped by person ID ───────────────

@app.route("/results")
def results():
    # Group evidence images by person ID, pick 2 per ID sorted by frame number
    person_evidence = defaultdict(list)

    # Getaway vehicle + plate evidence (separate lists)
    getaway_vehicles = []   # (frame_num, filename)
    getaway_plates   = []   # (frame_num, filename)

    for fname in os.listdir(EVIDENCE_FOLDER):
        # Match only person crops: person_id<N>_frame<M>.jpg
        match = re.match(r"person_id(\d+)_frame(\d+)\.jpg", fname)
        if match:
            pid        = int(match.group(1))
            frame_num  = int(match.group(2))
            person_evidence[pid].append((frame_num, fname))
            continue

        # Match getaway plate crops: getaway_plate_frame<N>_v<M>.jpg
        match_gp = re.match(r"getaway_plate_frame(\d+)_v\d+\.jpg", fname)
        if match_gp:
            getaway_plates.append((int(match_gp.group(1)), fname))
            continue

        # Match getaway vehicle crops: getaway_frame<N>_v<M>.jpg
        match_gv = re.match(r"getaway_frame(\d+)_v\d+\.jpg", fname)
        if match_gv:
            getaway_vehicles.append((int(match_gv.group(1)), fname))
            continue

    # Sort by frame number and keep only the first 2 images per person
    evidence_display = {}
    for pid, items in person_evidence.items():
        items.sort(key=lambda x: x[0])
        evidence_display[pid] = [fname for _, fname in items[:2]]

    # Sort getaway evidence by frame number, keep only the LAST few for display
    getaway_vehicles.sort(key=lambda x: x[0])
    getaway_plates.sort(key=lambda x: x[0])

    getaway_display = {
        "vehicles": [fname for _, fname in getaway_vehicles[-4:]],   # last 4
        "plates":   [fname for _, fname in getaway_plates[-2:]],     # last 2
    }

    return render_template(
        "results.html",
        output_video   = current_video["output"],
        evidence       = evidence_display,       # {person_id: [fname1, fname2]}
        getaway        = getaway_display,         # {vehicles: [...], plates: [...]}
    )


# ─── STATIC SERVING: output video + evidence images ──────────────────────────

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_FOLDER, filename)


# ─── DOWNLOAD HELPERS ─────────────────────────────────────────────────────────

@app.route("/download/video")
def download_video():
    return send_from_directory(
        OUTPUT_FOLDER, current_video["output"], as_attachment=True
    )

@app.route("/download/evidence/<filename>")
def download_evidence(filename):
    return send_from_directory(EVIDENCE_FOLDER, filename, as_attachment=True)


# ─── EXIT: wipe all folders, return to home ───────────────────────────────────

@app.route("/exit")
def exit_app():
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, EVIDENCE_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)

    current_video["input"]  = None
    current_video["output"] = None

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)