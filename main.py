import os
import cv2
import time
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

GUN_MODEL_PATH = "models/best.pt"
VEHICLE_MODEL_PATH = "models/vehicle_detection.pt"
PLATE_MODEL_PATH = "models/license_plate.pt"
PERSON_MODEL_PATH = "models/yolov8n.pt"  

INPUT_VIDEO  = os.environ.get("INPUT_VIDEO",  "test3.mp4")
OUTPUT_VIDEO = os.environ.get("OUTPUT_VIDEO", "output3.mp4")
EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", "evidence")

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# =========================
# IOU FUNCTION
# =========================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0

    return intersection / union


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

# YOLOv8n used for person detection + tracking (replaces custom person model)

person_model = YOLO(PERSON_MODEL_PATH)  
gun_model = YOLO(GUN_MODEL_PATH)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

print("Models loaded")


# =========================
# VIDEO SETUP
# =========================

cap = cv2.VideoCapture(INPUT_VIDEO)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Fallback: if avc1 failed to open (not supported on this system), try mp4v
if not out.isOpened():
    print("avc1 codec unavailable, falling back to mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps if fps > 0 else 0

print(f"Video info: {width}x{height} | {fps} FPS | {total_frames} frames | {duration_sec:.1f}s duration")
print("-" * 60)

# =========================
# STATS TRACKING
# =========================

stats = {
    "persons_detected"   : 0,
    "guns_detected"      : 0,
    "vehicles_detected"  : 0,
    "plates_detected"    : 0,
    "armed_persons"      : 0,
    "evidence_saved"     : 0,
    "threat_frames"      : 0,
}

start_time = time.time()

# =========================
# MAIN LOOP
# =========================

frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    annotated = frame.copy()

    # Per-frame threat flag (reset each frame)
    frame_has_threat = False

    # Progress indicator every 10 frames
    if frame_id % 10 == 0 or frame_id == 1:
        elapsed   = time.time() - start_time
        progress  = (frame_id / total_frames * 100) if total_frames > 0 else 0
        eta_sec   = (elapsed / frame_id) * (total_frames - frame_id) if frame_id > 0 else 0
        print(
            f"Frame {frame_id:>5}/{total_frames} "
            f"({progress:5.1f}%) | "
            f"Elapsed: {elapsed:6.1f}s | "
            f"ETA: {eta_sec:6.1f}s | "
            f"Armed: {stats['armed_persons']:>3} | "
            f"Evidence: {stats['evidence_saved']:>3}"
        )

    # =========================
    # PERSON DETECTION + TRACKING
    # Using YOLOv8n with persist=True for stable track IDs
    # classes=[0] limits detection to persons only
    # =========================

    person_boxes = []      # stores [x1, y1, x2, y2]
    person_track_ids = []  # stores track ID per person box

    person_results = person_model.track(
        frame, persist=True, classes=[0], conf=0.5, verbose=False
    )

    if person_results[0].boxes is not None:
        for box in person_results[0].boxes:

            # Skip if tracker hasn't assigned an ID yet
            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            person_boxes.append([x1, y1, x2, y2])
            person_track_ids.append(track_id)

    stats["persons_detected"] += len(person_boxes)


    # =========================
    # GUN DETECTION
    # =========================

    gun_boxes = []

    gun_results = gun_model(frame, conf=0.4, verbose=False)

    if gun_results[0].boxes is not None:

        for box in gun_results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            gun_boxes.append([x1, y1, x2, y2])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 5)

            cv2.putText(
                        annotated,
                        "Gun",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

    stats["guns_detected"] += len(gun_boxes)


    # =========================
    # VEHICLE DETECTION
    # =========================

    vehicle_boxes = []

    vehicle_results = vehicle_model(frame, conf=0.4, verbose=False)

    if vehicle_results[0].boxes is not None:

        for box in vehicle_results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_boxes.append([x1, y1, x2, y2])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 5)

    stats["vehicles_detected"] += len(vehicle_boxes)
    for x1, y1, x2, y2 in vehicle_boxes:
        cv2.putText(
                    annotated,
                    "Vehicle",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )


    # =========================
    # PLATE DETECTION
    # =========================

    plate_boxes = []

    if vehicle_boxes:
        plate_results = plate_model(frame, conf=0.4, verbose=False)

        if plate_results[0].boxes is not None:

            for box in plate_results[0].boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_boxes.append([x1, y1, x2, y2])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        stats["plates_detected"] += len(plate_boxes)
        for x1, y1, x2, y2 in plate_boxes:
            cv2.putText(
                        annotated,
                        "Plate",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )


    # =========================
    # CHECK PERSON HOLDING GUN
    # Now annotates with Track ID for identity tracking
    # =========================

    for idx, pbox in enumerate(person_boxes):

        px1, py1, px2, py2 = pbox
        track_id = person_track_ids[idx]

        holding_gun = False

        for gun_box in gun_boxes:

            if calculate_iou(pbox, gun_box) > 0.05:
                holding_gun = True
                break

        if holding_gun:
            stats["armed_persons"] += 1
            frame_has_threat = True

        color = (0, 0, 255) if holding_gun else (0, 255, 0)

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)

        # Label includes track ID so each person is identifiable across frames
        label = f"Person {track_id}"

        if holding_gun:
            label += " (Gun)"

        cv2.putText(
            annotated,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =========================
        # SAVE EVIDENCE
        # Filenames now include track ID for easy cross-frame linking
        # =========================

        if holding_gun:

            person_crop = frame[py1:py2, px1:px2]

            cv2.imwrite(
                os.path.join(EVIDENCE_DIR, f"person_id{track_id}_frame{frame_id}.jpg"),
                person_crop
            )
            stats["evidence_saved"] += 1
            print(f"  ⚠  THREAT  — Frame {frame_id} | Person ID {track_id} armed | Evidence saved")

            for vehicle_box in vehicle_boxes:

                if calculate_iou(pbox, vehicle_box) > 0.1:

                    vx1, vy1, vx2, vy2 = vehicle_box

                    vehicle_crop = frame[vy1:vy2, vx1:vx2]

                    cv2.imwrite(
                        os.path.join(EVIDENCE_DIR, f"vehicle_id{track_id}_frame{frame_id}.jpg"),
                        vehicle_crop
                    )
                    stats["evidence_saved"] += 1
                    print(f"  ⚠  VEHICLE — Frame {frame_id} | Linked to Person ID {track_id}")

                    for plate_box in plate_boxes:

                        if calculate_iou(vehicle_box, plate_box) > 0.2:

                            ppx1, ppy1, ppx2, ppy2 = plate_box

                            plate_crop = frame[ppy1:ppy2, ppx1:ppx2]

                            cv2.imwrite(
                                os.path.join(EVIDENCE_DIR, f"plate_id{track_id}_frame{frame_id}.jpg"),
                                plate_crop
                            )
                            stats["evidence_saved"] += 1
                            print(f"  ⚠  PLATE   — Frame {frame_id} | Linked to Person ID {track_id}")


    # Count frames where any threat was present
    if frame_has_threat:
        stats["threat_frames"] += 1

    # =========================
    # ON-SCREEN HUD OVERLAY
    # Shows live stats burned into the output video
    # =========================

    hud_lines = [
        f"Frame : {frame_id}/{total_frames}",
        f"People: {len(person_boxes)}",
        f"Guns  : {len(gun_boxes)}",
        f"Veh.  : {len(vehicle_boxes)}",
        f"Plates: {len(plate_boxes)}",
        f"THREAT: {'YES' if frame_has_threat else 'no'}",
    ]

    hud_x, hud_y0, hud_dy = 10, 25, 22

    for i, line in enumerate(hud_lines):
        y = hud_y0 + i * hud_dy
        # Dark background for readability
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (hud_x - 4, y - th - 4), (hud_x + tw + 4, y + 4), (0, 0, 0), -1)
        color = (0, 0, 255) if ("THREAT" in line and frame_has_threat) else (0, 255, 0)
        cv2.putText(annotated, line, (hud_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # =========================
    # SAVE VIDEO FRAME
    # =========================

    out.write(annotated)


cap.release()
out.release()

total_time = time.time() - start_time

# =========================
# DIAGNOSTICS
# =========================

print()
print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)
print(f"  Frames actually written  : {frame_id}")
print(f"  Expected frames          : {total_frames}")
print(f"  FPS used by writer       : {fps}")
print(f"  Expected duration        : {total_frames / fps:.2f}s")
print(f"  Actual processing time   : {total_time:.2f}s")

# Check output file properties via OpenCV
check = cv2.VideoCapture(OUTPUT_VIDEO)
out_frames  = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
out_fps     = check.get(cv2.CAP_PROP_FPS)
out_w       = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
out_h       = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
check.release()

print()
print("OUTPUT FILE (as read back by OpenCV):")
print(f"  Frame count   : {out_frames}")
print(f"  FPS           : {out_fps}")
print(f"  Resolution    : {out_w}x{out_h}")
print(f"  Duration      : {out_frames / out_fps if out_fps > 0 else 'N/A':.2f}s")

# Check file size
file_size = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
print(f"  File size     : {file_size:.2f} MB")

# Use ffprobe if available for ground truth
print()
print("FFPROBE (ground truth — most reliable):")
ret = os.system(f'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,r_frame_rate,duration -of default=noprint_wrappers=1 "{OUTPUT_VIDEO}"')
if ret != 0:
    print("  ffprobe not available")

print("=" * 60)

print()
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
print(f"  Total frames processed : {frame_id}")
print(f"  Total time             : {total_time:.1f}s  ({frame_id/total_time:.1f} FPS)")
print(f"  Output video           : {OUTPUT_VIDEO}")
print()
print("DETECTION TOTALS (cumulative across all frames):")
print(f"  Persons detected       : {stats['persons_detected']}")
print(f"  Guns detected          : {stats['guns_detected']}")
print(f"  Vehicles detected      : {stats['vehicles_detected']}")
print(f"  Plates detected        : {stats['plates_detected']}")
print()
print("THREAT SUMMARY:")
print(f"  Armed-person events    : {stats['armed_persons']}")
print(f"  Threat frames          : {stats['threat_frames']}  ({stats['threat_frames']/frame_id*100:.1f}% of video)")
print(f"  Evidence files saved   : {stats['evidence_saved']}")
print(f"  Evidence directory     : {EVIDENCE_DIR}/")
print("=" * 60)