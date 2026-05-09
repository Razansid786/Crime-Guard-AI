import os
import cv2
import shutil
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
# PERSISTENT ARMED-PERSON TRACKING
# Once a person is seen holding a gun, their track ID is
# remembered forever so their box stays red in all future frames.
# =========================

armed_person_ids = set()

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
    # PLATE DETECTION (runs ONLY on detected vehicle crops)
    # Instead of scanning the full frame, we crop each vehicle
    # region, run the plate model on that crop, and remap the
    # plate coordinates back to the full frame.
    # =========================

    plate_boxes = []

    for vx1, vy1, vx2, vy2 in vehicle_boxes:
        # Add padding around the vehicle crop so edge-plates aren't clipped
        pad = 10
        crop_x1 = max(0, vx1 - pad)
        crop_y1 = max(0, vy1 - pad)
        crop_x2 = min(width, vx2 + pad)
        crop_y2 = min(height, vy2 + pad)

        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if vehicle_crop.size == 0:
            continue

        plate_results = plate_model(vehicle_crop, conf=0.4, verbose=False)

        if plate_results[0].boxes is not None:
            for box in plate_results[0].boxes:
                # Coordinates are relative to the crop; remap to full frame
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                px1 += crop_x1
                py1 += crop_y1
                px2 += crop_x1
                py2 += crop_y1

                plate_boxes.append([px1, py1, px2, py2])

                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    "Plate",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    stats["plates_detected"] += len(plate_boxes)


    # =========================
    # CHECK PERSON HOLDING GUN  (with persistence)
    # If a person is seen holding a gun even once, their track ID
    # is added to armed_person_ids and they stay flagged (red box)
    # for the rest of the video.
    # =========================

    for idx, pbox in enumerate(person_boxes):

        px1, py1, px2, py2 = pbox
        track_id = person_track_ids[idx]

        # Check if this person is holding a gun RIGHT NOW
        holding_gun_now = False
        for gun_box in gun_boxes:
            if calculate_iou(pbox, gun_box) > 0.05:
                holding_gun_now = True
                break

        # If detected now, remember this person permanently
        if holding_gun_now:
            armed_person_ids.add(track_id)

        # A person is "armed" if they were EVER seen with a gun
        is_armed = track_id in armed_person_ids

        if is_armed:
            frame_has_threat = True

        # Only count a NEW armed-person event the first time the
        # IoU fires (not on every subsequent frame of persistence)
        if holding_gun_now:
            stats["armed_persons"] += 1

        # Red if armed (current or past), green otherwise
        color = (0, 0, 255) if is_armed else (0, 255, 0)

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)

        # Label includes track ID so each person is identifiable across frames
        label = f"Person {track_id}"
        if is_armed:
            label += " [ARMED]"

        # White text for normal, bright red for armed
        text_color = (0, 0, 255) if is_armed else (255, 255, 255)
        cv2.putText(
            annotated,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )


        # =========================
        # SAVE EVIDENCE
        # Evidence is saved only when the gun is actively detected
        # this frame (not every persistence frame).
        # =========================

        if holding_gun_now:

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
    # GETAWAY VEHICLE MARKING
    # For each armed person (current or persistent), find the
    # vehicle they overlap with RIGHT NOW and mark it "GETAWAY".
    # This recalculates every frame — only the latest overlap
    # counts, so old vehicles go back to normal.
    # =========================

    getaway_vehicle_indices = set()

    for idx, pbox in enumerate(person_boxes):
        track_id = person_track_ids[idx]
        if track_id not in armed_person_ids:
            continue
        for v_idx, vehicle_box in enumerate(vehicle_boxes):
            if calculate_iou(pbox, vehicle_box) > 0.1:
                getaway_vehicle_indices.add(v_idx)

    for v_idx in getaway_vehicle_indices:
        gx1, gy1, gx2, gy2 = vehicle_boxes[v_idx]
        # Thick red border drawn over the existing blue one
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 5)
        # Black background for label readability
        label_text = "GETAWAY"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (gx1, gy1 - th - 14), (gx1 + tw + 8, gy1), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label_text,
            (gx1 + 4, gy1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # ── Save getaway vehicle crop to evidence ──
        getaway_crop = frame[gy1:gy2, gx1:gx2]
        if getaway_crop.size > 0:
            cv2.imwrite(
                os.path.join(EVIDENCE_DIR, f"getaway_frame{frame_id}_v{v_idx}.jpg"),
                getaway_crop
            )
            stats["evidence_saved"] += 1
            print(f"  🚗 GETAWAY — Frame {frame_id} | Vehicle saved to evidence")

            # Also save any plate found inside this getaway vehicle
            for plate_box in plate_boxes:
                if calculate_iou([gx1, gy1, gx2, gy2], plate_box) > 0.2:
                    ppx1, ppy1, ppx2, ppy2 = plate_box
                    plate_crop = frame[ppy1:ppy2, ppx1:ppx2]
                    if plate_crop.size > 0:
                        cv2.imwrite(
                            os.path.join(EVIDENCE_DIR, f"getaway_plate_frame{frame_id}_v{v_idx}.jpg"),
                            plate_crop
                        )
                        stats["evidence_saved"] += 1
                        print(f"  🚗 PLATE   — Frame {frame_id} | Getaway plate saved")

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
if shutil.which('ffprobe'):
    print("FFPROBE (ground truth — most reliable):")
    os.system(f'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,r_frame_rate,duration -of default=noprint_wrappers=1 "{OUTPUT_VIDEO}"')
else:
    print("FFPROBE: skipped (ffprobe not installed — install FFmpeg to enable)")

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