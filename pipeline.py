import time
import cv2
from identity_manager import IdentityManager
from counter import PersonCounter


def draw_tracks(frame, tracked_persons, counter_info: dict = None, total_unique_count: int = 0):
    """
    Draws bounding boxes, persistent Person ID tags, virtual counting line,
    and live counting status overlay.

    Label format: ID: P001 | Person | 0.91
    """
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    box_color = (255, 144, 30)   # BGR color for bounding box (cyan/orange hue)
    text_color = (255, 255, 255)  # White text

    # 1. Draw Bounding Boxes & Person IDs
    for person in tracked_persons:
        bbox = person['bbox']
        display_id = person.get('person_id', person['track_id'])
        conf = person['conf']

        x1, y1, x2, y2 = map(int, bbox)

        # Draw Bounding Box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

        # Prepare Label: ID: P001 | Person | Y.YY
        label = f"ID: {display_id} | Person | {conf:.2f}"

        # Label Background Box
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - lh - 10)
        label_y2 = max(lh + 10, y1)
        cv2.rectangle(annotated_frame, (x1, label_y1), (x1 + lw + 10, label_y2), box_color, -1)

        # Text Overlay
        text_y = label_y2 - 5
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2,
            cv2.LINE_AA
        )

    # 2. Draw Virtual Counting Line & Overlay Statistics
    if counter_info is not None:
        line_y = counter_info.get('line_y', h // 2)
        in_cnt = counter_info.get('in_count', 0)
        out_cnt = counter_info.get('out_count', 0)
        curr_inside = counter_info.get('current_inside', 0)

        # Virtual Counting Line (Bright Yellow / Cyan)
        line_color = (0, 230, 255)
        cv2.line(annotated_frame, (0, line_y), (w, line_y), line_color, 2)

        # Counting Line Label
        line_label = "COUNTING LINE  [ Top -> Bottom = IN ]"
        (lbl_w, lbl_h), _ = cv2.getTextSize(line_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated_frame, (w - lbl_w - 20, line_y - lbl_h - 6), (w - 5, line_y - 2), (15, 23, 42), -1)
        cv2.putText(annotated_frame, line_label, (w - lbl_w - 12, line_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

        # Top-Left Live Counts Banner Overlay
        stats_banner = f"IN: {in_cnt} | OUT: {out_cnt} | CURRENT: {curr_inside} | UNIQUE: {total_unique_count}"
        (sb_w, sb_h), _ = cv2.getTextSize(stats_banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (10, 10), (10 + sb_w + 20, 10 + sb_h + 16), (15, 23, 42), -1)
        cv2.rectangle(annotated_frame, (10, 10), (10 + sb_w + 20, 10 + sb_h + 16), (59, 130, 246), 1)
        cv2.putText(annotated_frame, stats_banner, (20, 10 + sb_h + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated_frame


def process_frame(
    frame,
    detector,
    tracker,
    metrics,
    conf_threshold: float = 0.5,
    identity_manager: IdentityManager = None,
    counter: PersonCounter = None
):
    """
    Common frame processing pipeline function used for both Video Upload and Webcam inputs.

    Steps:
    1. Read frame
    2. Run YOLOv8 detection
    3. Filter for person class & confidence threshold
    4. Send detections to ByteTrack (temporary Track IDs)
    5. Map temporary Track IDs to persistent Person IDs (IdentityManager / Re-ID)
    6. Evaluate Person IN/OUT Counting across virtual line (PersonCounter)
    7. Draw bounding boxes, persistent IDs, virtual line, and overlay stats
    8. Calculate FPS & update metrics
    9. Return annotated frame & metrics

    Returns:
        annotated_frame: Frame with drawn bounding boxes, persistent IDs, and counting line
        frame_metrics: Dict with current count, total unique count, FPS, IN/OUT counts, occupancy, events
    """
    start_time = time.time()

    # STEP 2 & 3: YOLOv8 Detection & Person Filtering
    person_boxes = detector.detect_persons(frame, conf_threshold=conf_threshold)

    # STEP 4: ByteTrack Tracking
    tracked_persons = tracker.track_persons(person_boxes, frame)

    # STEP 5: Persistent Identity & Person Re-Identification
    if identity_manager is None:
        if not hasattr(tracker, 'identity_manager'):
            tracker.identity_manager = IdentityManager()
        identity_manager = tracker.identity_manager

    tracked_persons = identity_manager.update(tracked_persons, frame)

    # STEP 6: Person IN/OUT Counting (keyed by persistent person_id e.g. 'P001')
    if counter is None:
        if not hasattr(tracker, 'counter'):
            tracker.counter = PersonCounter()
        counter = tracker.counter

    counter_info = counter.update(tracked_persons, frame.shape, metrics.frame_count)

    # STEP 7: Visualization
    total_unique = identity_manager.get_total_unique_count() if identity_manager is not None else tracker.get_total_unique_count()
    annotated_frame = draw_tracks(frame, tracked_persons, counter_info, total_unique)

    end_time = time.time()
    process_duration = end_time - start_time

    # STEP 8: Calculate FPS & Update Metrics
    metrics.update(
        process_duration,
        tracked_persons,
        total_unique
    )

    frame_metrics = {
        'current_people': len(tracked_persons),
        'total_unique': total_unique,
        'fps': metrics.get_fps(),
        'tracks': tracked_persons,
        'in_count': counter_info['in_count'],
        'out_count': counter_info['out_count'],
        'current_inside': counter_info['current_inside'],
        'events_log': counter_info['events_log']
    }

    return annotated_frame, frame_metrics
