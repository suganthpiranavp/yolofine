import time
from typing import Dict, List, Tuple, Optional


class PersonCounter:
    """
    PersonCounter manages virtual line-crossing detection for person IN/OUT counting.

    Key Features:
      - Keyed strictly by persistent person_id (e.g. 'P001') from IdentityManager
      - Configurable virtual counting line (horizontal ratio or fixed Y coordinate)
      - Configurable direction mode ('A_TO_B_IS_IN': Top->Bottom = IN, Bottom->Top = OUT)
      - Hysteresis dead-zone protection against bounding box jitter near the line
      - Single-crossing event generation (prevents frame-by-frame duplicate counting)
      - Current occupancy calculation max(0, IN - OUT)
      - Comprehensive IN/OUT event history logging
    """

    def __init__(
        self,
        line_ratio: float = 0.5,
        direction_mode: str = "A_TO_B_IS_IN",
        margin: int = 15
    ):
        """
        Args:
            line_ratio: Position of the horizontal counting line relative to frame height (0.1 to 0.9)
            direction_mode: 'A_TO_B_IS_IN' (Top->Bottom is IN) or 'B_TO_A_IS_IN' (Bottom->Top is IN)
            margin: Dead-zone margin in pixels above and below the counting line
        """
        self.line_ratio = line_ratio
        self.direction_mode = direction_mode
        self.margin = margin

        # State tracking keyed by persistent person_id (e.g. 'P001')
        self.person_sides: Dict[str, str] = {}            # person_id -> 'SIDE_A' or 'SIDE_B'
        self.person_last_centers: Dict[str, Tuple[float, float]] = {}  # person_id -> (cx, cy)

        # Counter metrics
        self.in_count: int = 0
        self.out_count: int = 0
        self.events_log: List[dict] = []

    def update(
        self,
        tracked_persons: List[dict],
        frame_shape: Tuple[int, int],
        frame_idx: int = 0
    ) -> dict:
        """
        Processes current frame's tracked persons and updates crossing counts.

        Args:
            tracked_persons: List of dicts [{'person_id': 'P001', 'bbox': [x1, y1, x2, y2], ...}]
            frame_shape: (height, width) of the video frame
            frame_idx: Current frame index

        Returns:
            dict containing in_count, out_count, current_inside, line_y, events_log
        """
        frame_height, frame_width = frame_shape[:2]
        line_y = int(frame_height * self.line_ratio)

        upper_bound = line_y - self.margin  # SIDE_A (Top) boundary
        lower_bound = line_y + self.margin  # SIDE_B (Bottom) boundary

        for person in tracked_persons:
            # Strictly use persistent person_id (e.g. 'P001')
            person_id = str(person.get("person_id", person["track_id"]))
            bbox = person["bbox"]

            # Calculate centroid of bounding box as specified in Requirement 4
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            # Determine side with dead-zone hysteresis
            if cy < upper_bound:
                raw_side = "SIDE_A"
            elif cy > lower_bound:
                raw_side = "SIDE_B"
            else:
                raw_side = "DEAD_ZONE"

            prev_side = self.person_sides.get(person_id, None)

            if prev_side is None:
                # First observation of this persistent identity: establish baseline side
                if raw_side != "DEAD_ZONE":
                    self.person_sides[person_id] = raw_side
            else:
                # Valid side transition outside dead-zone
                if raw_side != "DEAD_ZONE" and raw_side != prev_side:
                    # Transition detected! Determine IN vs OUT direction
                    if prev_side == "SIDE_A" and raw_side == "SIDE_B":
                        direction = "IN" if self.direction_mode == "A_TO_B_IS_IN" else "OUT"
                    elif prev_side == "SIDE_B" and raw_side == "SIDE_A":
                        direction = "OUT" if self.direction_mode == "A_TO_B_IS_IN" else "IN"
                    else:
                        direction = None

                    if direction == "IN":
                        self.in_count += 1
                    elif direction == "OUT":
                        self.out_count += 1

                    if direction in ("IN", "OUT"):
                        event = {
                            "person_id": person_id,
                            "direction": direction,
                            "frame": frame_idx,
                            "timestamp": time.time()
                        }
                        self.events_log.append(event)

                    # Update persistent side state to prevent duplicate counts
                    self.person_sides[person_id] = raw_side

            self.person_last_centers[person_id] = (cx, cy)

        current_inside = max(0, self.in_count - self.out_count)

        return {
            "in_count": self.in_count,
            "out_count": self.out_count,
            "current_inside": current_inside,
            "line_y": line_y,
            "events_log": self.events_log
        }

    def get_current_inside(self) -> int:
        """Returns current inside occupancy (never negative)."""
        return max(0, self.in_count - self.out_count)

    def reset(self):
        """Resets all counting states and event logs."""
        self.person_sides.clear()
        self.person_last_centers.clear()
        self.events_log.clear()
        self.in_count = 0
        self.out_count = 0
