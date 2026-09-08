import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from counter import PersonCounter


def test_counter_scenarios():
    counter = PersonCounter(line_ratio=0.5, direction_mode="A_TO_B_IS_IN", margin=15)
    frame_shape = (480, 640)  # line_y = 240, upper_bound = 225, lower_bound = 255

    # -------------------------------------------------------------
    # TEST 1: One person enters (SIDE_A -> SIDE_B)
    # -------------------------------------------------------------
    # Baseline observation in SIDE_A (Top, cy=100)
    p001_frame1 = [{'person_id': 'P001', 'bbox': [100, 80, 140, 120], 'track_id': 'P001'}]
    res = counter.update(p001_frame1, frame_shape, frame_idx=1)
    assert res['in_count'] == 0
    assert res['out_count'] == 0
    assert res['current_inside'] == 0

    # Crossing into SIDE_B (Bottom, cy=300)
    p001_frame2 = [{'person_id': 'P001', 'bbox': [100, 280, 140, 320], 'track_id': 'P001'}]
    res = counter.update(p001_frame2, frame_shape, frame_idx=2)
    assert res['in_count'] == 1
    assert res['out_count'] == 0
    assert res['current_inside'] == 1
    assert len(res['events_log']) == 1
    assert res['events_log'][0]['direction'] == 'IN'
    assert res['events_log'][0]['person_id'] == 'P001'

    # -------------------------------------------------------------
    # TEST 2: Person remains inside for 100 frames (No duplicate counts)
    # -------------------------------------------------------------
    for f_idx in range(3, 103):
        p001_move = [{'person_id': 'P001', 'bbox': [100, 285 + (f_idx % 5), 140, 325 + (f_idx % 5)], 'track_id': 'P001'}]
        res = counter.update(p001_move, frame_shape, frame_idx=f_idx)

    assert res['in_count'] == 1
    assert res['out_count'] == 0
    assert res['current_inside'] == 1

    # -------------------------------------------------------------
    # TEST 3: Jitter near line inside dead-zone (225 <= cy <= 255)
    # -------------------------------------------------------------
    # Move into dead zone (cy=240)
    p001_dz = [{'person_id': 'P001', 'bbox': [100, 220, 140, 260], 'track_id': 'P001'}]
    res = counter.update(p001_dz, frame_shape, frame_idx=104)
    assert res['in_count'] == 1
    assert res['out_count'] == 0  # No count change while inside dead zone

    # -------------------------------------------------------------
    # TEST 4: Person exits (SIDE_B -> SIDE_A)
    # -------------------------------------------------------------
    # Move completely to SIDE_A (cy=150)
    p001_exit = [{'person_id': 'P001', 'bbox': [100, 130, 140, 170], 'track_id': 'P001'}]
    res = counter.update(p001_exit, frame_shape, frame_idx=105)
    assert res['in_count'] == 1
    assert res['out_count'] == 1
    assert res['current_inside'] == 0
    assert len(res['events_log']) == 2
    assert res['events_log'][1]['direction'] == 'OUT'

    # -------------------------------------------------------------
    # TEST 5: Person enters again after exiting (P001 enters a 2nd time)
    # -------------------------------------------------------------
    p001_reenter = [{'person_id': 'P001', 'bbox': [100, 280, 140, 320], 'track_id': 'P001'}]
    res = counter.update(p001_reenter, frame_shape, frame_idx=106)
    assert res['in_count'] == 2
    assert res['out_count'] == 1
    assert res['current_inside'] == 1

    # -------------------------------------------------------------
    # TEST 6: Temporary track loss & Re-ID mapping back to P001
    # -------------------------------------------------------------
    # ByteTrack temporary ID changed to 99, but IdentityManager resolved to 'P001'
    p001_reid = [{'person_id': 'P001', 'bbox': [100, 290, 140, 330], 'track_id': 'P001', 'temporary_track_id': 99}]
    res = counter.update(p001_reid, frame_shape, frame_idx=107)
    # Still inside, no extra IN event triggered
    assert res['in_count'] == 2
    assert res['out_count'] == 1
    assert res['current_inside'] == 1

    # -------------------------------------------------------------
    # TEST 7 & 8: Multiple / simultaneous crossings (P002 & P003)
    # -------------------------------------------------------------
    # Baseline SIDE_A for P002 and P003
    p_multi_base = [
        {'person_id': 'P002', 'bbox': [200, 50, 240, 90], 'track_id': 'P002'},
        {'person_id': 'P003', 'bbox': [300, 50, 340, 90], 'track_id': 'P003'},
    ]
    counter.update(p_multi_base, frame_shape, frame_idx=108)

    # Simultaneous crossing for P002 & P003 to SIDE_B
    p_multi_cross = [
        {'person_id': 'P002', 'bbox': [200, 280, 240, 320], 'track_id': 'P002'},
        {'person_id': 'P003', 'bbox': [300, 280, 340, 320], 'track_id': 'P003'},
    ]
    res = counter.update(p_multi_cross, frame_shape, frame_idx=109)
    assert res['in_count'] == 4  # P001 (2 times) + P002 + P003 = 4
    assert res['out_count'] == 1
    assert res['current_inside'] == 3  # P001 + P002 + P003 = 3 inside

    print("ALL COUNTER UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_counter_scenarios()
