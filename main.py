#!/usr/bin/env python3
"""
Fara ‒ Hand-centric cursor control
----------------------------------
• Fist-center steers the mouse (80 % camera → 100 % screen)
• Pinch   (thumb-index OR thumb-middle)
    - quick      → left-click
    - hold 0.2 s → drag
• Thumb-pinky pinch → right-click
• Fist-vertical motion → scroll

Keys
----
c   toggle cursor control
q   quit
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ---------------------------------------------------------------------------

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# Detection / gesture constants
PINCH_PRIMARY_THRESH = 0.06     # thumb–index
PINCH_SECONDARY_THRESH = 0.05   # thumb–middle
PINCH_DRAG_HOLD = 0.20          # s

THUMB_PINKY_THRESH = 0.05       # right-click

SCROLL_COOLDOWN = 0.10          # s
SCROLL_SENSITIVITY = 2

MOVEMENT_SMOOTH = 0.30          # 0..1
MOVEMENT_MIN_PX = 3

CONTROL_ZONE = 0.80             # 80 % centre → 100 % screen


# ---------------------------------------------------------------------------


class HandCursor:
    """Main controller"""

    mp_hands = mp.solutions.hands

    def __init__(self) -> None:
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.30,
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # cursor & gesture state
        self.cursor_x = SCREEN_W // 2
        self.cursor_y = SCREEN_H // 2
        self._last_move_time = time.time()

        self._pinch_state = False
        self._pinch_start = 0.0
        self._dragging = False
        self._pinch_frames = 0

        self._scroll_ref_y: float | None = None
        self._last_scroll_t = 0.0

        self.cursor_enabled = True

    # --------------------------------------------------------------------- #
    # Hand landmark helpers                                                 #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _is_pinch(self, lm) -> bool:
        ti = self._dist(lm[4], lm[8]) < PINCH_PRIMARY_THRESH
        tm = self._dist(lm[4], lm[12]) < PINCH_SECONDARY_THRESH
        return ti or tm

    def _is_thumb_pinky(self, lm) -> bool:
        return self._dist(lm[4], lm[20]) < THUMB_PINKY_THRESH

    # --------------------------------------------------------------------- #
    # Cursor mapping & smoothing                                            #
    # --------------------------------------------------------------------- #
    def _cam_to_screen(self, cx: float, cy: float) -> Tuple[int, int]:
        m = (1 - CONTROL_ZONE) / 2
        lx, ly = m, m
        rx, ry = 1 - m, 1 - m

        nx = 0 if cx < lx else (SCREEN_W - 1) if cx > rx else (cx - lx) / (
            rx - lx
        )
        ny = 0 if cy < ly else (SCREEN_H - 1) if cy > ry else (cy - ly) / (
            ry - ly
        )

        sx = int(nx * SCREEN_W) if isinstance(nx, float) else nx
        sy = int(ny * SCREEN_H) if isinstance(ny, float) else ny
        return sx, sy

    # --------------------------------------------------------------------- #
    def _move_cursor(self, tx: int, ty: int) -> None:
        dx, dy = tx - self.cursor_x, ty - self.cursor_y
        dist = math.hypot(dx, dy)

        if dist < MOVEMENT_MIN_PX:
            return

        if dist < 10:
            self.cursor_x += int(dx * MOVEMENT_SMOOTH)
            self.cursor_y += int(dy * MOVEMENT_SMOOTH)
        else:
            self.cursor_x += int(dx * 0.8)
            self.cursor_y += int(dy * 0.8)

        pyautogui.moveTo(self.cursor_x, self.cursor_y, duration=0)

    # --------------------------------------------------------------------- #
    # Gesture handlers                                                      #
    # --------------------------------------------------------------------- #
    def _handle_pinch(self, pinched: bool) -> str:
        now = time.time()
        if pinched:
            self._pinch_frames += 1
        else:
            self._pinch_frames = 0

        stable = self._pinch_frames >= 2
        action = ""

        if stable and not self._pinch_state:
            # pinch enters
            self._pinch_state = True
            self._pinch_start = now
            action = "pinch start"

        elif stable and self._pinch_state:
            if not self._dragging and now - self._pinch_start >= PINCH_DRAG_HOLD:
                pyautogui.mouseDown()
                self._dragging = True
                action = "drag start"

        elif not stable and self._pinch_state:
            # pinch leaves
            if self._dragging:
                pyautogui.mouseUp()
                action = "drag end"
            else:
                pyautogui.click()
                action = "click"
            self._pinch_state = False
            self._dragging = False

        return action

    def _handle_scroll(self, cy: float, fist: bool) -> str:
        if not fist:
            self._scroll_ref_y = None
            return ""

        now = time.time()
        if now - self._last_scroll_t < SCROLL_COOLDOWN:
            return ""

        if self._scroll_ref_y is None:
            self._scroll_ref_y = cy
            return ""

        diff = (cy - self._scroll_ref_y) * SCROLL_SENSITIVITY * 10
        if abs(diff) > 0.02:
            pyautogui.scroll(int(diff))
            self._last_scroll_t = now
            return f"scroll {'down' if diff>0 else 'up'}"
        return ""

    # --------------------------------------------------------------------- #
    def run(self) -> None:
        fps_clock, fps_counter, fps_val = time.time(), 0, 0.0
        print("Press q to quit – c to toggle cursor control")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            res = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            action = ""
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark

                # ---- cursor ------------------------------------------------
                hx, hy = self._cam_to_screen(*self._fist_center(lm))
                if self.cursor_enabled:
                    self._move_cursor(hx, hy)

                # ---- gestures ----------------------------------------------
                fist = self._is_fist(lm)
                action = self._handle_pinch(self._is_pinch(lm))
                if not action:
                    action = self._handle_scroll(lm[0].y, fist)
                if not action and self._is_thumb_pinky(lm):
                    pyautogui.rightClick()
                    action = "right-click"

            # ---- FPS ------------------------------------------------------
            fps_counter += 1
            if fps_counter == 30:
                now = time.time()
                fps_val = 30 / (now - fps_clock)
                fps_clock, fps_counter = now, 0

            # ---- UI -------------------------------------------------------
            cv2.putText(
                frame,
                f"{'ON ' if self.cursor_enabled else 'OFF'}  |  {fps_val:4.1f} fps  |  {action}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if self.cursor_enabled else (0, 0, 255),
                2,
            )
            cv2.imshow("Fara hand cursor  (q=quit, c=toggle)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.cursor_enabled = not self.cursor_enabled

        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # Small helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fist_center(lm) -> Tuple[float, float]:
        idx = [0, 5, 9, 13, 17]
        cx = sum(lm[i].x for i in idx) / len(idx)
        cy = sum(lm[i].y for i in idx) / len(idx)
        return cx, cy

    def _is_fist(self, lm) -> bool:
        up = sum(self._finger_up(lm, i) for i in range(5))
        return up <= 2

    def _finger_up(self, lm, i: int) -> bool:
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        mcps = [2, 5, 9, 13, 17]
        if i == 0:
            return lm[tips[0]].x > lm[pips[0]].x
        return (lm[tips[i]].y < lm[pips[i]].y) and (lm[tips[i]].y < lm[mcps[i]].y)


# ---------------------------------------------------------------------------


def main() -> None:
    try:
        HandCursor().run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
