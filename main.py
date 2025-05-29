#!/usr/bin/env python3
"""
Advanced Gesture Recognition System with Hand Center Cursor Control
Uses hand center for cursor, fist for click/drag, pinch for scroll, peace for right-click
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from typing import List, Tuple, Optional

# Disable pyautogui fail-safe and set instant movement
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove any pause between pyautogui calls

class AdvancedGestureController:
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand for best performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get screen dimensions for cursor control
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")

        # Control zone settings (center 75% of camera controls 100% of screen)
        self.control_zone_percent = 0.75
        margin = (1 - self.control_zone_percent) / 2

        self.control_left = margin
        self.control_right = 1 - margin
        self.control_top = margin
        self.control_bottom = 1 - margin

        print(f"Control zone: {self.control_left:.1%} to {self.control_right:.1%} horizontally, {self.control_top:.1%} to {self.control_bottom:.1%} vertically")

        # Gesture recognition variables
        self.current_gesture = "None"
        self.frame_count = 0

        # Advanced cursor control settings
        self.enable_cursor_control = True

        # Pinch gesture timing for click vs drag (switched from fist)
        self.pinch_start_time = None
        self.pinch_threshold = 0.2  # Reduced from 0.3 to 0.2 seconds for easier drag
        self.is_dragging = False
        self.last_pinch_state = False
        self.pinch_consistent_frames = 0  # Track consistent pinch detection
        self.pinch_required_frames = 2    # Reduced from 3 to 2 for faster response

        # Right click cooldown
        self.last_right_click_time = 0
        self.right_click_cooldown = 0.5

        # Scroll control (switched to fist)
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1
        self.last_fist_y = None
        self.scroll_sensitivity = 2

        # Previous positions for smooth tracking
        self.prev_hand_center = None
        self.prev_cursor_pos = (self.screen_width // 2, self.screen_height // 2)

        # Movement smoothing settings
        self.movement_threshold = 3  # Minimum pixels to move cursor
        self.smoothing_factor = 0.3  # How much to smooth movement (0 = no smoothing, 1 = no movement)

    def get_hand_center(self, landmarks) -> Tuple[float, float]:
        """Calculate the center point of the hand"""
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)

    def get_fist_center(self, landmarks) -> Tuple[float, float]:
        """Calculate the center of the fist/palm (stable regardless of finger position)"""
        # Use landmarks that don't move when fingers curl/extend
        # Wrist and finger base landmarks: 0 (wrist), 5, 9, 13, 17 (finger bases)
        stable_landmarks = [0, 5, 9, 13, 17]

        x_coords = [landmarks.landmark[i].x for i in stable_landmarks]
        y_coords = [landmarks.landmark[i].y for i in stable_landmarks]

        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)

    def map_to_screen_coordinates(self, cam_x: float, cam_y: float) -> Tuple[int, int]:
        """Map camera coordinates: 80% center = 100% screen, outer 20% clamps to edges"""
        # Define the 80% center zone
        center_zone = 0.8
        margin = (1 - center_zone) / 2  # 10% margin on each side

        zone_left = margin
        zone_right = 1 - margin
        zone_top = margin
        zone_bottom = 1 - margin

        # Handle X coordinate
        if cam_x < zone_left:
            # Left 10% clamps to screen left edge (0%)
            screen_x = 0
        elif cam_x > zone_right:
            # Right 10% clamps to screen right edge (100%)
            screen_x = self.screen_width - 1
        else:
            # Center 80% maps to full screen (0-100%)
            normalized_x = (cam_x - zone_left) / (zone_right - zone_left)
            screen_x = int(normalized_x * self.screen_width)

        # Handle Y coordinate
        if cam_y < zone_top:
            # Top 10% clamps to screen top edge (0%)
            screen_y = 0
        elif cam_y > zone_bottom:
            # Bottom 10% clamps to screen bottom edge (100%)
            screen_y = self.screen_height - 1
        else:
            # Center 80% maps to full screen (0-100%)
            normalized_y = (cam_y - zone_top) / (zone_bottom - zone_top)
            screen_y = int(normalized_y * self.screen_height)

        # Safety clamps
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return screen_x, screen_y

    def is_in_control_zone(self, cam_x: float, cam_y: float) -> bool:
        """Check if hand position is within the control zone"""
        return (self.control_left <= cam_x <= self.control_right and
                self.control_top <= cam_y <= self.control_bottom)

    def draw_control_zones(self, image):
        """Draw the 80% center zone and 20% edge zones"""
        h, w = image.shape[:2]

        # Calculate 80% center zone
        margin = 0.1  # 10% margin on each side = 80% center
        left = int(margin * w)
        right = int((1 - margin) * w)
        top = int(margin * h)
        bottom = int((1 - margin) * h)

        # Draw center zone (80%) in green
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, "80% Center = 100% Screen", (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw edge zones (20%) in yellow
        # Left edge
        cv2.rectangle(image, (0, 0), (left, h), (0, 255, 255), 2)
        # Right edge
        cv2.rectangle(image, (right, 0), (w, h), (0, 255, 255), 2)
        # Top edge
        cv2.rectangle(image, (left, 0), (right, top), (0, 255, 255), 2)
        # Bottom edge
        cv2.rectangle(image, (left, bottom), (right, h), (0, 255, 255), 2)

        cv2.putText(image, "20% Edges = Screen Clamp", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return image

    def move_cursor_smoothly(self, target_x: int, target_y: int) -> Tuple[int, int]:
        """Move cursor with smoothing for small movements"""
        # Calculate distance from current position
        dx = target_x - self.prev_cursor_pos[0]
        dy = target_y - self.prev_cursor_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)

        # Apply different logic based on movement size
        if distance < self.movement_threshold:
            # Very small movement - don't move cursor to avoid jitter
            return self.prev_cursor_pos
        elif distance < 10:
            # Small movement - apply smoothing
            smooth_x = int(self.prev_cursor_pos[0] + dx * self.smoothing_factor)
            smooth_y = int(self.prev_cursor_pos[1] + dy * self.smoothing_factor)
        else:
            # Large movement - move directly but with slight smoothing
            smooth_x = int(self.prev_cursor_pos[0] + dx * 0.8)
            smooth_y = int(self.prev_cursor_pos[1] + dy * 0.8)

        # Update position and move cursor
        self.prev_cursor_pos = (smooth_x, smooth_y)
        try:
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
        except Exception as e:
            print(f"Error moving cursor: {e}")

        return (smooth_x, smooth_y)

    def detect_pinch_gesture(self, landmarks) -> bool:
        """Improved pinch detection with multiple checks"""
        # Check thumb-index pinch (primary)
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        thumb_index_dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )

        # Also check thumb-middle pinch (secondary)
        middle_tip = landmarks.landmark[12]
        thumb_middle_dist = math.sqrt(
            (thumb_tip.x - middle_tip.x)**2 +
            (thumb_tip.y - middle_tip.y)**2
        )

        # More forgiving thresholds
        primary_pinch = thumb_index_dist < 0.06  # Relaxed threshold
        secondary_pinch = thumb_middle_dist < 0.05  # Relaxed threshold

        # Return true if either pinch is detected
        return primary_pinch or secondary_pinch

    def detect_thumb_pinky_pinch(self, landmarks) -> bool:
        """Detect a pinch between thumb tip and pinky tip (used for right-click)"""
        thumb_tip = landmarks.landmark[4]
        pinky_tip = landmarks.landmark[20]

        distance = math.sqrt(
            (thumb_tip.x - pinky_tip.x) ** 2 +
            (thumb_tip.y - pinky_tip.y) ** 2
        )

        return distance < 0.05  # Threshold tuned for reliability

    def get_pinch_distance(self, landmarks) -> float:
        """Get the primary pinch distance for debugging"""
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )

        return distance

    def recognize_gesture(self, landmarks) -> str:
        """Recognize specific gestures based on hand landmarks"""
        if self.detect_fist(landmarks):
            return "Fist"
        # Prioritize pinch over three-finger sign so right-click doesn't trigger during a pinch
        elif self.detect_pinch_gesture(landmarks):
            return "Pinch"
        elif self.detect_three_finger_sign(landmarks):
            return "3 Fingers"
        else:
            finger_count = self.count_fingers(landmarks)
            return f"{finger_count} Fingers"

    def handle_pinch_gesture(self, is_pinch_detected: bool, cursor_pos: Tuple[int, int]) -> str:
        """Handle pinch gesture for click vs drag logic with improved stability"""
        current_time = time.time()
        action = ""

        # Handle pinch consistency (reduce false positives/negatives)
        if is_pinch_detected:
            self.pinch_consistent_frames += 1
        else:
            self.pinch_consistent_frames = 0

        # Consider pinch "stable" only after consistent detection
        stable_pinch = self.pinch_consistent_frames >= self.pinch_required_frames

        # State transitions
        if stable_pinch and not self.last_pinch_state:
            # Stable pinch just started
            self.pinch_start_time = current_time
            self.last_pinch_state = True
            action = "Pinch Started"
            print(f"DEBUG: Pinch started at {current_time}")

        elif stable_pinch and self.last_pinch_state:
            # Pinch continues - check for drag threshold
            if self.pinch_start_time:
                hold_duration = current_time - self.pinch_start_time

                if hold_duration > self.pinch_threshold and not self.is_dragging:
                    # Start dragging
                    try:
                        pyautogui.mouseDown(duration=0)
                        self.is_dragging = True
                        action = "Drag Started"
                        print(f"DEBUG: Drag started after {hold_duration:.2f}s")
                    except Exception as e:
                        print(f"DEBUG: Error starting drag: {e}")

                elif self.is_dragging:
                    action = "Dragging"
                else:
                    action = f"Pinch Held ({hold_duration:.1f}s)"
            else:
                action = "Pinch Held (no start time)"

        elif not stable_pinch and self.last_pinch_state:
            # Pinch ended
            if self.is_dragging:
                # End drag
                try:
                    pyautogui.mouseUp(duration=0)
                    self.is_dragging = False
                    action = "Drag Ended"
                    print(f"DEBUG: Drag ended")
                except Exception as e:
                    print(f"DEBUG: Error ending drag: {e}")

            elif self.pinch_start_time:
                hold_duration = current_time - self.pinch_start_time
                if hold_duration <= self.pinch_threshold:
                    # Quick pinch = click
                    try:
                        pyautogui.click(duration=0)
                        action = "Quick Click"
                        print(f"DEBUG: Quick click after {hold_duration:.2f}s at {cursor_pos}")
                    except Exception as e:
                        print(f"DEBUG: Error clicking: {e}")
                else:
                    action = "Pinch Released (no drag)"

            # Reset state
            self.pinch_start_time = None
            self.last_pinch_state = False
            self.pinch_consistent_frames = 0

        elif not stable_pinch and not self.last_pinch_state:
            # No pinch detected
            if self.is_dragging:
                # Failsafe: end drag if somehow still dragging
                try:
                    pyautogui.mouseUp(duration=0)
                    self.is_dragging = False
                    action = "Drag Force Ended"
                    print(f"DEBUG: Force ended drag (failsafe)")
                except Exception as e:
                    print(f"DEBUG: Error force ending drag: {e}")

        return action

    def handle_fist_scroll(self, landmarks, is_fist: bool) -> str:
        """Handle fist gesture for scrolling (switched from pinch)"""
        if not is_fist:
            self.last_fist_y = None
            return ""

        current_time = time.time()
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return "Scroll Cooldown"

        # Get fist center Y position for scroll direction
        fist_center_x, fist_center_y = self.get_fist_center(landmarks)

        if self.last_fist_y is not None:
            # Calculate scroll direction and amount (direction inverted)
            y_diff = fist_center_y - self.last_fist_y  # Inverted for requested scroll direction

            if abs(y_diff) > 0.02:  # Minimum movement threshold
                scroll_amount = int(y_diff * self.scroll_sensitivity * 10)
                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount)
                    self.last_scroll_time = current_time
                    direction = "Up" if scroll_amount > 0 else "Down"
                    return f"Scroll {direction} ({scroll_amount})"

        self.last_fist_y = fist_center_y
        return "Fist Scroll"

    def handle_right_click(self, is_pinky_pinch: bool, cursor_pos: Tuple[int, int]) -> str:
        """Handle thumb-pinky pinch gesture for right click"""
        if not is_pinky_pinch:
            return ""

        current_time = time.time()
        if current_time - self.last_right_click_time > self.right_click_cooldown:
            pyautogui.rightClick(duration=0)
            self.last_right_click_time = current_time
            print(f"Right click at {cursor_pos}")
            return "Right Click"

        return "Right Click Cooldown"

    def draw_info_panel(self, image, gesture: str, cursor_pos: Tuple[int, int], action: str, fps: float, pinch_distance: float = 0.0):
        """Draw information panel on the image"""
        h, w = image.shape[:2]

        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (500, 320), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        # Add text information
        cv2.putText(image, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Action: {action}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"Cursor: ({cursor_pos[0]}, {cursor_pos[1]})", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show control scheme
        cv2.putText(image, "80% Center = 100% Screen | 20% Edges = Clamp", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, f"Dragging: {'YES' if self.is_dragging else 'NO'}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if self.is_dragging else (255, 255, 255), 2)

        # Pinch state debugging
        cv2.putText(image, f"Pinch Frames: {self.pinch_consistent_frames}/{self.pinch_required_frames}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Pinch distance debugging with updated threshold
        if pinch_distance > 0:
            pinch_color = (0, 255, 0) if pinch_distance < 0.06 else (255, 255, 255)
            cv2.putText(image, f"Pinch Dist: {pinch_distance:.3f} (thresh: 0.06)", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2)

        # Show movement smoothing info
        cv2.putText(image, f"Smooth: {self.smoothing_factor} | Thresh: {self.movement_threshold}px", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show pinch timing for drag
        if self.pinch_start_time and self.last_pinch_state:
            hold_duration = time.time() - self.pinch_start_time
            timer_color = (0, 255, 0) if hold_duration > self.pinch_threshold else (255, 255, 255)
            cv2.putText(image, f"Hold: {hold_duration:.1f}s (drag at {self.pinch_threshold}s)", (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)

        cv2.putText(image, f"FPS: {fps:.1f}", (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(image, "Pinch=Click/Drag | Fist=Scroll | T+Pinky=RightClick", (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def is_finger_up(self, landmarks, finger_id: int) -> bool:
        """Check if a specific finger is up (improved accuracy)"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_mcp = [2, 5, 9, 13, 17]  # Metacarpal joints for better detection

        if finger_id == 0:  # Thumb (special case)
            # Thumb is up if tip is further from wrist than the pip joint
            return landmarks.landmark[finger_tips[0]].x > landmarks.landmark[finger_pips[0]].x
        else:
            # For other fingers, check both pip and mcp joints for better accuracy
            tip_y = landmarks.landmark[finger_tips[finger_id]].y
            pip_y = landmarks.landmark[finger_pips[finger_id]].y
            mcp_y = landmarks.landmark[finger_mcp[finger_id]].y

            # Finger is up if tip is significantly above both pip and mcp
            return tip_y < pip_y and tip_y < mcp_y

    def count_fingers(self, landmarks) -> int:
        """Count the number of extended fingers"""
        fingers_up = 0
        for i in range(5):
            if self.is_finger_up(landmarks, i):
                fingers_up += 1
        return fingers_up

    def detect_fist(self, landmarks) -> bool:
        """Detect fist gesture with improved accuracy"""
        fingers_up = 0
        for i in range(5):
            if self.is_finger_up(landmarks, i):
                fingers_up += 1

        # Relaxed fist detection: allow up to 2 extended fingers for better tolerance
        return fingers_up <= 2

    def detect_three_finger_sign(self, landmarks) -> bool:
        """Detect three-finger sign (index, middle, ring fingers up)"""
        fingers_up = [self.is_finger_up(landmarks, i) for i in range(5)]
        # Three fingers: index (1), middle (2), ring (3) up; thumb and pinky down
        return (fingers_up[1] and fingers_up[2] and fingers_up[3] and
                not fingers_up[0] and not fingers_up[4])

    def run(self):
        """Main execution loop"""
        print("Starting IMPROVED Gesture Recognition System...")
        print("✊  Fist center controls cursor (stable + smoothed)")
        print("🎯 80% center = 100% screen | 20% edges = clamp to edges")
        print("🤏 Improved pinch detection (thumb-index OR thumb-middle)")
        print("🤏 Quick pinch = Click | Hold pinch 0.2s = Drag")
        print("🔍 Real-time drag with smoothed movement")
        print("✊  Fist + move = Scroll up/down")
        print("🤏  Thumb-Pinky pinch = Right click")
        print("💡 'c' = toggle cursor control | 'q' = quit")

        cursor_pos = (self.screen_width // 2, self.screen_height // 2)
        action = ""

        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break

            # FPS calculation
            fps_frame_count += 1
            if fps_frame_count % 30 == 0:
                fps_end_time = time.time()
                current_fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time

            # Flip the frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            gesture = "No Hand Detected"
            action = ""
            pinch_distance = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand

                # Draw minimal hand landmarks
                if not self.enable_cursor_control:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get gesture
                gesture = self.recognize_gesture(hand_landmarks)

                # Always get pinch distance for debugging
                pinch_distance = self.get_pinch_distance(hand_landmarks)

                # FIST CENTER CURSOR CONTROL
                if self.enable_cursor_control:
                    # Get fist center position (stable regardless of finger position)
                    center_x, center_y = self.get_fist_center(hand_landmarks)

                    # Map to screen coordinates
                    target_screen_x, target_screen_y = self.map_to_screen_coordinates(center_x, center_y)

                    # Apply smoothed cursor movement
                    actual_cursor_x, actual_cursor_y = self.move_cursor_smoothly(target_screen_x, target_screen_y)
                    cursor_pos = (actual_cursor_x, actual_cursor_y)

                    # Draw fist center highlight (always green since it's always functional)
                    pixel_x = int(center_x * w)
                    pixel_y = int(center_y * h)

                    # Always show as active (green) since full camera area works
                    cv2.circle(frame, (pixel_x, pixel_y), 20, (0, 255, 0), 4)
                    cv2.putText(frame, "FIST CENTER", (pixel_x + 25, pixel_y - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Show movement info for debugging
                    movement_distance = math.sqrt(
                        (target_screen_x - self.prev_cursor_pos[0])**2 +
                        (target_screen_y - self.prev_cursor_pos[1])**2
                    )
                    if movement_distance > 0:
                        cv2.putText(frame, f"Move: {movement_distance:.1f}px", (pixel_x + 25, pixel_y + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Handle different gestures
                    is_fist = self.detect_fist(hand_landmarks)
                    # Detect primary (thumb-index) pinch first
                    raw_pinch = self.detect_pinch_gesture(hand_landmarks)
                    # Ignore pinch when a fist is detected to prevent conflicting actions
                    is_pinch = False if is_fist else raw_pinch

                    # Detect thumb-pinky pinch for right click
                    raw_pinky_pinch = self.detect_thumb_pinky_pinch(hand_landmarks)
                    # Suppress right-click if fist or primary pinch are active
                    is_pinky_pinch = raw_pinky_pinch and not is_fist and not is_pinch

                    # Pinch handling (click/drag) - switched from fist
                    pinch_action = self.handle_pinch_gesture(is_pinch, cursor_pos)
                    if pinch_action:
                        action = pinch_action

                    # Right-click handling (right click)
                    right_click_action = self.handle_right_click(is_pinky_pinch, cursor_pos)
                    if right_click_action:
                        action = right_click_action

                    # Fist handling (scroll) - switched from pinch
                    fist_action = self.handle_fist_scroll(hand_landmarks, is_fist)
                    if fist_action:
                        action = fist_action

                    # Visual indicators for gestures
                    if is_pinch:
                        cv2.putText(frame, "PINCH!", (w//2 - 60, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                        # Show both pinch distances for debugging
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        middle_tip = hand_landmarks.landmark[12]

                        thumb_index_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                        thumb_middle_dist = math.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)

                        cv2.putText(frame, f"T-I: {thumb_index_dist:.3f}", (w//2 - 80, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"T-M: {thumb_middle_dist:.3f}", (w//2 - 80, 130),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Show drag status prominently
                        if self.is_dragging:
                            cv2.putText(frame, "DRAGGING!", (w//2 - 100, 180),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
                            # Draw drag indicator around the screen
                            cv2.rectangle(frame, (5, 5), (w-5, h-5), (255, 0, 0), 5)
                            # Show that cursor is still moving during drag
                            cv2.putText(frame, "Cursor Active During Drag", (w//2 - 140, 220),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        # Show hold timer for drag threshold
                        if self.pinch_start_time and self.last_pinch_state:
                            hold_duration = time.time() - self.pinch_start_time
                            if hold_duration > 0.1:  # Only show after 0.1s to avoid flicker
                                timer_text = f"Hold: {hold_duration:.1f}s"
                                if hold_duration >= self.pinch_threshold:
                                    timer_text += " - DRAG READY!"
                                cv2.putText(frame, timer_text, (w//2 - 120, 250),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    elif is_fist:
                        cv2.putText(frame, "FIST!", (w//2 - 50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif is_pinky_pinch:
                        cv2.putText(frame, "PINKY PINCH!", (w//2 - 120, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # Draw cursor control status
            if self.enable_cursor_control:
                cv2.putText(frame, "FIST CENTER CURSOR: ON", (w - 280, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "CURSOR CONTROL: OFF", (w - 250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw control zones (80% center + 20% edges)
            frame = self.draw_control_zones(frame)

            # Draw info panel
            frame = self.draw_info_panel(frame, gesture, cursor_pos, action, current_fps, pinch_distance)

            # Display the frame
            cv2.imshow('Advanced Gesture Control', frame)

            # Print info less frequently
            self.frame_count += 1
            if self.frame_count % 120 == 0:
                print(f"FPS: {current_fps:.1f} | Gesture: {gesture} | Action: {action}")

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.enable_cursor_control = not self.enable_cursor_control
                print(f"Cursor control {'enabled' if self.enable_cursor_control else 'disabled'}")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Advanced Gesture Control stopped.")

def main():
    """Main function"""
    try:
        controller = AdvancedGestureController()
        controller.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
