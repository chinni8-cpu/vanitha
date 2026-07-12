# -*- coding: utf-8 -*-
import os
import cv2
import json
import shutil
import urllib.request
import numpy as np
import mediapipe as mp
import time

# Configuration
DATABASE_FILE = "gesture_database.json"
DATASET_DIR = "google_gestures"

GESTURE_URLS = {
    "fist": [
        "https://images.unsplash.com/photo-1498673394965-85cb14905c89?w=640",
        "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=640",
        "https://images.unsplash.com/photo-1481819613568-3701ccd7f175?w=640",
        "https://images.unsplash.com/photo-1516962215378-7fa2e137ae93?w=640"
    ],
    "peace": [
        "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=640",
        "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=640",
        "https://images.unsplash.com/photo-151675150278-77136aed6920?w=640",
        "https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=640"
    ],
    "thumbs_up": [
        "https://images.unsplash.com/photo-1582213782179-e0d53f98f2ca?w=640",
        "https://images.unsplash.com/photo-1568602471122-7832951cc4c5?w=640",
        "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=640",
        "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=640"
    ],
    "ok": [
        "https://images.unsplash.com/photo-1601412436009-d964bd02edbc?w=640",
        "https://images.unsplash.com/photo-1516575150278-77136aed6920?w=640",
        "https://images.unsplash.com/photo-1603575450878-7cfab4266d3c?w=640",
        "https://images.unsplash.com/photo-1544717305-2782549b5136?w=640"
    ],
    "palm": [
        "https://images.unsplash.com/photo-1490195117352-aa267f1750d2?w=640",
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=640",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640",
        "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=640"
    ]
}

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def download_images():
    """Download references images from Google / Wikimedia if they don't exist"""
    print("\n--- Step 1: Checking and Downloading Reference Images ---")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    for gesture, urls in GESTURE_URLS.items():
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        for idx, url in enumerate(urls, 1):
            target_path = os.path.join(gesture_dir, f"{idx}.jpg")
            if not os.path.exists(target_path):
                print(f"Downloading {gesture} image {idx}/4...")
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as response, open(target_path, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"Warning: Failed to download {url}. Error: {e}")
            else:
                print(f"{gesture} image {idx}/4 already exists.")


def normalize_landmarks(landmarks):
    """Translate to wrist and scale normalize hand landmarks to be size/position invariant"""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 1. Center around wrist (landmark 0)
    wrist = coords[0]
    coords = coords - wrist
    
    # 2. Scale normalize relative to the distance between wrist (0) and middle finger MCP joint (9)
    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords = coords / scale
    else:
        # Fallback to max distance if joint 9 is overlaying joint 0
        scale = np.max(np.linalg.norm(coords, axis=1))
        if scale > 0:
            coords = coords / scale
            
    return coords.flatten().tolist()


def fallback_local_dataset():
    """Copy hand gesture images from local Fingers dataset if downloading fails"""
    local_dataset_path = "dataset/hand_dataset/archive/train"
    if not os.path.exists(local_dataset_path):
        print("Local Fingers dataset not found at dataset/hand_dataset/archive/train. Skipping local fallback.")
        return False
        
    print("\n--- Fallback: Using Local Fingers Dataset ---")
    # Mapping finger count to gesture names
    gesture_map = {
        "0": "fist",
        "1": "point",
        "2": "peace",
        "3": "three",
        "4": "four",
        "5": "palm"
    }
    
    # Track copies
    copies_per_gesture = {g: 0 for g in gesture_map.values()}
    max_copies = 4
    
    all_files = os.listdir(local_dataset_path)
    all_files.sort()  # Sort to keep selection deterministic
    
    for file_name in all_files:
        if not file_name.lower().endswith(".png"):
            continue
            
        # Fingers dataset format: <uuid>_<finger_count><L/R>.png
        parts = file_name.split("_")
        if len(parts) < 2:
            continue
        suffix = parts[-1]
        if len(suffix) < 6:
            continue
        finger_count_char = suffix[0]
        
        if finger_count_char in gesture_map:
            gesture = gesture_map[finger_count_char]
            if copies_per_gesture[gesture] < max_copies:
                src_path = os.path.join(local_dataset_path, file_name)
                dest_dir = os.path.join(DATASET_DIR, gesture)
                os.makedirs(dest_dir, exist_ok=True)
                
                copies_per_gesture[gesture] += 1
                dest_path = os.path.join(dest_dir, f"local_{copies_per_gesture[gesture]}.png")
                try:
                    shutil.copy(src_path, dest_path)
                    print(f"Copied {file_name} -> {dest_path}")
                except Exception as e:
                    print(f"Error copying {file_name}: {e}")
                
    return True


def train_classifier():
    """Extract hand landmarks from reference images and save them as templates"""
    print("\n--- Step 2: Training / Extracting Landmark Templates ---")
    database = {}
    
    # Try local fallback if directory is empty or downloads failed
    has_downloaded_images = False
    for gesture in GESTURE_URLS.keys():
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        if os.path.exists(gesture_dir) and len([f for f in os.listdir(gesture_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) > 0:
            has_downloaded_images = True
            break
            
    if not has_downloaded_images:
        print("No downloaded images found. Attempting local dataset fallback...")
        fallback_local_dataset()
        
    # Use static image mode for better detection on independent photos
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4
    ) as hands:
        
        # Read folders in DATASET_DIR dynamically
        gestures_to_process = os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else GESTURE_URLS.keys()
        
        for gesture in gestures_to_process:
            gesture_dir = os.path.join(DATASET_DIR, gesture)
            if not os.path.isdir(gesture_dir):
                continue
            
            database[gesture] = []
            
            for file_name in os.listdir(gesture_dir):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(gesture_dir, file_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # Take the first detected hand
                    landmarks = results.multi_hand_landmarks[0].landmark
                    normalized = normalize_landmarks(landmarks)
                    database[gesture].append(normalized)
                    print(f"Successfully processed: {img_path}")
                else:
                    print(f"Warning: No hand detected in reference image: {img_path}")
                    
    # Check if database contains enough templates
    total_samples = sum(len(v) for v in database.values())
    if total_samples < 3:
        print("\nDatabase contains very few templates. Forcing local dataset fallback training...")
        if fallback_local_dataset():
            # Re-train using local images
            # Clear current database structure first to avoid infinite recursion if local files also fail
            # (though they shouldn't as they are clean hand images)
            database = {}
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.4
            ) as hands:
                gestures_to_process = os.listdir(DATASET_DIR)
                for gesture in gestures_to_process:
                    gesture_dir = os.path.join(DATASET_DIR, gesture)
                    if not os.path.isdir(gesture_dir):
                        continue
                    database[gesture] = []
                    for file_name in os.listdir(gesture_dir):
                        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            continue
                        img_path = os.path.join(gesture_dir, file_name)
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)
                        if results.multi_hand_landmarks:
                            landmarks = results.multi_hand_landmarks[0].landmark
                            normalized = normalize_landmarks(landmarks)
                            database[gesture].append(normalized)
                            print(f"Successfully processed (fallback): {img_path}")
                        else:
                            print(f"Warning: No hand detected in fallback image: {img_path}")

    # Save the database
    with open(DATABASE_FILE, 'w') as f:
        json.dump(database, f, indent=4)
    print(f"\nTraining completed! Saved gesture templates to {DATABASE_FILE}")


def load_database():
    """Load gesture database from JSON file"""
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
    return {}


def draw_hud(frame, gesture, distance, fps):
    """Draw a visual premium futuristic HUD overlay on the frame"""
    height, width, _ = frame.shape
    
    # 1. Overlay semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (20, 20, 20), -1)
    # Apply alpha blend
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # 2. Title and Status info
    cv2.putText(frame, "REAL-TIME GESTURE HUD", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width - 150, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Status bar divider line
    cv2.line(frame, (0, 80), (width, 80), (0, 255, 255), 1, cv2.LINE_AA)
    
    # 3. Gesture Prediction banner at bottom
    overlay_bottom = frame.copy()
    cv2.rectangle(overlay_bottom, (0, height - 90), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(overlay_bottom, 0.5, frame, 0.5, 0, frame)
    cv2.line(frame, (0, height - 90), (width, height - 90), (0, 255, 255), 1, cv2.LINE_AA)
    
    if gesture:
        # Confidence score represented as inverse of distance (threshold is 0.5)
        # Scale to percentage
        confidence = max(0.0, min(100.0, (1.0 - (distance / 0.5)) * 100))
        
        # Emoji labels for display
        emoji_map = {
            "fist":            "✊  FIST",
            "palm":            "🖐  PALM",
            "peace":           "✌  PEACE",
            "thumbs_up":       "👍  THUMBS UP",
            "thumbs_down":     "👎  THUMBS DOWN",
            "ok":              "👌  OK",
            "rock_on":         "🤘  ROCK ON!",
            "call_me":         "🤙  CALL ME",
            "love_you":        "🤟  LOVE YOU",
            "point_up":        "☝  POINT UP",
            "crossed_fingers": "🤞  CROSSED FINGERS",
            "good_luck":       "🤞  GOOD LUCK",
            "three":           "3️⃣  THREE",
            "four":            "4️⃣  FOUR",
        }
        # Color palette for predicted gesture (BGR)
        color_map = {
            "fist":            (0,   0,   255),  # Red
            "palm":            (0,   165, 255),  # Orange
            "peace":           (255, 255,   0),  # Cyan
            "thumbs_up":       (0,   255,   0),  # Green
            "thumbs_down":     (0,   80,  180),  # Dark Red
            "ok":              (255,   0, 255),  # Magenta
            "rock_on":         (0,   200, 255),  # Gold
            "call_me":         (255, 180,   0),  # Light Blue
            "love_you":        (180,   0, 255),  # Purple
            "point_up":        (0,   255, 200),  # Teal
            "crossed_fingers": (30,  200, 100),  # Olive
            "good_luck":       (30,  200, 100),  # Olive
            "three":           (200, 100,   0),  # Blue
            "four":            (100,   0, 200),  # Violet
        }
        display_label = emoji_map.get(gesture, gesture.upper())
        color = color_map.get(gesture, (0, 255, 255))
        
        # Draw gesture text
        cv2.putText(frame, display_label, (30, height - 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        # Draw confidence progress bar
        bar_x = int(width * 0.6)
        bar_y = height - 48
        bar_w = 200
        bar_h = 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int((confidence / 100) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"CONF: {confidence:.0f}%", (bar_x + bar_w + 15, bar_y + 13), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "GESTURE: SEARCHING...", (30, height - 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (128, 128, 128), 1, cv2.LINE_AA)
        
    # Keyboard shortcut instructions
    cv2.putText(frame, "[Q] Quit | [R] Retrain templates | [C] Capture current pose", 
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


# Gesture emoji guide printed on startup
GESTURE_GUIDE = """
  ✊  FIST              - Close all fingers
  👍  THUMBS UP         - Thumb up, all fingers closed
  👎  THUMBS DOWN       - Thumb down, all fingers closed
  ✌  PEACE             - Index + Middle up, others closed
  🤞  CROSSED FINGERS   - Index + Middle up & tips close (cross them!)
  🤞  GOOD LUCK         - Same as crossed fingers
  🖐  PALM              - All five fingers open
  👌  OK                - Thumb + Index tips touching, others open
  🤘  ROCK ON           - Index + Pinky up, Middle + Ring closed
  🤙  CALL ME           - Thumb + Pinky out, others closed
  🤟  LOVE YOU (ILY)    - Thumb + Index + Pinky out, others closed
  ☝  POINT UP          - Only Index finger raised
  3️⃣  THREE             - Index + Middle + Ring open
  4️⃣  FOUR              - Index + Middle + Ring + Pinky open
"""


def classify_by_rules(landmarks):
    """Rule-based gesture classifier using 21-point MediaPipe hand landmarks."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist   = coords[0]

    # ── Finger open/closed: compare tip distance vs PIP distance from wrist ──
    def dist(a, b):
        return np.linalg.norm(coords[a] - coords[b])

    def tip_above_pip(tip, pip):
        """True when fingertip is further from wrist than the PIP joint."""
        return dist(tip, 0) > dist(pip, 0)

    index_open  = tip_above_pip(8,  6)
    middle_open = tip_above_pip(12, 10)
    ring_open   = tip_above_pip(16, 14)
    pinky_open  = tip_above_pip(20, 18)

    # ── Thumb: use y-axis direction (y increases downward in image coords) ──
    # Thumb extended = tip further from wrist than MCP joint
    thumb_extended = dist(4, 0) > dist(2, 0)
    # Thumb UP  = tip is ABOVE wrist in image (lower y value)
    thumb_up    = coords[4][1] < coords[2][1]
    # Thumb DOWN = tip is BELOW wrist (higher y value)
    thumb_down  = coords[4][1] > coords[2][1]

    # ── Measurements ──
    hand_size = dist(9, 0)  # wrist to middle MCP

    # OK: thumb tip touching index tip
    thumb_idx_dist = dist(4, 8)
    is_ok = (thumb_idx_dist / hand_size < 0.32) if hand_size > 0 else False

    # Crossed fingers: index and middle both up but tips very close together
    idx_mid_tip_dist = dist(8, 12)
    is_crossed = (idx_mid_tip_dist / hand_size < 0.28) if hand_size > 0 else False

    # ═══════════════════════════════════════════════════════════════
    # Priority-ordered classification
    # ═══════════════════════════════════════════════════════════════

    # OK  (before peace/palm — thumb-index contact wins)
    if is_ok and middle_open and ring_open and pinky_open:
        return "ok"

    # CROSSED FINGERS / GOOD LUCK  (before peace — tips close together)
    if index_open and middle_open and not ring_open and not pinky_open and is_crossed:
        return "crossed_fingers"

    # LOVE YOU  (ILY) — thumb + index + pinky out, middle + ring closed
    if thumb_extended and index_open and not middle_open and not ring_open and pinky_open:
        return "love_you"

    # CALL ME — thumb + pinky out, middle fingers closed
    if thumb_extended and not index_open and not middle_open and not ring_open and pinky_open:
        return "call_me"

    # ROCK ON — index + pinky up, middle + ring closed
    if index_open and not middle_open and not ring_open and pinky_open:
        return "rock_on"

    # THUMBS UP — thumb pointing up, all fingers closed
    if thumb_extended and thumb_up and not index_open and not middle_open and not ring_open and not pinky_open:
        return "thumbs_up"

    # THUMBS DOWN — thumb pointing down, all fingers closed
    if thumb_extended and thumb_down and not index_open and not middle_open and not ring_open and not pinky_open:
        return "thumbs_down"

    # FIST — all fingers closed, thumb folded
    if not index_open and not middle_open and not ring_open and not pinky_open:
        return "fist"

    # PALM — all fingers open
    if index_open and middle_open and ring_open and pinky_open:
        return "palm"

    # PEACE / VICTORY — index + middle up, ring + pinky closed
    if index_open and middle_open and not ring_open and not pinky_open:
        return "peace"

    # FOUR — four fingers open, thumb folded
    if index_open and middle_open and ring_open and pinky_open and not thumb_extended:
        return "four"

    # THREE — index + middle + ring, pinky closed
    if index_open and middle_open and ring_open and not pinky_open:
        return "three"

    # POINT UP — only index finger raised
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "point_up"

    return None


def run_camera_inference():
    """Start the webcam feed, detect hands, and classify hand gestures in real-time"""
    print("\n--- Step 3: Starting Real-Time Camera Recognition ---")
    database = load_database()
    if not database:
        print("Warning: Gesture database is empty. Please run training first or use 'c' key to capture gestures live.")
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure your camera is connected and not in use by another app.")
        return

    print("\nWebcam started successfully.")
    print("Press 'q' to quit.")
    print("Press 'r' to re-run training/downloading.")
    print("Press 'c' to capture current hand shape and save as a custom gesture.")

    # MediaPipe Hands instance for real-time tracking
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        
        last_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip horizontally for mirrored view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            recognized_gesture = None
            min_dist = float('inf')
            landmarks_to_save = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on frame with custom colors
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),  # yellow joints
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)     # green connections
                    )
                    
                    # Normalize landmarks for prediction
                    query_v = normalize_landmarks(hand_landmarks.landmark)
                    landmarks_to_save = query_v
                    
                    # Compare to template database (Nearest Neighbor classifier)
                    for gesture, templates in database.items():
                        for template in templates:
                            dist = np.linalg.norm(np.array(query_v) - np.array(template))
                            if dist < min_dist:
                                min_dist = dist
                                recognized_gesture = gesture
                                
                    # If template match is not confident, try rule-based classifier
                    if min_dist > 0.45 or recognized_gesture is None:
                        rules_gesture = classify_by_rules(hand_landmarks.landmark)
                        if rules_gesture:
                            recognized_gesture = rules_gesture
                            min_dist = 0.2  # Set a mock confident distance (0.2 means 60% confidence)
                        else:
                            recognized_gesture = None
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - last_time)
            last_time = curr_time
            
            # Render visual overlays
            draw_hud(frame, recognized_gesture, min_dist, fps)
            
            # Show processed frame
            cv2.imshow("Hand Gesture Recognition HUD", frame)
            
            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\nRetraining requested...")
                cap.release()
                cv2.destroyAllWindows()
                download_images()
                train_classifier()
                # Restart camera loop
                run_camera_inference()
                return
            elif key == ord('c'):
                if landmarks_to_save is not None:
                    print("\n--- Live Capture Mode ---")
                    g_name = input("Enter gesture name to register this pose: ").strip().lower()
                    if g_name:
                        if g_name not in database:
                            database[g_name] = []
                        database[g_name].append(landmarks_to_save)
                        # Save back to database file
                        with open(DATABASE_FILE, 'w') as f:
                            json.dump(database, f, indent=4)
                        print(f"Successfully saved current hand pose to database as gesture: '{g_name}'")
                else:
                    print("\nWarning: No hand detected in frame to capture. Please place your hand in front of the camera.")
                    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 60)
    print("  HAND GESTURE RECOGNITION SYSTEM")
    print("=" * 60)
    print(GESTURE_GUIDE)
    print("=" * 60)

    # Check if database exists, if not, download & train
    if not os.path.exists(DATABASE_FILE) or not os.path.exists(DATASET_DIR):
        print("No gesture database or images found. Initializing...")
        download_images()
        train_classifier()

    # Run the real-time webcam visual application
    run_camera_inference()
