import cv2
import mediapipe as mp
import numpy as np
import time
import math

# 1. Setup Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       
    (0, 5), (5, 6), (6, 7), (7, 8),       
    (5, 9), (9, 10), (10, 11), (11, 12),  
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)                               
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

canvas = None
px, py = 0, 0 

colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Pink
]
color_idx = 0
current_color = colors[color_idx]

thicknesses = [2, 5, 10, 25] 
thick_idx = 1
current_thickness = thicknesses[thick_idx]

# --- UNIFIED STATE MACHINE VARIABLES ---
active_gesture = None
gesture_start_time = 0
ui_cooldown = 0
canvas_history = [] 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)
    
    save_image_flag = False 
    filename = ""

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            lm = []
            for landmark in hand_landmarks:
                lm.append((int(landmark.x * w), int(landmark.y * h)))
                
            for connection in HAND_CONNECTIONS:
                start_point = lm[connection[0]]
                end_point = lm[connection[1]]
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
            for joint_x, joint_y in lm:
                cv2.circle(frame, (joint_x, joint_y), 4, (0, 255, 0), -1)

            # Detect Fingers
            thumb_extended = 1 if math.hypot(lm[4][0]-lm[9][0], lm[4][1]-lm[9][1]) > math.hypot(lm[3][0]-lm[9][0], lm[3][1]-lm[9][1]) else 0
            fingers_up = [
                thumb_extended,
                1 if lm[8][1] < lm[6][1] else 0,   
                1 if lm[12][1] < lm[10][1] else 0, 
                1 if lm[16][1] < lm[14][1] else 0, 
                1 if lm[20][1] < lm[18][1] else 0  
            ]
            
            index_x, index_y = lm[8]

            # --- STEP 1: IDENTIFY GESTURE ---
            gesture = None
            if fingers_up == [0, 1, 0, 0, 0]: gesture = "DRAW"
            elif fingers_up == [0, 1, 1, 0, 0]: gesture = "COLOR"
            elif fingers_up == [0, 1, 0, 0, 1]: gesture = "THICKNESS"
            elif fingers_up == [0, 1, 1, 1, 0]: gesture = "SCREENSHOT"
            elif fingers_up == [0, 1, 1, 1, 1]: gesture = "UNDO"
            elif fingers_up == [0, 0, 0, 0, 1]: gesture = "CLEAR"
            elif fingers_up == [1, 1, 1, 1, 1]: gesture = "DUSTER"

            # Reset drawing coordinates if we aren't drawing
            if gesture != "DRAW":
                px, py = 0, 0

            # --- STEP 2: PROCESS TIMED UI GESTURES ---
            timed_gestures = ["COLOR", "THICKNESS", "SCREENSHOT", "UNDO", "CLEAR"]

            if gesture in timed_gestures:
                # Draw constant visual hints so you know the gesture is recognized
                if gesture == "COLOR":
                    cv2.circle(frame, lm[8], 10, current_color, -1)
                    cv2.circle(frame, lm[12], 10, current_color, -1)
                elif gesture == "THICKNESS":
                    cv2.circle(frame, lm[8], current_thickness + 2, current_color, -1)
                    cv2.circle(frame, lm[20], current_thickness + 2, current_color, -1)

                if ui_cooldown == 0:
                    if active_gesture != gesture:
                        # New gesture detected, start the clock!
                        active_gesture = gesture
                        gesture_start_time = time.time()
                    else:
                        # Gesture is being held, calculate elapsed time
                        elapsed = time.time() - gesture_start_time
                        trigger_time = 1.2
                        
                        # Determine where to draw the loading ring
                        if gesture == "CLEAR": anchor = lm[20] # Pinky
                        elif gesture == "UNDO": anchor = lm[9] # Palm
                        else: anchor = lm[8] # Index finger
                        
                        # Draw the loading ring
                        progress = int((elapsed / trigger_time) * 360)
                        cv2.ellipse(frame, anchor, (40, 40), 0, 0, progress, (0, 255, 255), 4)
                        cv2.putText(frame, f"Hold for {gesture}...", (anchor[0] - 50, anchor[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Trigger the action when time is up!
                        if elapsed >= trigger_time:
                            if gesture == "COLOR":
                                color_idx = (color_idx + 1) % len(colors)
                                current_color = colors[color_idx]
                            elif gesture == "THICKNESS":
                                thick_idx = (thick_idx + 1) % len(thicknesses)
                                current_thickness = thicknesses[thick_idx]
                            elif gesture == "SCREENSHOT":
                                filename = f"AR_Drawing_{int(time.time())}.jpg"
                                save_image_flag = True
                            elif gesture == "UNDO":
                                if len(canvas_history) > 0:
                                    canvas = canvas_history.pop()
                            elif gesture == "CLEAR":
                                canvas_history.append(canvas.copy())
                                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                
                            # Reset the tracker and trigger a cooldown so it doesn't double-fire
                            active_gesture = None
                            ui_cooldown = 30
            else:
                # If doing anything else, reset the timer
                active_gesture = None
                
                # --- STEP 3: PROCESS INSTANT GESTURES ---
                if gesture == "DRAW":
                    if px == 0 and py == 0:
                        # Save state for the Undo function right before a new line starts
                        canvas_history.append(canvas.copy())
                        if len(canvas_history) > 15: 
                            canvas_history.pop(0)
                        px, py = index_x, index_y
                    
                    cv2.line(canvas, (px, py), (index_x, index_y), current_color, current_thickness)
                    px, py = index_x, index_y 
                    cv2.circle(frame, (index_x, index_y), current_thickness + 2, current_color, -1)
                    
                elif gesture == "DUSTER":
                    palm_x, palm_y = lm[9] 
                    cv2.circle(canvas, (palm_x, palm_y), 60, (0, 0, 0), -1)
                    cv2.circle(frame, (palm_x, palm_y), 60, (255, 255, 255), 2)
    else:
        px, py = 0, 0
        active_gesture = None

    if ui_cooldown > 0:
        ui_cooldown -= 1

    # Apply Opaque Paint Logic
    paint_locations = canvas.any(axis=-1)
    frame[paint_locations] = canvas[paint_locations]
    
    if save_image_flag:
        cv2.imwrite(filename, frame)
        print(f"Saved screenshot as {filename}")

    cv2.imshow('AIR Writing App by Mehran', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()