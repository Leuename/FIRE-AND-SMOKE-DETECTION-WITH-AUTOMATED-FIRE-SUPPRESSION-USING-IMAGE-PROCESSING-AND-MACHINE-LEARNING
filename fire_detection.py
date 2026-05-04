import cv2
import serial
import time
import numpy as np
from ultralytics import YOLO
import torch

# =====================================================
# CAMERA CONFIGURATION
# =====================================================
# Tapo Camera RTSP (ML Detection)
USERNAME = "Thesis"
PASSWORD = "12345678"
IP_ADDRESS = "192.168.1.157"
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/stream1"

# FLIR Lepton (Thermal Tracking)
FLIR_CAMERA_INDEX = 1

# ML Model
MODEL_PATH = "yolo11n_best.pt"

# =====================================================
# ARDUINO CONNECTION
# =====================================================
ARDUINO_PORT = 'com4'
BAUD_RATE = 9600

# =====================================================
# DETECTION PARAMETERS
# =====================================================
BASE_FIRE_THRESHOLD = 80
MIN_FIRE_AREA = 1
SMOOTHING_KERNEL = 5
ADAPTIVE_SENSITIVITY = 40

# Body heat rejection
MAX_FIRE_AREA = 15000000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0

# Flicker detection
USE_FLICKER_FILTER = False
MIN_INTENSITY_VARIANCE = 15
TEMPORAL_FRAMES = 8
intensity_history = []

# Smoothing
SMOOTHING_FACTOR = 0.6
last_x, last_y = None, None

# Motor tracking
current_x_angle = 0.0
current_y_angle = 0.0
X_AXIS_LIMIT_RIGHT = 150.0
X_AXIS_LIMIT_LEFT = 120.0
Y_AXIS_LIMIT = 30

# Control parameters
DEADZONE = 20
MIN_MOVE_ANGLE = 3
MAX_MOVE_ANGLE_X = 6
MAX_MOVE_ANGLE_Y = 2

frame_center_x = 400
frame_center_y = 320

# Servo state
servo_state = False

# Search mode
SEARCH_MODE = False
SEARCH_STEP_X = 30
SEARCH_STEP_Y = 10
search_direction_x = 1
search_direction_y = 1
search_pattern_state = 'horizontal'

# System state
SYSTEM_STATE = "MONITORING"  # MONITORING, TRACKING, SUPPRESSING
NO_FIRE_COUNTER = 0
NO_FIRE_THRESHOLD = 30  # Frames without fire before returning to monitoring

# IR Camera minimum search time
IR_SEARCH_START_TIME = None
MIN_IR_SEARCH_TIME = 75.0  # Minimum 30 seconds of IR searching before giving up

# =====================================================
# MACHINE LEARNING FIRE DETECTION (TAPO CAMERA)
# =====================================================
def initialize_ml_model():
    """Initialize YOLO model for fire detection"""
    try:
        print(f"🤖 Loading ML model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"✅ ML Model loaded on device: {device}")
        return model
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        return None

def monitor_with_tapo(model, rtsp_cap):
    """
    Monitor for fire using Tapo camera with YOLO
    Returns: (fire_detected, confidence, annotated_frame)
    """
    try:
        ret, frame = rtsp_cap.read()
       
        if not ret or frame is None:
            return False, 0.0, None
       
        # Run YOLO inference
        results = model(frame, conf=0.45, verbose=False)
       
        # Check detections
        fire_detected = False
        max_confidence = 0.0
       
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                fire_detected = True
                confidences = result.boxes.conf.cpu().numpy()
                max_confidence = float(np.max(confidences))
       
        # Get annotated frame
        frame_with_boxes = results[0].plot() if len(results) > 0 else frame
       
        # Add status overlay
        status_color = (0, 255, 0) if fire_detected else (200, 200, 200)
        cv2.putText(frame_with_boxes, "TAPO MONITORING MODE",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_with_boxes, f"Fire: {'DETECTED' if fire_detected else 'None'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        if fire_detected:
            cv2.putText(frame_with_boxes, f"Confidence: {max_confidence:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
       
        return fire_detected, max_confidence, frame_with_boxes
       
    except Exception as e:
        print(f"❌ Tapo monitoring error: {e}")
        return False, 0.0, None

# =====================================================
# THERMAL TRACKING FUNCTIONS (FLIR CAMERA)
# =====================================================
def calculate_intensity_variance(roi):
    """Calculate intensity variance for flicker detection"""
    return np.std(roi)

def check_flame_characteristics(contour, gray_frame):
    """
    Additional checks to distinguish flame from body heat
    Returns: (is_valid, confidence, reason)
    """
    area = cv2.contourArea(contour)
   
    if area < MIN_FIRE_AREA:
        return False, 0.0, "Too small"
    if area > MAX_FIRE_AREA:
        return False, 0.0, "Too large (body?)"
   
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
   
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False, 0.0, "Invalid shape"
   
    confidence = 0.5
    reason = "Valid"
   
    if area < 10:
        confidence += 0.3
        reason = "Small flame"
   
    if USE_FLICKER_FILTER:
        roi = gray_frame[y:y+h, x:x+w]
        intensity_var = calculate_intensity_variance(roi)
       
        global intensity_history
        intensity_history.append(intensity_var)
        if len(intensity_history) > TEMPORAL_FRAMES:
            intensity_history.pop(0)
       
        temporal_variance = np.std(intensity_history) if len(intensity_history) > 2 else 0
       
        if intensity_var < MIN_INTENSITY_VARIANCE and temporal_variance < 5:
            return False, 0.0, "No flicker (body?)"
       
        confidence += 0.2
        reason += " + flicker"
   
    return True, confidence, reason

def detect_flame_adaptive(frame):
    """
    Adaptive flame detection using FLIR thermal camera
    Returns: (x, y, detected, debug_frame, stats)
    """
    global last_x, last_y
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (SMOOTHING_KERNEL, SMOOTHING_KERNEL), 0)
   
    avg_temp = np.mean(gray_smooth)
    fire_threshold = min(255, avg_temp + ADAPTIVE_SENSITIVITY)
    fire_threshold = max(BASE_FIRE_THRESHOLD, fire_threshold)
   
    _, thresh = cv2.threshold(gray_smooth, fire_threshold, 255, cv2.THRESH_BINARY)
   
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    debug_frame = thermal.copy()
   
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    # Display stats
    state_color = {
        "MONITORING": (200, 200, 200),
        "TRACKING": (0, 255, 255),
        "SUPPRESSING": (0, 255, 0)
    }
    cv2.putText(debug_frame, f"IR CAM - {SYSTEM_STATE}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color.get(SYSTEM_STATE, (255, 255, 255)), 2)
    cv2.putText(debug_frame, f"Thermal: {int(avg_temp)} | Threshold: {int(fire_threshold)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_frame, f"X: {current_x_angle:.1f}° | Y: {current_y_angle:.1f}°",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(debug_frame, f"Servo: {'ON' if servo_state else 'OFF'}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if servo_state else (0, 0, 255), 2)
   
    if SEARCH_MODE:
        cv2.putText(debug_frame, "🔍 SEARCHING",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
   
    if len(contours) == 0:
        last_x, last_y = None, None
        intensity_history.clear()
        cv2.putText(debug_frame, "NO THERMAL TARGET", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        stats = {'avg_temp': avg_temp, 'threshold': fire_threshold, 'contours': 0}
        return None, None, False, debug_frame, stats
   
    # Filter contours
    valid_flames = []
    for contour in contours:
        is_valid, confidence, reason = check_flame_characteristics(contour, gray_smooth)
       
        if is_valid:
            area = cv2.contourArea(contour)
            valid_flames.append((contour, confidence, area, reason))
            cv2.drawContours(debug_frame, [contour], -1, (0, 255, 255), 2)
        else:
            cv2.drawContours(debug_frame, [contour], -1, (0, 165, 255), 1)
   
    if len(valid_flames) == 0:
        last_x, last_y = None, None
        cv2.putText(debug_frame, f"DETECTED {len(contours)} - ALL REJECTED", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        stats = {'avg_temp': avg_temp, 'threshold': fire_threshold, 'contours': len(contours), 'valid': 0}
        return None, None, False, debug_frame, stats
   
    # Select best flame
    best_flame = max(valid_flames, key=lambda x: x[1] * x[2])
    contour, confidence, area, reason = best_flame
   
    # Get centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, False, debug_frame, {}
   
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
   
    # Apply smoothing
    if last_x is not None and last_y is not None:
        cx = int(last_x * SMOOTHING_FACTOR + cx * (1 - SMOOTHING_FACTOR))
        cy = int(last_y * SMOOTHING_FACTOR + cy * (1 - SMOOTHING_FACTOR))
   
    last_x, last_y = cx, cy
   
    # Draw detection
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.circle(debug_frame, (cx, cy), 15, (0, 0, 255), 3)
    cv2.line(debug_frame, (cx-25, cy), (cx+25, cy), (0, 0, 255), 2)
    cv2.line(debug_frame, (cx, cy-25), (cx, cy+25), (0, 0, 255), 2)
   
    # Draw deadzone
    cv2.circle(debug_frame, (frame_center_x, frame_center_y), DEADZONE, (255, 0, 255), 2)
   
    cv2.putText(debug_frame, f"🔥 TARGET LOCKED", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(debug_frame, f"Position: ({cx}, {cy}) | Area: {int(area)}px", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
   
    # Center crosshair
    h, w = debug_frame.shape[:2]
    cv2.line(debug_frame, (w//2-30, h//2), (w//2+30, h//2), (255, 255, 255), 2)
    cv2.line(debug_frame, (w//2, h//2-30), (w//2, h//2+30), (255, 255, 255), 2)
   
    stats = {
        'avg_temp': avg_temp,
        'threshold': fire_threshold,
        'contours': len(contours),
        'valid': len(valid_flames),
        'area': area,
        'confidence': confidence
    }
   
    return cx, cy, True, debug_frame, stats

def calculate_movement(target_x, target_y):
    """Calculate motor movement needed with axis limits"""
    global current_x_angle, current_y_angle
   
    error_x = target_x - frame_center_x
    error_y = target_y - frame_center_y
   
    if abs(error_x) < DEADZONE and abs(error_y) < DEADZONE:
        return None, None, 0, False, True
   
    if abs(error_x) > abs(error_y):
        degrees_per_pixel = 60.0 / 640.0
        degrees = abs(error_x) * degrees_per_pixel
        degrees = max(MIN_MOVE_ANGLE, min(degrees, MAX_MOVE_ANGLE_X))
        degrees = int(degrees)
       
        if error_x > 0:
            direction = 'R'
            new_angle = current_x_angle + degrees
            if new_angle > X_AXIS_LIMIT_RIGHT:
                print(f"⚠️ X-AXIS RIGHT LIMIT ({X_AXIS_LIMIT_RIGHT}°)")
                return None, None, 0, False, False
        else:
            direction = 'L'
            new_angle = current_x_angle - degrees
            if new_angle < -X_AXIS_LIMIT_LEFT:
                print(f"⚠️ X-AXIS LEFT LIMIT ({-X_AXIS_LIMIT_LEFT}°)")
                return None, None, 0, False, False
       
        current_x_angle = new_angle
        return 1, direction, degrees, True, False
    else:
        degrees_per_pixel = 40.0 / 480.0
        degrees = abs(error_y) * degrees_per_pixel
        degrees = max(MIN_MOVE_ANGLE, min(degrees, MAX_MOVE_ANGLE_Y))
        degrees = int(degrees)
       
        if error_y > 0:
            direction = 'R'
            new_angle = current_y_angle + degrees
        else:
            direction = 'L'
            new_angle = current_y_angle - degrees
       
        if abs(new_angle) > Y_AXIS_LIMIT:
            print(f"⚠️ Y-AXIS LIMIT ({Y_AXIS_LIMIT}°)")
            return None, None, 0, False, False
       
        current_y_angle = new_angle
        return 2, direction, degrees, True, False

def perform_search(arduino):
    """Perform sweep search with axis limits"""
    global current_x_angle, current_y_angle, search_direction_x, search_direction_y, search_pattern_state
   
    if search_pattern_state == 'horizontal':
        next_angle_x = current_x_angle + (SEARCH_STEP_X * search_direction_x)
       
        if search_direction_x > 0:
            if next_angle_x > X_AXIS_LIMIT_RIGHT:
                search_direction_x *= -1
                search_pattern_state = 'vertical'
                return perform_search(arduino)
        else:
            if next_angle_x < -X_AXIS_LIMIT_LEFT:
                search_direction_x *= -1
                search_pattern_state = 'vertical'
                return perform_search(arduino)
       
        direction = 'R' if search_direction_x > 0 else 'L'
        send_motor_command(arduino, 1, direction, SEARCH_STEP_X)
        current_x_angle = next_angle_x
       
    else:
        next_angle_y = current_y_angle + (SEARCH_STEP_Y * search_direction_y)
       
        if abs(next_angle_y) > Y_AXIS_LIMIT:
            search_direction_y *= -1
            search_pattern_state = 'horizontal'
            return perform_search(arduino)
       
        direction = 'R' if search_direction_y > 0 else 'L'
        send_motor_command(arduino, 2, direction, SEARCH_STEP_Y)
        current_y_angle = next_angle_y
        search_pattern_state = 'horizontal'
   
    return True

def send_motor_command(arduino, motor, direction, degrees):
    """Send command to Arduino"""
    if motor is None:
        return
   
    command = f"M{motor} {direction} {degrees}\n"
    arduino.write(command.encode())
   
    time_per_deg = 3
    delay = (degrees * time_per_deg) / 1000.0
    time.sleep(delay + 0.05)

def control_servo(arduino, turn_on):
    """Control servo based on fire state"""
    global servo_state
   
    if turn_on and not servo_state:
        arduino.write(b"ON\n")
        servo_state = True
        print("💧 SERVO ON - Suppressing fire!")
    elif not turn_on and servo_state:
        arduino.write(b"OFF\n")
        servo_state = False
        print("🛑 SERVO OFF")

def reset_system_position(arduino):
    """Reset turret to center position"""
    global current_x_angle, current_y_angle
   
    print("🔄 Resetting turret position...")
   
    # Reset X-axis
    if current_x_angle > 0:
        degrees = int(abs(current_x_angle))
        arduino.write(f"M1 L {degrees}\n".encode())
        time.sleep((degrees * 3) / 1000.0 + 0.1)
    elif current_x_angle < 0:
        degrees = int(abs(current_x_angle))
        arduino.write(f"M1 R {degrees}\n".encode())
        time.sleep((degrees * 3) / 1000.0 + 0.1)
   
    # Reset Y-axis
    if current_y_angle > 0:
        degrees = int(abs(current_y_angle))
        arduino.write(f"M2 L {degrees}\n".encode())
        time.sleep((degrees * 3) / 1000.0 + 0.1)
    elif current_y_angle < 0:
        degrees = int(abs(current_y_angle))
        arduino.write(f"M2 R {degrees}\n".encode())
        time.sleep((degrees * 3) / 1000.0 + 0.1)
   
    current_x_angle = 0.0
    current_y_angle = 0.0
    print("✅ Turret centered")

# =====================================================
# MAIN SYSTEM LOOP
# =====================================================
def main():
    global SEARCH_MODE, SYSTEM_STATE, NO_FIRE_COUNTER, IR_SEARCH_START_TIME
    global frame_center_x, frame_center_y
   
    print("\n" + "=" * 70)
    print("🔥 DUAL-CAMERA FIRE DETECTION & SUPPRESSION SYSTEM")
    print("=" * 70)
    print("\n📹 Camera System:")
    print("  • TAPO (RTSP) - ML Fire Detection & Monitoring")
    print("  • FLIR (IR)   - Thermal Tracking & Targeting")
    print("\n🔄 System Flow:")
    print("  1. MONITORING: Tapo camera watches for fire")
    print("  2. TRACKING:   IR camera searches and locks target")
    print("  3. SUPPRESSING: Servo activated when fire centered")
    print("  4. AUTO-LOOP:  Returns to monitoring after suppression")
    print("\n⚙️  Axis Limits:")
    print(f"  • X-Axis: +{X_AXIS_LIMIT_RIGHT}° (Right) / -{X_AXIS_LIMIT_LEFT}° (Left)")
    print(f"  • Y-Axis: ±{Y_AXIS_LIMIT}°")
    print("=" * 70)
   
    # Initialize ML model
    ml_model = initialize_ml_model()
    if ml_model is None:
        print("❌ Cannot start without ML model")
        return
   
    # Connect to Arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Arduino connected on {ARDUINO_PORT}")
    except Exception as e:
        print(f"❌ Arduino connection error: {e}")
        return
   
    # Open Tapo RTSP stream
    print(f"\n📡 Connecting to Tapo camera: {IP_ADDRESS}")
    rtsp_cap = cv2.VideoCapture(rtsp_url)
    rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
   
    if not rtsp_cap.isOpened():
        print("❌ Cannot open Tapo RTSP stream")
        return
    print("✅ Tapo camera connected")
   
    # FLIR camera will be opened only when needed
    flir_cap = None
   
    print("\n🟢 SYSTEM ACTIVE - Monitoring with Tapo camera...")
    print("⌨️  Press 'q' to quit | 's' for emergency stop\n")
   
    last_move_time = time.time()
    last_search_time = time.time()
    move_delay = 0.5
    search_delay = 1.0
   
    try:
        while True:
           
            # ============================================
            # STATE 1: MONITORING (Tapo Camera Only)
            # ============================================
            if SYSTEM_STATE == "MONITORING":
                fire_detected, confidence, tapo_frame = monitor_with_tapo(ml_model, rtsp_cap)
               
                if tapo_frame is not None:
                    cv2.imshow('Fire Detection System', tapo_frame)
               
                if fire_detected:
                    print(f"\n🚨 FIRE DETECTED by Tapo! Confidence: {confidence:.2f}")
                    print("🔄 Switching to IR camera for tracking...")
                   
                    # Close Tapo, open FLIR
                    rtsp_cap.release()
                   
                    flir_cap = cv2.VideoCapture(FLIR_CAMERA_INDEX, cv2.CAP_DSHOW)
                    flir_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    flir_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    flir_cap.set(cv2.CAP_PROP_FPS, 30)
                   
                    ret, test_frame = flir_cap.read()
                    if ret:
                        frame_center_y, frame_center_x = test_frame.shape[0] // 2, test_frame.shape[1] // 2
                   
                    SYSTEM_STATE = "TRACKING"
                    SEARCH_MODE = True
                    NO_FIRE_COUNTER = 0
                    IR_SEARCH_START_TIME = time.time()  # Record when IR search started
                    print("✅ IR camera active - Starting 30-second minimum search")
                    print("⏱️  IR will search for at least 30 seconds before returning to Tapo")
           
            # ============================================
            # STATE 2 & 3: TRACKING/SUPPRESSING (IR Camera)
            # ============================================
            elif SYSTEM_STATE in ["TRACKING", "SUPPRESSING"]:
                ret, flir_frame = flir_cap.read()
                if not ret:
                    print("❌ Failed to read FLIR frame")
                    break
               
                x, y, thermal_detected, debug_frame, stats = detect_flame_adaptive(flir_frame)
               
                if thermal_detected:
                    NO_FIRE_COUNTER = 0
                   
                    if SEARCH_MODE:
                        print("🎯 Thermal target acquired! Precision tracking engaged")
                        SEARCH_MODE = False
                        SYSTEM_STATE = "TRACKING"
                   
                    if (time.time() - last_move_time) > move_delay:
                        motor, direction, degrees, should_move, is_centered = calculate_movement(x, y)
                       
                        if is_centered:
                            SYSTEM_STATE = "SUPPRESSING"
                            control_servo(arduino, True)
                        else:
                            if should_move:
                                send_motor_command(arduino, motor, direction, degrees)
                                last_move_time = time.time()
                else:
                    # No thermal target
                    NO_FIRE_COUNTER += 1
                    control_servo(arduino, False)
                   
                    # Check if minimum IR search time has elapsed
                    ir_search_elapsed = time.time() - IR_SEARCH_START_TIME
                    time_remaining = max(0, MIN_IR_SEARCH_TIME - ir_search_elapsed)
                   
                    # Display countdown on frame
                    if time_remaining > 0:
                        cv2.putText(debug_frame, f"Min search time: {time_remaining:.1f}s remaining",
                                    (10, debug_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                   
                    # Only return to monitoring if:
                    # 1. Minimum search time has passed AND
                    # 2. No fire detected for threshold frames
                    if ir_search_elapsed >= MIN_IR_SEARCH_TIME and NO_FIRE_COUNTER > NO_FIRE_THRESHOLD:
                        print(f"\n✅ IR search complete ({ir_search_elapsed:.1f}s)")
                        print(f"✅ No fire detected for {NO_FIRE_COUNTER} frames")
                        print("🔄 Returning to monitoring mode with Tapo camera...")
                       
                        # Reset and switch back
                        control_servo(arduino, False)
                        reset_system_position(arduino)
                       
                        flir_cap.release()
                       
                        rtsp_cap = cv2.VideoCapture(rtsp_url)
                        rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                       
                        SYSTEM_STATE = "MONITORING"
                        SEARCH_MODE = False
                        NO_FIRE_COUNTER = 0
                        IR_SEARCH_START_TIME = None
                        print("🟢 Monitoring mode active\n")
                    elif ir_search_elapsed < MIN_IR_SEARCH_TIME:
                        # Still within minimum search time - continue searching
                        if SEARCH_MODE and (time.time() - last_search_time) > search_delay:
                            if perform_search(arduino):
                                last_search_time = time.time()
                    else:
                        # Minimum time passed but not enough no-fire frames yet
                        if SEARCH_MODE and (time.time() - last_search_time) > search_delay:
                            if perform_search(arduino):
                                last_search_time = time.time()
               
                cv2.imshow('Fire Detection System', debug_frame)
           
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Shutting down system...")
                break
            elif key == ord('s'):
                print("\n⚠️ EMERGENCY STOP")
                arduino.write(b"STOP\n")
                control_servo(arduino, False)
               
                if SYSTEM_STATE != "MONITORING":
                    reset_system_position(arduino)
                    if flir_cap is not None:
                        flir_cap.release()
                    rtsp_cap = cv2.VideoCapture(rtsp_url)
                    rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
               
                SYSTEM_STATE = "MONITORING"
                SEARCH_MODE = False
                print("🟢 Returned to monitoring mode")
   
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        arduino.write(b"STOP\n")
        control_servo(arduino, False)
        time.sleep(0.1)
       
        if flir_cap is not None:
            flir_cap.release()
        rtsp_cap.release()
        arduino.close()
        cv2.destroyAllWindows()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main()