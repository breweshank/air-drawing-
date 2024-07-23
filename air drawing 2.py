import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a black image for drawing
drawing_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables for drawing
color = (255, 0, 0)  # Initial color (red)
radius = 5
is_erasing = False

# Initialize hand tracking
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the tip of the index finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Get the base of the palm
                palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Convert the normalized coordinates to image coordinates
                h, w, _ = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                px, py = int(palm_base.x * w), int(palm_base.y * h)

                # Check for color selection gestures
                if cy < 50:
                    if cx < 50:
                        color = (255, 0, 0)  # Red
                    elif 50 <= cx < 100:
                        color = (0, 255, 0)  # Green
                    elif 100 <= cx < 150:
                        color = (0, 0, 255)  # Blue
                    elif 150 <= cx < 200:
                        color = (0, 255, 255)  # Yellow
                    elif 200 <= cx < 250:
                        color = (255, 0, 255)  # Magenta

                # Check if the palm is close to the index finger (erasing)
                if abs(cx - px) < 50 and abs(cy - py) < 50:
                    is_erasing = True
                else:
                    is_erasing = False

                # Draw or erase on the drawing image
                if is_erasing:
                    cv2.circle(drawing_image, (cx, cy), 20, (0, 0, 0), -1)
                else:
                    cv2.circle(drawing_image, (cx, cy), radius, color, -1)

        # Create color selection boxes on the top of the frame
        cv2.rectangle(frame, (0, 0), (50, 50), (255, 0, 0), -1)
        cv2.rectangle(frame, (50, 0), (100, 50), (0, 255, 0), -1)
        cv2.rectangle(frame, (100, 0), (150, 50), (0, 0, 255), -1)
        cv2.rectangle(frame, (150, 0), (200, 50), (0, 255, 255), -1)
        cv2.rectangle(frame, (200, 0), (250, 50), (255, 0, 255), -1)
        
        # Overlay the drawing on the frame
        combined_frame = cv2.addWeighted(frame, 0.5, drawing_image, 0.5, 0)

        # Show the frame
        cv2.imshow("Air Drawing", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
