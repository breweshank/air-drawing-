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
                
                # Convert the normalized coordinates to image coordinates
                h, w, _ = frame.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Draw on the drawing image
                cv2.circle(drawing_image, (cx, cy), 7, (255, 0, 0), -1)

        # Overlay the drawing on the frame
        combined_frame = cv2.addWeighted(frame, 0.5, drawing_image, 0.5, 0)

        # Show the frame
        cv2.imshow("Air Mouse Drawing", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
