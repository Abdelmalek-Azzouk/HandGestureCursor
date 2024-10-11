import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Unable to read video frame.")
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Process the image to find hands
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks for index and middle fingers
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Get the y-coordinates of the finger tips
            index_y = index_finger_tip.y
            middle_y = middle_finger_tip.y
            
            # Get the finger base (for detecting raised fingers)
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

            # Determine if fingers are raised (higher y-coordinates indicate raised fingers)
            index_raised = index_y < index_base  # Index finger is raised
            middle_raised = middle_y < middle_base  # Middle finger is raised

            # Move cursor if index finger is raised
            if index_raised:
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(x, y)

                # Check if the middle finger is also raised for a left click
                if middle_raised:
                    pyautogui.click()  # Perform a left click
                    print("Left Click")

            # Draw landmarks for visualization (optional)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Finger Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
