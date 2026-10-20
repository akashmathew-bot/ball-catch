import cv2
import mediapipe as mp
import random
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

# Game state
ball_x = random.randint(50, w - 50)
ball_y = 0
ball_speed = 6
ball_radius = 20
score = 0
palm_radius = 50

start_time = time.time()
speed_increase_interval = 5  # seconds
speed_increment = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    palm_centers = []
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            palm_points = [0, 5, 9, 13, 17]
            palm_x = int(sum(landmarks[i].x for i in palm_points) / len(palm_points) * w)
            palm_y = int(sum(landmarks[i].y for i in palm_points) / len(palm_points) * h)
            palm_centers.append((palm_x, palm_y))

    # Increase speed over time
    elapsed = time.time() - start_time
    current_speed = ball_speed + int(elapsed // speed_increase_interval) * speed_increment

    # Update ball position
    ball_y += current_speed

    # Draw ball
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

    # Draw palm markers and check catch
    for (palm_x, palm_y) in palm_centers:
        cv2.circle(frame, (palm_x, palm_y), palm_radius, (0, 255, 0), 3)

        distance = ((ball_x - palm_x) ** 2 + (ball_y - palm_y) ** 2) ** 0.5
        if distance < (palm_radius + ball_radius):
            score += 1
            ball_x = random.randint(50, w - 50)
            ball_y = 0

    # Reset ball if missed
    if ball_y > h:
        ball_x = random.randint(50, w - 50)
        ball_y = 0

    # Display score and speed
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Speed: {current_speed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Catch Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()