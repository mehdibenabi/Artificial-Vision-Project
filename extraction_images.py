import cv2
import os

video_path = "./video.mov"
output_dir = "./images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:  # Prendre une image tous les 5 frames
        filename = f"{output_dir}/{frame_count:04d}.png"
        cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
print("Extraction terminée.")
