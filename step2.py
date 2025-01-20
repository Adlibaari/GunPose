import cv2
from ultralytics import YOLO
import pandas as pd
import os

# Ensure directories exist
os.makedirs('/level2/image', exist_ok=True)
os.makedirs('/level2/gun', exist_ok=True)

model = YOLO('yolov8n-pose.pt')
video_path = "standgun.mp4"
cap = cv2.VideoCapture(video_path)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
seconds = round(frames / fps)

frame_total = 500
i, a = 0, 0
all_data = []

while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds / frame_total) * 1000)))
    flag, frame = cap.read()

    if not flag or frame is None:
        print(f"Failed to read frame {i}.")
        break

    image_path = f'C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level2/image/img_{i}.jpg'
    cv2.imwrite(image_path, frame)

    results = model(frame, verbose=False)

    for r in results:
        bound_box = r.boxes.xyxy
        conf = r.boxes.conf.tolist()
        keypoints = r.keypoints.xyn.tolist()

        for index, box in enumerate(bound_box):
            if conf[index] > 0.75:
                x1, y1, x2, y2 = box.tolist()
                height, width, _ = frame.shape
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(width, int(x2)), min(height, int(y2))

                pict = frame[y1:y2, x1:x2]

                if pict.size == 0:
                    print(f"Skipping empty image for person_{a}.jpg")
                    continue

                output_path = f'C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level2/gun/person_{a}.jpg' 
                cv2.imwrite(output_path, pict)

                data = {'image_name': f'person_{a}.jpg'}
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]

                all_data.append(data)
                a += 1

    i += 1

print(f"Processed {i-1} frames and detected {a-1} persons.")
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(all_data)
csv_file_path = 'level2/keypoints.csv'
df.to_csv(csv_file_path, index=False)
