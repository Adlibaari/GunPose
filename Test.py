from ultralytics import YOLO
import cv2
import xgboost as xgb
import pandas as pd
import time

# Load models on GPU if available
model_yolo_pose = YOLO('yolov8n-pose.pt').to('cpu')
model_xgb = xgb.Booster()
model_xgb.load_model('C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level3/model_weights.xgb') # Assume this model has been optimized/quantized
model_yolo_gun = YOLO("best.pt").to('cpu')

# Load video
video_path = "videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)  # Downscale width by 50%
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)  # Downscale height by 50%

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
output_path = "C:/Users/Barry/Documents/Uni/Projects/Object Tracking/Pose Estimation/Gunestimation/XGboost/level4/cpu.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))  # Scaled back up for writing

frame_tot = 0
fps_list = []

# Process every nth frame
n = 1

while cap.isOpened():
    start_time = time.time()  # Start time for processing the frame
    
    success, frame = cap.read()
    if not success:
        break

    # Resize frame for faster processing
    # frame = cv2.resize(frame, (width, height))
    annotated_frame = frame.copy()

    # Run detection every nth frame
    if frame_tot % n == 0:
        # YOLO Pose Detection
        pose_results = model_yolo_pose(frame, verbose=False)
        annotated_frame = pose_results[0].plot(boxes=False)

        gun_detected = False
        for r in pose_results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:
                    x1, y1, x2, y2 = box.tolist()
                    data = {f'x{j}': keypoints[index][j][0] for j in range(len(keypoints[index]))}
                    data.update({f'y{j}': keypoints[index][j][1] for j in range(len(keypoints[index]))})

                    df = pd.DataFrame(data, index=[0])
                    column_order = [f'{axis}{i}' for i in range(17) for axis in ['x', 'y']]
                    df = df[column_order]

                    dmatrix = xgb.DMatrix(df)
                    cut = model_xgb.predict(dmatrix)
                    print("cut ",cut)
                    binary_predictions = (cut > 0.5).astype(int)
                    print("bp ",binary_predictions)

                    if binary_predictions == 0:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, 'gun pose', (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 3)

                        # Custom YOLO model for gun detection
                        # gun_results = model_yolo_gun(frame, verbose=False)
                        # for gun in gun_results:
                        #     gun_boxes = gun.boxes.xyxy
                        #     gun_conf = gun.boxes.conf.tolist()
                        #     if any(g > 0.3 for g in gun_conf):
                        #         gun_detected = True
                        #         break

        # if gun_detected:
        #     cv2.putText(annotated_frame, 'Gun Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    # Calculate and display processing FPS
    end_time = time.time()
    processing_fps = 1 / (end_time - start_time)
    fps_list.append(processing_fps)
    cv2.putText(annotated_frame, f'FPS: {processing_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize annotated frame back to original dimensions
    annotated_frame = cv2.resize(annotated_frame, (width * 2, height * 2))
    out.write(annotated_frame)
    frame_tot += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate and print average processing FPS
average_fps = sum(fps_list) / len(fps_list)
print(f'Average Processing FPS: {average_fps:.2f}')
