from ultralytics import YOLO
import cv2

# Change the path to where you save the pretrained weight
model = YOLO('yolov8n-pose.pt')

# change the video path to where you save your video
video_path = "/path/to/your/directory"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
output_path = "video/output_level1.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, verbose = False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes = False)

        # Display the annotated frame
        cv2.imshow(annotated_frame) # run this if you use your personal device

        # save the video
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
