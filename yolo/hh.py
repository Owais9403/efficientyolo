from ultralytics import YOLO
import cv2
import os

# Set environment variable to allow multiple OpenMP runtimes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

# Read frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break
    
    # Detect and track objects
    results = model.track(frame, persist=True)
    
    # Check if 'person' class detected with confidence > 0.7
    if 0 in results[0].boxes.cls and results[0].boxes.conf[0] > 0.7:
        frame = results[0].plot()  # Plot results
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()