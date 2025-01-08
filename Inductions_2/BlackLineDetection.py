import cv2
import numpy as np

def detect_black_road(video_path, delay):
    # Open the video file or webcam
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold to detect black regions
        _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        

        # Morphological operations to refine detection
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        refined_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)

        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        refined_binary = cv2.morphologyEx(refined_binary, cv2.MORPH_CLOSE, kernel_fill)

        # Display the results
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Detected Black Road", refined_binary)

        # Exit on 'q' key press, slow playback by introducing a delay
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Replace with your video file path or use 0 for webcam
delay=50
video_path = "C:/Users/Tanay/Downloads/LINE_FOLLOWER_PS(BEGINNER).mp4"
detect_black_road(video_path, delay)  # Adjust delay (in milliseconds) to control speed