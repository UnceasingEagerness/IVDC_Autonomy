
# Black Road Detection Using OpenCV  

This project demonstrates how to detect black road regions in a video using Python and OpenCV. It processes video frames to identify and highlight black road areas using image processing techniques such as grayscale conversion, thresholding, and morphological operations.

---

## **Approach**  

### **1. Video Input**  
The program can process either a video file or a webcam feed. The video is read frame by frame, ensuring real-time or near-real-time detection.  
```python
cap = cv2.VideoCapture(video_path)
```
Here you can add your video file path or put Zero if you want to use WEBCAM

## ***Preprocessing***
### **2. Grayscale Conversion**  
Each frame is converted to grayscale for easier processing. Grayscale reduces computation by focusing on intensity values rather than color channels.

```python  
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
We prefer grayscale image over a colour image because it has only one color channel(reduced data size) which increases the computation speed.Also it requires less memeory storage(occupies a third of the space required by RGB images).

### **3.Black Region Detection(Thresholding)**
Thresholding is applied to detect dark regions in the grayscale image, isolating black road sections:
```python  
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
```
* 50: The intensity threshold. Pixels with intensity below this value are considered part of the black region.
* cv2.THRESH_BINARY_INV: Inverts the binary output, making dark areas white in the binary mask.

### **4.Morphological Refinement**
Morphological operations are applied to clean up noise and refine the detected black road region.

***Line Refinement:***
A rectangular kernel is used for detecting continuous lines and filling gaps:

```python  
kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 1))
refined_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)

```

***Hole Filling:***
An elliptical kernel is applied to fill small holes in the detected road region:

```python  
kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
refined_binary = cv2.morphologyEx(refined_binary, cv2.MORPH_CLOSE, kernel_fill)

```

### **5.Display Results**
The original frame and the refined binary mask showing the detected black road are displayed side by side.

```python
cv2.imshow("Original Frame", frame)
cv2.imshow("Detected Black Road", refined_binary)
```

### **6. Playback Control**
The program includes a playback delay to control the speed of frame rendering. Press the q key to exit.


## ****HOW TO USE****
### **Prerequisites**
* Python 3.x
* OpenCV library (pip install opencv-python)

### **Steps to run**
1. Replace video_path with the path to your video file or set it to 0 to use the webcam.
```python
video_path = "path_to_your_video.mp4"  

```

2. Set the desired playback delay in milliseconds:
```python
delay = 50  # Adjust for faster or slower playback  
  

```
3. Run the script.

4. To stop the program, press the q key

## ****Concepts Explored****
1. Grayscale Conversion: Simplifies image processing by focusing on intensity values.
2. Thresholding: Identifies regions of interest based on intensity thresholds.
3. Morphological Operations:
* Closing: Fills small gaps and connects broken lines.
* Kernel Shapes:
     * Rectangular: Enhances linear structures.
     * Elliptical: Smooths and fills irregular shapes.
4. Video Processing: Real-time frame-by-frame processing for dynamic inputs.

## ****Example Output****
When applied to a video, the program highlights black road areas as white regions in the binary mask.










