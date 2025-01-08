
# Black Road Detection and Occupancy Grid Publisher

This ROS 2 node detects black road regions in a video feed, processes the detection into a ROS 2 OccupancyGrid message, and publishes it to the /occupancy_grid topic. The grid is used for path planning or navigation in autonomous robotic systems.

## Overview
This code performs the following steps:

1. Captures frames from a video feed or webcam.
2. Processes each frame to detect black regions (representing roads).
3. Converts the detection into an OccupancyGrid message, which is a grid-based map commonly used in ROS 2 navigation stacks.
4. Publishes the grid for use by other ROS 2 nodes.
5. Visualizes the results using OpenCV for debugging and refinement.

## Purpose of Each steps
### 1.Video Input(Frame Acquisition)
The program captures frames from a video file or webcam using OpenCV’s cv2.VideoCapture.

* Why: Provides the input video for processing.
* How to Set It Up:
     * For a video file, specify its path:
     ```python
     self.cap = cv2.VideoCapture("/path/to/road_vid.mp4")

     ```
     * For a live webcam feed, use 0:
     ```python
     self.cap = cv2.VideoCapture(0)
     ```
### 2.Grayscale Conversion
Each frame is converted to grayscale for easier processing. Grayscale reduces computation by focusing on intensity values rather than color channels:
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

```
We prefer grayscale image over a colour image because it has only one color channel(reduced data size) which increases the computation speed.Also it requires less memeory storage(occupies a third of the space required by RGB images).
### 3.Thresholding for Black Regions
The grayscale image is thresholded to detect black pixels (low intensity):
```python
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
```
* Why: Black pixels represent the road, which we aim to detect.
* How It Works:
     * Pixels with intensity values below 50 are set to 255 (white in the binary mask).
     * All other pixels are set to 255 (white in the binary mask).
     * The result is a binary image where the detected road appears as black regions.

### 4.Refinement Using Morphological Operations
To improve the quality of detection, we perform morphological operations. These operations help refine the binary mask by removing noise, filling gaps, and improving the continuity of detected road areas:
1. #### Close Gaps Between Lines:
```python
kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 1))
refined_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)
```
* Why: Horizontal gaps (e.g., dashed road markings) are closed to form continuous road segments.
* Kernel Shape: A rectangle (90, 1) detects and connects horizontal lines.
2. #### Fill Gaps Within Road Regions:
```python
kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
refined_binary = cv2.morphologyEx(refined_binary, cv2.MORPH_CLOSE, kernel_fill)

```
* Why: Smoothens the detected road and fills small gaps in road regions.
* Kernel Shape: An ellipse (10, 10) approximates circular gaps.

### 5.Convert Binary Mask to Occupancy Grid
The binary road mask is converted into a 2D occupancy grid:
```python
occupancy_grid = (refined_binary == 0).astype(np.int8) * 100
```

* How It Works:
     * Black pixels in the binary mask (road) are set to 100 (free space).
     * White pixels (non-road regions) are set to 0 (occupied space).
* Why: This format is compatible with ROS 2 navigation.

### 6.Publish as OccupancyGrid Message
The occupancy grid is wrapped into a ROS 2 OccupancyGrid message:
```python
grid_msg = OccupancyGrid()
grid_msg.header.stamp = self.get_clock().now().to_msg()
grid_msg.header.frame_id = 'map'
```
* Meta Information:
     * Resolution: Specifies the size of each cell in meters:
```python
meta.resolution = 0.01  # 1 cm per cell
```

     * Width and Height: Match the binary mask dimensions.
     * Origin: The bottom-left corner of the grid is at (0, 0).
The OccupancyGrid is then published to the /occupancy_grid topic.

### 7. Debugging and Visualization
Intermediate results are visualized using OpenCV:
```python
cv2.imshow("Original Frame", frame)
cv2.imshow("Refined Binary Road", refined_binary)
```
* Why: Allows real-time monitoring of the detection process for tuning thresholds or kernel sizes.
* Exiting Visualization:
     * Press q to quit and shut down the node.

## HOW TO RUN THE CODE
1. ### Install ROS 2 and Python Dependencies:
Ensure the following libraries are installed

2. ### Set Up the ROS 2 Package:
* Place the script in a ROS 2 package.
* Build and source the workspace:
```python
cd ~/ros2_ws
colcon build
source install/setup.bash
```
3. ### Run the node
```python
ros2 run <package_name> black_road_detection_node
```







