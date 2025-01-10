import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
import rclpy
from rclpy.node import Node

class BlackRoadDetectionNode(Node):
    def __init__(self):
        super().__init__('black_road_detection_node')
        self.publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)

        # Timer for periodic publishing
        self.timer = self.create_timer(0.1, self.detect_and_publish_road)

        # Video capture
        self.cap = cv2.VideoCapture("/home/tanay/Downloads/road_vid.mp4")  # Use 0 for webcam or replace with a video file path

    def detect_and_publish_road(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Failed to grab frame.")
            return

        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Threshold to detect black regions
        # Black pixels will have lower intensity values, so use a low threshold
        _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        # Step 3: Apply morphological operations to refine the road
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Horizontal line detection
        refined_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)

        # Use another morphological operation to fill gaps
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        refined_binary = cv2.morphologyEx(refined_binary, cv2.MORPH_CLOSE, kernel_fill)

        # Step 4: Convert the binary road mask to occupancy grid format
        occupancy_grid = (refined_binary == 0).astype(np.int8) * 100  # Black pixels (road) = 100, white = 0

        # Step 5: Publish as OccupancyGrid
        self.publish_occupancy_grid(occupancy_grid, refined_binary.shape)

        # Step 6: Debugging - display images
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Refined Binary Road", refined_binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def publish_occupancy_grid(self, grid_data, shape):
        # Create OccupancyGrid message
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        # Set metadata
        meta = MapMetaData()
        meta.resolution = 0.01  # Meters per cell
        meta.width = shape[1]  # Number of columns
        meta.height = shape[0]  # Number of rows
        meta.origin.position.x = 0.0
        meta.origin.position.y = 0.0
        meta.origin.position.z = 0.0
        grid_msg.info = meta

        # Flatten and assign grid data
        grid_msg.data = grid_data.flatten().tolist()
        self.publisher.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BlackRoadDetectionNode()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
