#!/usr/bin/env python3

import cv2
import csv
import dlib
import numpy as np
import os
import rclpy
import time
from rclpy.node import Node
from flask import Flask, Response, request, send_from_directory, jsonify
from flask_cors import CORS
from rembg import remove
from threading import Thread
from std_msgs.msg import Bool

app = Flask(__name__) # Flask Web Server
CORS(app)  # Enable CORS for all routes

class ImageProcessor(Node):
    #---------- Initialiser ----------
    def __init__(self):
        super().__init__('image_processor')
        self.get_logger().info("Image Processor Node Started")

        # Output directory for saved images
        self.output_dir = os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/output")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.delete_old_files()  # Delete old images

        # Load Dlib face detector
        self.detector = dlib.get_frontal_face_detector()
        model_path = os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/models/shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(model_path)

        # Start video capture
        self.cap = cv2.VideoCapture(2) # 0 for default camera, 1/2 for external camera
        self.current_frame = None

        # Start Flask server in a separate thread
        self.server_thread = Thread(target=self.run_flask, daemon=True)
        self.server_thread.start()

        # Create a publisher
        self.publisher_ = self.create_publisher(Bool, 'image_processed', 10)
        self.publisherStarter_ = self.create_publisher(Bool, 'starter', 10)

    #---------- Destructor ----------
    def delete_old_files(self):
        """Deletes all images in the output directory."""
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    self.get_logger().info(f'Deleted old image: {file_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to delete {file_path}. Reason: {e}')

    #---------- Flask Server ----------
    def run_flask(self):
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    #---------- Flask Server Frames ----------
    def generate_frames(self):
        """ Continuously captures frames from the webcam for streaming. """
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            else:
                self.current_frame = frame  # Store the latest frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    #---------- Image Processing ----------
    def process_image(self):
        """ Processes the captured image (removes background & applies sketch effect). """
        if self.current_frame is None:
            return jsonify({"error": "No image captured yet!"}), 400
        
        # Scale down image to 960x720px
        image = self.current_frame.copy() # Original is 640x480px
        desired_width = 960
        aspect_ratio = image.shape[1] / image.shape[0]
        new_height = int(desired_width / aspect_ratio)
        dim = (desired_width, new_height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Save webcam image
        webcam_image_path = os.path.join(self.output_dir, "0_webcam.jpg")
        cv2.imwrite(webcam_image_path, image)
        self.get_logger().info(f'Webcam Image saved to: {webcam_image_path}')

        # Remove background using rembg
        no_bg_face = remove(image)
        no_bg_face_bgr = cv2.cvtColor(no_bg_face, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(no_bg_face_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray_image)
        if len(faces) == 0:
            self.get_logger().error('No face detected!')
            return jsonify({"error": "No face detected! Retake Image"}), 400

        for face in faces:
            landmarks = self.predictor(gray_image, face)

        # Convert to grayscale for sketch effect
        face_gray = cv2.cvtColor(no_bg_face_bgr, cv2.COLOR_BGR2GRAY)
        output_path_gray = os.path.join(self.output_dir, '1_no_background.jpg')
        cv2.imwrite(output_path_gray, face_gray)
        self.get_logger().info(f'Face with background removed saved to: {output_path_gray}')

        # Apply Blur
        blurred_face = cv2.bilateralFilter(gray_image, 9, 75, 75)

        # Detect edges using Canny
        edges = cv2.Canny(blurred_face, 50, 150)

        # Remove small contours (noise)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 10  # Change this value based on small noise removal needs

        mask = np.zeros_like(edges)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_contour_area:
                cv2.drawContours(mask, [cnt], -1, 255, 1)  # Set thickness to 1px

        # Convert edges to 3 channels
        edges_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Draw landmark lines
        for face in faces:
            landmarks = self.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), face)

            def draw_landmark_lines(image, landmarks, points, color=(255, 255, 255), thickness=1):
                for i in range(len(points) - 1):
                    x1, y1 = landmarks.part(points[i]).x, landmarks.part(points[i]).y
                    x2, y2 = landmarks.part(points[i + 1]).x, landmarks.part(points[i + 1]).y
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

            facial_features = {
                "jaw": list(range(0, 17)),
                "eyebrows": list(range(17, 27)),
                "nose": list(range(27, 36)),
                "eyes": [list(range(36, 42)), list(range(42, 48))],
                "mouth": [list(range(48, 60)), list(range(60, 68))]
            }

            for feature in ["jaw", "nose"]:
                draw_landmark_lines(edges_colored, landmarks, facial_features[feature])

            draw_landmark_lines(edges_colored, landmarks, list(range(17, 22)))
            draw_landmark_lines(edges_colored, landmarks, list(range(22, 27)))

            for feature in ["eyes", "mouth"]:
                for part in facial_features[feature]:
                    draw_landmark_lines(edges_colored, landmarks, part)

        # Save final sketch
        final_sketch = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2GRAY)
        sketch_image_path = os.path.join(self.output_dir, "2_sketch.jpg")
        cv2.imwrite(sketch_image_path, final_sketch)
        self.get_logger().info(f'Sketch face saved to: {sketch_image_path}')

        # Publish message
        time.sleep(1)
        msg = Bool()
        msg.data = True
        self.publisher_.publish(msg)

        return "Processing complete!", 200

# Flask Routes
@app.route('/video_feed')
def video_feed():
    return Response(image_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_image():
    return image_processor.process_image()

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(image_processor.output_dir, filename)

@app.route('/draw', methods=['POST'])
def start_drawing():
    """Publishes to the publisherStarter_ topic when the start drawing button is pressed."""
    msg = Bool()
    msg.data = True
    image_processor.publisherStarter_.publish(msg)
    return jsonify({"message": "Drawing started!"}), 200

def main(args=None):
    rclpy.init(args=args)
    global image_processor
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
