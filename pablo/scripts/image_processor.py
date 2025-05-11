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
        self.cap = cv2.VideoCapture(0) # 0 for default camera, 1/2 for external camera
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

    #---------- Image Capture ----------
    def capture_image(self):
        """ Captures an image from the webcam and saves it to the output directory. """
        self.delete_old_files() # Delete old images
        success, frame = self.cap.read() # Capture a frame from the webcam
        if not success:
            return jsonify({"error": "Failed to capture image!"}), 500
        else:
            # Save the captured image
            image_path = os.path.join(self.output_dir, "0_webcam.jpg")
            cv2.imwrite(image_path, frame)
            self.resize_image()
            self.get_logger().info(f'Captured Image saved to: {image_path}')
            return jsonify({"message": "Image captured successfully!"}), 200

    def resize_image(self):
        image_path = os.path.join(self.output_dir, "0_webcam.jpg")
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error('No image captured yet!')
            return jsonify({"error": "No image captured yet!"}), 400
        
        # Resize the image
        desired_width = 960
        aspect_ratio = 4 / 3
        new_height = int(desired_width / aspect_ratio)
        dim = (desired_width, new_height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, image)
        self.get_logger().info(f'Captured Image saved to: {image_path}')
        return jsonify({"message": "Image captured successfully!"}), 200

    #---------- Image Processing ----------
    def process_image(self):
        # Read the captured image 
        image = cv2.imread(os.path.join(self.output_dir, "0_webcam.jpg"))
        if image is None:
            self.get_logger().error('No image captured yet!')
            return jsonify({"error": "No image captured yet!"}), 400

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

        # Add a border to the sketch
        border_thickness = 1  # Thickness of the border in pixels
        border_color = (255, 255, 255)  # White border
        final_sketch_with_border = cv2.copyMakeBorder(
            final_sketch,
            top=border_thickness,
            bottom=border_thickness,
            left=border_thickness,
            right=border_thickness,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )

        # Save the final sketch
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
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/video_feed')
def video_feed():
    return Response(image_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_frame():
    return image_processor.capture_image()

@app.route('/process', methods=['POST'])
def process_frame():
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

@app.route('/upload', methods=['POST'])
def upload_image():
    image_processor.delete_old_files()
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save the file to the output folder
    new_filename = "0_webcam.jpg"
    file_path = os.path.join(image_processor.output_dir, new_filename)
    file.save(file_path)
    image_processor.resize_image()
    return f"File uploaded successfully: {new_filename}", 200

def main(args=None):
    rclpy.init(args=args)
    global image_processor
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




# #!/usr/bin/env python3

# import cv2
# import csv
# import dlib
# import numpy as np
# import os
# import rclpy
# import time
# from rclpy.node import Node
# from flask import Flask, Response, request, send_from_directory, jsonify
# from flask_cors import CORS
# from rembg import remove
# from threading import Thread
# from std_msgs.msg import Bool

# app = Flask(__name__) # Flask Web Server
# CORS(app)  # Enable CORS for all routes

# class ImageProcessor(Node):
#     #---------- Initialiser ----------
#     def __init__(self):
#         super().__init__('image_processor')
#         self.get_logger().info("Image Processor Node Started")

#         # Output directory for saved images
#         self.output_dir = os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/output")
        
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.delete_old_files()  # Delete old images

#         # Load Dlib face detector
#         self.detector = dlib.get_frontal_face_detector()
#         model_path = os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/models/shape_predictor_68_face_landmarks.dat")
#         self.predictor = dlib.shape_predictor(model_path)

#         # Start video capture
#         self.cap = cv2.VideoCapture(0) # 0 for default camera, 1/2 for external camera
#         self.current_frame = None

#         # Start Flask server in a separate thread
#         self.server_thread = Thread(target=self.run_flask, daemon=True)
#         self.server_thread.start()

#         # Create a publisher
#         self.publisher_ = self.create_publisher(Bool, 'image_processed', 10)
#         self.publisherStarter_ = self.create_publisher(Bool, 'starter', 10)

#     #---------- Destructor ----------
#     def delete_old_files(self):
#         """Deletes all images in the output directory."""
#         for filename in os.listdir(self.output_dir):
#             file_path = os.path.join(self.output_dir, filename)
#             try:
#                 if os.path.isfile(file_path):
#                     os.unlink(file_path)
#                     self.get_logger().info(f'Deleted old image: {file_path}')
#             except Exception as e:
#                 self.get_logger().error(f'Failed to delete {file_path}. Reason: {e}')

#     #---------- Flask Server ----------
#     def run_flask(self):
#         app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

#     #---------- Flask Server Frames ----------
#     def generate_frames(self):
#         """ Continuously captures frames from the webcam for streaming. """
#         while True:
#             success, frame = self.cap.read()
#             if not success:
#                 break
#             else:
#                 self.current_frame = frame  # Store the latest frame
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     #---------- Image Capture ----------
#     def capture_image(self):
#         """ Captures an image from the webcam and saves it to the output directory. """
#         self.delete_old_files() # Delete old images
#         success, frame = self.cap.read() # Capture a frame from the webcam
#         if not success:
#             return jsonify({"error": "Failed to capture image!"}), 500
#         else:
#             # Save the captured image
#             image_path = os.path.join(self.output_dir, "0_webcam.jpg")
#             cv2.imwrite(image_path, frame)
#             self.resize_image()
#             self.get_logger().info(f'Captured Image saved to: {image_path}')
#             return jsonify({"message": "Image captured successfully!"}), 200

#     def resize_image(self):
#         image_path = os.path.join(self.output_dir, "0_webcam.jpg")
#         image = cv2.imread(image_path)
#         if image is None:
#             self.get_logger().error('No image captured yet!')
#             return jsonify({"error": "No image captured yet!"}), 400
        
#         # Resize the image
#         desired_width = 960
#         aspect_ratio = 4 / 3
#         new_height = int(desired_width / aspect_ratio)
#         dim = (desired_width, new_height)
#         image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#         cv2.imwrite(image_path, image)
#         self.get_logger().info(f'Captured Image saved to: {image_path}')
#         return jsonify({"message": "Image captured successfully!"}), 200

#     #---------- Image Processing ----------
#     def process_image(self):
#         # Read the captured image 
#         image = cv2.imread(os.path.join(self.output_dir, "0_webcam.jpg"))
#         if image is None:
#             self.get_logger().error('No image captured yet!')
#             return jsonify({"error": "No image captured yet!"}), 400

#         # Save original for reference
#         original_image_path = os.path.join(self.output_dir, "0_original.jpg")
#         cv2.imwrite(original_image_path, image)

#         # Remove background using rembg
#         no_bg_face = remove(image)
#         no_bg_face_bgr = cv2.cvtColor(no_bg_face, cv2.COLOR_RGB2BGR)
#         gray_image = cv2.cvtColor(no_bg_face_bgr, cv2.COLOR_BGR2GRAY)

#         # Detect faces
#         faces = self.detector(gray_image)
#         if len(faces) == 0:
#             self.get_logger().error('No face detected!')
#             return jsonify({"error": "No face detected! Retake Image"}), 400

#         # Save image with background removed
#         output_path_no_bg = os.path.join(self.output_dir, '1_no_background.jpg')
#         cv2.imwrite(output_path_no_bg, no_bg_face_bgr)
#         self.get_logger().info(f'Face with background removed saved to: {output_path_no_bg}')

#         # Create an alpha mask from the RGBA image
#         # This will help us isolate hair regions better
#         alpha_mask = no_bg_face[:, :, 3]
        
#         # Extract face region to identify hair region
#         face_landmarks = None
#         for face in faces:
#             landmarks = self.predictor(gray_image, face)
#             face_landmarks = landmarks
        
#         # Create a sketch with enhanced hair detail
#         sketch = self.create_enhanced_sketch(no_bg_face_bgr, gray_image, face_landmarks, alpha_mask)
        
#         # Save the final sketch
#         sketch_image_path = os.path.join(self.output_dir, "2_sketch.jpg")
#         cv2.imwrite(sketch_image_path, sketch)
#         self.get_logger().info(f'Enhanced sketch with hair details saved to: {sketch_image_path}')

#         # Publish message
#         time.sleep(1)
#         msg = Bool()
#         msg.data = True
#         self.publisher_.publish(msg)

#         return "Processing complete!", 200
    
#     def create_enhanced_sketch(self, image, gray_image, landmarks, alpha_mask):
#         """Creates an enhanced sketch with better hair detail capture"""
#         height, width = gray_image.shape[:2]
        
#         # Create a hair mask using intensity thresholding and alpha channel
#         _, hair_mask = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
        
#         # Refine hair mask using alpha channel information
#         hair_mask = cv2.bitwise_and(hair_mask, alpha_mask)
        
#         # Create a face region mask from landmarks
#         face_mask = np.zeros_like(gray_image)
#         if landmarks:
#             # Get face boundary points
#             points = []
#             for i in range(17):  # Jaw line
#                 x, y = landmarks.part(i).x, landmarks.part(i).y
#                 points.append([x, y])
                
#             # Add forehead points (estimated from eyebrows)
#             left_eyebrow_x = landmarks.part(19).x
#             left_eyebrow_y = landmarks.part(19).y
#             right_eyebrow_x = landmarks.part(24).x
#             right_eyebrow_y = landmarks.part(24).y
            
#             # Estimate forehead points (higher than eyebrows)
#             forehead_height = int((landmarks.part(8).y - landmarks.part(27).y) * 0.7)  # Proportional to face height
#             points.append([right_eyebrow_x, right_eyebrow_y - forehead_height])
#             points.append([left_eyebrow_x, left_eyebrow_y - forehead_height])
            
#             # Convert to numpy array
#             points = np.array(points, dtype=np.int32)
#             cv2.fillPoly(face_mask, [points], 255)
            
#             # Dilate the face mask to better separate face from hair
#             kernel = np.ones((5, 5), np.uint8)
#             face_mask = cv2.dilate(face_mask, kernel, iterations=2)
        
#         # Create a hair region mask (outside face region but inside alpha)
#         hair_region = cv2.bitwise_and(alpha_mask, cv2.bitwise_not(face_mask))
        
#         # Enhance hair region
#         # 1. Apply Bilateral Filter with parameters tuned for hair
#         hair_filtered = cv2.bilateralFilter(gray_image, 9, 25, 25)
        
#         # 2. Apply adaptive thresholding to get more hair detail
#         adaptive_thresh = cv2.adaptiveThreshold(
#             hair_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#         )
        
#         # 3. Apply Canny edge detection with parameters tuned for hair edges
#         hair_edges = cv2.Canny(hair_filtered, 30, 100)
        
#         # Combine hair edges with the hair region mask
#         hair_detail = cv2.bitwise_and(hair_edges, hair_region)
        
#         # Create face edges using standard edge detection 
#         face_blurred = cv2.bilateralFilter(gray_image, 9, 75, 75)
#         face_edges = cv2.Canny(face_blurred, 50, 150)
#         face_detail = cv2.bitwise_and(face_edges, face_mask)
        
#         # Remove small contours (noise) from both hair and face edges
#         def clean_edges(edges, min_area=10):
#             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             mask = np.zeros_like(edges)
#             for cnt in contours:
#                 if cv2.contourArea(cnt) > min_area:
#                     cv2.drawContours(mask, [cnt], -1, 255, 1)
#             return mask
        
#         clean_hair_detail = clean_edges(hair_detail, min_area=5)  # Lower threshold for hair
#         clean_face_detail = clean_edges(face_detail, min_area=10)
        
#         # Combine face and hair details
#         combined_sketch = cv2.bitwise_or(clean_face_detail, clean_hair_detail)
        
#         # Convert to BGR for drawing landmarks
#         sketch_colored = cv2.cvtColor(combined_sketch, cv2.COLOR_GRAY2BGR)
        
#         # Draw facial feature lines using landmarks
#         if landmarks:
#             def draw_landmark_lines(image, landmarks, points, color=(255, 255, 255), thickness=1):
#                 for i in range(len(points) - 1):
#                     x1, y1 = landmarks.part(points[i]).x, landmarks.part(points[i]).y
#                     x2, y2 = landmarks.part(points[i + 1]).x, landmarks.part(points[i + 1]).y
#                     cv2.line(image, (x1, y1), (x2, y2), color, thickness)

#             facial_features = {
#                 "jaw": list(range(0, 17)),
#                 "eyebrows": list(range(17, 27)),
#                 "nose": list(range(27, 36)),
#                 "eyes": [list(range(36, 42)), list(range(42, 48))],
#                 "mouth": [list(range(48, 60)), list(range(60, 68))]
#             }

#             for feature in ["jaw", "nose"]:
#                 draw_landmark_lines(sketch_colored, landmarks, facial_features[feature])

#             draw_landmark_lines(sketch_colored, landmarks, list(range(17, 22)))
#             draw_landmark_lines(sketch_colored, landmarks, list(range(22, 27)))

#             for feature in ["eyes", "mouth"]:
#                 for part in facial_features[feature]:
#                     draw_landmark_lines(sketch_colored, landmarks, part)
        
#         # Final cleanup and enhancement
#         final_sketch = cv2.cvtColor(sketch_colored, cv2.COLOR_BGR2GRAY)
        
#         # Add a subtle edge enhancement for hair detail
#         kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#         final_sketch = cv2.filter2D(final_sketch, -1, kernel)
        
#         # Add a border to the sketch
#         border_thickness = 1  # Thickness of the border in pixels
#         border_color = (255, 255, 255)  # White border
#         final_sketch_with_border = cv2.copyMakeBorder(
#             final_sketch,
#             top=border_thickness,
#             bottom=border_thickness,
#             left=border_thickness,
#             right=border_thickness,
#             borderType=cv2.BORDER_CONSTANT,
#             value=border_color
#         )
        
#         return final_sketch_with_border

# # Flask Routes
# OUTPUT_FOLDER = "output"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# @app.route('/video_feed')
# def video_feed():
#     return Response(image_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/capture', methods=['POST'])
# def capture_frame():
#     return image_processor.capture_image()

# @app.route('/process', methods=['POST'])
# def process_frame():
#     return image_processor.process_image()

# @app.route('/image/<filename>')
# def get_image(filename):
#     return send_from_directory(image_processor.output_dir, filename)

# @app.route('/draw', methods=['POST'])
# def start_drawing():
#     """Publishes to the publisherStarter_ topic when the start drawing button is pressed."""
#     msg = Bool()
#     msg.data = True
#     image_processor.publisherStarter_.publish(msg)
#     return jsonify({"message": "Drawing started!"}), 200

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     image_processor.delete_old_files()
#     if 'image' not in request.files:
#         return "No file part", 400

#     file = request.files['image']
#     if file.filename == '':
#         return "No selected file", 400

#     # Save the file to the output folder
#     new_filename = "0_webcam.jpg"
#     file_path = os.path.join(image_processor.output_dir, new_filename)
#     file.save(file_path)
#     image_processor.resize_image()
#     return f"File uploaded successfully: {new_filename}", 200

# def main(args=None):
#     rclpy.init(args=args)
#     global image_processor
#     image_processor = ImageProcessor()
#     rclpy.spin(image_processor)
#     image_processor.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()