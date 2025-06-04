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

import torch
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/face-parsing.PyTorch"))
from model import BiSeNet

app = Flask(__name__) # Flask Web Server
CORS(app)  # Enable CORS for all routes

class ImageProcessor(Node):
    #---------- Initialiser ----------
    facesToggle = False

    def __init__(self):
        super().__init__('image_processor')
        self.get_logger().info("Image Processor Node Started")

        # Output directory for saved images
        self.output_dir = os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/output")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.delete_old_files()  # Delete old images

        # Load Dlib face detector
        self.detector = dlib.get_frontal_face_detector()
        model_path = os.path.expanduser("~/git/41069_WS_LAB4_G1/pablo/dlib/shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(model_path)

        # Start video capture
        self.cap = cv2.VideoCapture(2) # 0 for default camera, 2/4 for external camera
        self.current_frame = None

        # Start Flask server in a separate thread
        self.server_thread = Thread(target=self.run_flask, daemon=True)
        self.server_thread.start()

        # Create a publisher
        self.publisher_ = self.create_publisher(Bool, 'image_processed', 10)
        self.publisherStarter_ = self.create_publisher(Bool, 'starter', 10)

        # Load the BiSeNet model
        self.bisenet = BiSeNet(n_classes=19)
        self.bisenet.load_state_dict(torch.load(
            os.path.expanduser('~/git/41069_WS_LAB4_G1/pablo/face-parsing.PyTorch/res/cp/79999_iter.pth'),
            map_location='cpu'
        ))
        self.bisenet.eval()
        self.bisenet.to('cuda' if torch.cuda.is_available() else 'cpu')

        self.face_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

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

    #---------- Toggle ----------
    def face_toggle(self):
        ImageProcessor.facesToggle = not ImageProcessor.facesToggle
        self.get_logger().info(f'Face toggle set to: {ImageProcessor.facesToggle}')

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
            return "Image captured successfully!", 200

    #---------- Resize Image ----------
    def resize_image(self):
        image_path = os.path.join(self.output_dir, "0_webcam.jpg")
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().error('No image captured yet!')
            return jsonify({"error": "No image captured yet!"}), 400

        # Crop the image to a 4:3 landscape aspect ratio, centered
        h, w = image.shape[:2]
        desired_aspect = 4 / 3

        # Calculate new width and height to crop
        if w / h > desired_aspect:
            # Image is too wide, crop width
            new_w = int(h * desired_aspect)
            new_h = h
            x1 = (w - new_w) // 2
            y1 = 0
        else:
            # Image is too tall, crop height
            new_w = w
            new_h = int(w / desired_aspect)
            x1 = 0
            y1 = (h - new_h) // 2

        cropped = image[y1:y1 + new_h, x1:x1 + new_w]

        # Optionally, resize to a standard size (e.g., 960x720)
        final = cv2.resize(cropped, (960, 720), interpolation=cv2.INTER_AREA)

        cv2.imwrite(image_path, final)
        self.get_logger().info(f'Captured Image cropped to 4:3 and saved to: {image_path}')
        return jsonify({"message": "Image cropped to 4:3 successfully!"}), 200

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
        
        # # Find the largest face (by area)
        # if len(faces) > 1:
        #     faces = sorted(faces, key=lambda rect: rect.width() * rect.height(), reverse=True)

        # Find the face closest to the center of the image
        if len(faces) > 1:
            img_h, img_w = gray_image.shape
            img_cx, img_cy = img_w // 2, img_h // 2
            def face_center_distance(face):
                fx = face.left() + face.width() // 2
                fy = face.top() + face.height() // 2
                return (fx - img_cx) ** 2 + (fy - img_cy) ** 2
            faces = sorted(faces, key=face_center_distance)

        # Hide all faces except the largest by drawing black rectangles
        if not ImageProcessor.facesToggle and len(faces) > 1:
            img_height = no_bg_face_bgr.shape[0]
            for other_face in faces[1:]:
                ox, ow = other_face.left(), other_face.width()
                # Rectangle spans full image height
                cv2.rectangle(no_bg_face_bgr, (ox, 0), (ox + ow, img_height), (0, 0, 0), thickness=-1)
        
        gray_image = cv2.cvtColor(no_bg_face_bgr, cv2.COLOR_BGR2GRAY)

        # Use face_toggle to decide whether to process one or all faces
        # True = Group Photo
        if ImageProcessor.facesToggle:
            all_landmarks = [self.predictor(gray_image, face) for face in faces]
        else:
            if len(faces) > 1:
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Increase border by 40%
                border_x = int(w)
                border_y = int(h)

                # New coordinates with border, ensuring they stay within image bounds
                x1 = max(0, x - border_x)
                y1 = max(0, y - border_y)
                x2 = min(gray_image.shape[1], x + w + border_x)
                y2 = min(gray_image.shape[0], y + h + border_y)

                # Crop a 4:3 landscape region around the expanded face rectangle
                crop_w = x2 - x1
                crop_h = y2 - y1
                desired_w = crop_w
                desired_h = int(desired_w * 3 / 4)

                # Adjust height if needed to maintain 4:3 aspect ratio
                if crop_h < desired_h:
                    extra = desired_h - crop_h
                    y1 = max(0, y1 - extra // 2)
                    y2 = min(gray_image.shape[0], y2 + (extra - extra // 2))
                elif crop_h > desired_h:
                    # Center crop to desired_h
                    center_y = (y1 + y2) // 2
                    y1 = max(0, center_y - desired_h // 2)
                    y2 = y1 + desired_h
                    if y2 > gray_image.shape[0]:
                        y2 = gray_image.shape[0]
                        y1 = y2 - desired_h

                # Final crop
                cropped_bgr = no_bg_face_bgr[y1:y2, x1:x2]
                cropped_gray = gray_image[y1:y2, x1:x2]

                # Overwrite images for further processing
                no_bg_face_bgr = cropped_bgr
                gray_image = cropped_gray

                # Update face rectangle for cropped image
                new_face = dlib.rectangle(
                    left=face.left() - x1,
                    top=face.top() - y1,
                    right=face.right() - x1,
                    bottom=face.bottom() - y1
                )
                all_landmarks = [self.predictor(gray_image, new_face)]
            else:
                all_landmarks = [self.predictor(gray_image, faces[0])]

        # Convert to grayscale for sketch effect
        output_path_gray = os.path.join(self.output_dir, '1_no_background.jpg')
        cv2.imwrite(output_path_gray, gray_image)
        self.get_logger().info(f'Face with background removed saved to: {output_path_gray}')

        no_bg_face_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Resize and preprocess image for BiSeNet
        input_image = cv2.resize(no_bg_face_bgr, (512, 512))
        input_tensor = self.face_transforms(input_image)
        input_tensor = input_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Run through BiSeNet
        with torch.no_grad():
            out = self.bisenet(input_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # Hair mask = class index 17 in CelebAMask-HQ
        hair_mask = (parsing == 17).astype(np.uint8) * 255
        hair_mask = cv2.resize(hair_mask, (no_bg_face_bgr.shape[1], no_bg_face_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Save as PNG with transparency
        hair_mask_png = np.stack([hair_mask] * 3 + [hair_mask], axis=-1)  # RGBA
        hair_mask_path = os.path.join(self.output_dir, "3_hair_mask.png")
        cv2.imwrite(hair_mask_path, hair_mask_png)
        self.get_logger().info(f'Hair mask saved to: {hair_mask_path}')

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

        hair_contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw hair contours on the edges_colored image
        cv2.drawContours(mask, hair_contours, -1, 255, 1) 

        # Convert edges to 3 channels
        edges_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw landmark lines
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
        for landmarks in all_landmarks:
            draw_landmark_lines(edges_colored, landmarks, facial_features["jaw"])
            draw_landmark_lines(edges_colored, landmarks, list(range(17, 22)))
            draw_landmark_lines(edges_colored, landmarks, list(range(22, 27)))
            draw_landmark_lines(edges_colored, landmarks, facial_features["nose"])
            for part in facial_features["eyes"] + facial_features["mouth"]:
                draw_landmark_lines(edges_colored, landmarks, part)

        # Save final sketch
        final_sketch = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2GRAY)

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

@app.route('/faceToggle', methods=['POST'])
def toggle_face():
    return image_processor.face_toggle()

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


