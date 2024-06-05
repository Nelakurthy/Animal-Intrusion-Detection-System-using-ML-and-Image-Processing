import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
cred = credentials.Certificate('key.json')

# Initialize the app with a service account, granting admin privileges

firebase_admin.initialize_app(cred, {
    'databaseURL': "https://homeautomation-fbfc2-default-rtdb.firebaseio.com/"
})

class AnimalDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detector")

        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.animal_label = tk.Label(self.root, text="No animal detected", font=("Arial", 18))
        self.animal_label.pack(side=tk.RIGHT, padx=10, pady=10)

        self.cap = cv2.VideoCapture(0)  # Accessing default camera (0)

        # Load YOLO
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Define the specific animal classes to be detected
        self.target_animals = ['cow', 'tiger', 'elephant', 'lion', 'dog', 'cat']

        self.detect_animals()

    def detect_animals(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))

            # Convert the frame to a format suitable for displaying in Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            # Display the frame in Tkinter
            self.video_label.config(image=image)
            self.video_label.image = image

            # Object detection
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []

            animal_detected = False

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # threshold for detection confidence
                        label = self.classes[class_id]
                        if label in self.target_animals or label not in self.classes:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                            animal_detected = True

            if animal_detected:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = self.classes[class_ids[i]]
                        confidence = confidences[i]
                        animal_label = f"{label.capitalize()} Detected (Confidence: {confidence})"
                        self.animal_label.config(text="Animal detected")
                        ref = db.reference('ANIMAL')
                        ref.set({'STATUS': "1"})
                        break
            else:
                self.animal_label.config(text="No animal detected")
                ref = db.reference('ANIMAL')
                ref.set({'STATUS': "0"})

        self.root.after(10, self.detect_animals)  # Repeat the detection process

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalDetector(root)
    root.mainloop()
