import os
import cv2
import numpy as np

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

class ASLDetector:
    """
    Wrapper around YOLOv8 model for ASL detection.
    Includes fallback simulation if ultralytics is not available.
    """
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.use_yolo = False
        
        # Check if we can load YOLOv8
        try:
            from ultralytics import YOLO
            # If the weights file is just our dummy mock weights, don't load it with YOLO
            # to prevent runtime errors (YOLO expects actual PyTorch serialized dict)
            if os.path.exists(weights_path) and os.path.getsize(weights_path) > 1000:
                self.model = YOLO(weights_path)
                self.use_yolo = True
                print("Loaded YOLOv8 model successfully.")
            else:
                print("Using YOLOv8 mock inference (weights file is empty or mock placeholder).")
        except ImportError:
            print("Using YOLOv8 mock inference (ultralytics package not installed).")
            
    def predict(self, frame):
        """
        Runs model inference on a frame.
        Returns: list of dicts: [{'box': [xmin, ymin, xmax, ymax], 'class': 'A', 'confidence': 0.92}]
        """
        h, w, _ = frame.shape
        if self.use_yolo:
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                coords = box.xyxy[0].tolist() # xmin, ymin, xmax, ymax
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                detections.append({
                    'box': [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                    'class': CLASSES[cls_id],
                    'confidence': conf
                })
            return detections
        else:
            # Mock Detection: simulate finding a hand in the center of the screen
            # Detects a bounding box and cycles through classes to demonstrate visual overlay
            box_size = 200
            xmin = (w - box_size) // 2
            ymin = (h - box_size) // 2
            xmax = xmin + box_size
            ymax = ymin + box_size
            
            # Simple heuristic: calculate skin color pixel ratio in the center box
            # to adjust confidence (simulating a basic computer vision threshold)
            center_crop = frame[ymin:ymax, xmin:xmax]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            # Broad HSV range for skin tone
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(mask > 0) / (box_size * box_size)
            
            if skin_ratio > 0.15: # Hand detected
                # Determine class based on skin ratio or time cycle
                import time
                cls_idx = int(time.time() // 3) % len(CLASSES)
                confidence = 0.75 + skin_ratio * 0.2
                confidence = min(0.99, confidence)
                return [{
                    'box': [xmin, ymin, xmax, ymax],
                    'class': CLASSES[cls_idx],
                    'confidence': confidence
                }]
            return []

def main():
    print("--- ASL Webcam Live Demo (Local OpenCV) ---")
    weights_path = 'runs/detect/train/weights/best.pt'
    
    detector = ASLDetector(weights_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open local webcam device.")
        print("Tip: If you are running in a headless VM/remote terminal, run the Server Demo (server.py) instead.")
        return
        
    print("Webcam started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.predict(frame)
        
        # Draw detections
        for det in detections:
            box = det['box']
            cls_name = det['class']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Label background
            label = f"{cls_name} ({conf:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (box[0], box[1] - h - 10), (box[0] + w + 10, box[1]), (0, 255, 0), -1)
            # Label text
            cv2.putText(frame, label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.imshow("ASL Hand Sign Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
