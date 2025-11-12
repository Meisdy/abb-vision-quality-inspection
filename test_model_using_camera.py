# Modified test_model_using_camera.py
import logging
import cv2
import vision_pipeline
import ML.classifier_evaluation as ml
import numpy as np
from vision_pipeline import VisionProcessor

# Initialize camera once
Camera = vision_pipeline.Camera(exposure_time=30000.0, frame_rate=30.0)
ml_evaluator = ml.ClassifierEvaluator()
print(f"Loaded model with classes: {ml_evaluator.classes}")

while True:
    # Capture BGR image from camera (or read from disk with cv2.imread -> BGR)
    bgr = Camera.capture_raw()  # BGR, numpy array
    # Example: if you want to test using a file, uncomment this instead:
    # bgr = cv2.imread(r'image_data/test_images/correct_yellow_121125_00.jpg')

    if bgr is None:
        logging.error("Failed to capture image")
        break

    # Keep a BGR copy for display, convert to RGB for the model
    display_img = bgr.copy()

    # Convert BGR -> RGB so preprocessing matches Image.open(...).convert('RGB')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.asarray(rgb, dtype=np.uint8)  # ensure dtype

    # Show cropped view (use BGR for OpenCV display)
    try:
        cv2.imshow("Captured Image", VisionProcessor.crop(display_img))
    except Exception:
        cv2.imshow("Captured Image", display_img)

    # Classify using the trained model (expects RGB numpy)
    label, confidence = ml_evaluator.predict_one(rgb)
    print(f"Classification result: {label.upper()} with confidence {confidence:.2f}")

    if confidence < 0.6:
        logging.warning("Low confidence in prediction, marking as incorrect mix")
        label = 'incorrect_mix'

    part_status = False if label == 'incorrect_mix' else True
    print(f"Part status: {'GOOD' if part_status else 'Incorrect mix'}")

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
Camera.shutdown()
