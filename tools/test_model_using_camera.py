"""
test_model_using_camera.py

Interactive evaluator for testing ML classifier models using live camera images or saved files.
Captures an image, applies cropping for visualization, runs the classifier, and displays predictions.
Intended for development and validation of image-based classifiers.
"""

import logging
import cv2
import vision_pipeline
import ML.classifier_evaluation as ml
from vision_pipeline import VisionProcessor


def main():
    """
    Run camera/model evaluation loop with interactive key control.
    SPACE: step through predictions.
    ESC: exit.
    """
    # Initialize camera and evaluator
    Camera = vision_pipeline.Camera(exposure_time=30000.0, frame_rate=30.0)
    ml_evaluator = ml.ClassifierEvaluator()
    print(f"Loaded model with classes: {ml_evaluator.classes}")

    try:
        while True:
            # Capture BGR image from camera; can swap with direct file input for testing
            bgr = Camera.capture_raw()  # BGR, numpy array
            # To test using a file, uncomment:
            # bgr = cv2.imread(r'image_data/test_images/correct_yellow_121125_00.jpg')

            if bgr is None:
                logging.error("Failed to capture image")
                break

            display_img = bgr.copy()
            # Show cropped view (use BGR for OpenCV display)
            cv2.imshow("Captured Image", VisionProcessor.crop(display_img))

            # Classify using trained model
            results, status = ml_evaluator.predict_one(bgr, input_is_bgr=True)
            status_str = "GOOD" if status else "BAD"
            print(f"Prediction: {status_str}")
            for region, label, conf in results:
                print(f" {region}: {label:<20} conf: {conf * 100:.2f}%")

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC to exit
                break
            cv2.destroyAllWindows()
    finally:
        Camera.shutdown()


if __name__ == "__main__":
    main()
