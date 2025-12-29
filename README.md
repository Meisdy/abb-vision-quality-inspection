# Vision-Based Quality Inspection

Python + Basler camera + ABB robot system for automatic GOOD/BAD part inspection, using a supervised ML classifier and TCP/IP.

## Components

- `main.py` – Runs online inspection loop and TCP server for the robot.
- `vision_pipeline.py` – Basler Camera wrapper and image preprocessing/ROI utilities.
- `classifier_training.py` / `classifier_evaluation.py` – Train and evaluate the image classifier.
- `BaslerCapture*.py`, `preprocess_images.py`, `augment_images_in_folder.py`, `chose_ROI.py` – Dataset capture, ROI selection, preprocessing and augmentation tools.
- `abb_robot_comm.py` – TCP communication with ABB robot.
- `MesaMainSYI.mod` – RAPID program integrating the robot with the vision system.

## Usage (online inspection)

1. Configure IP, camera, and model paths in `main.py`, `abb_robot_comm.py` and RAPID Script.
2. Start the Python server
3. Start the ABB Script, which connects to the server when available and chosen in setting flags.
