import cv2
import socket
import numpy as np
import math
import sys
import os
from pypylon import genicam
from pypylon import pylon

os.environ["PYLON_CAMEMU"] = "3"

# Constants
USE_CAMERA = True
USE_ROBOT = True
IMAGE_NAME = 'Image6.jpg'
IP_ABB_ROBOT = '192.168.125.202'
LIGHT_BLUE = (255, 255, 102)  # Light blue (BGR)
BLACK = (0, 0, 0)  # Black (BGR)
RED = (1, 0, 255)  # Red (BGR)
PINK = (255, 20, 220)
GREEN = (0, 249, 77)
Picture = []

COLOR_RANGES = {
    'Blue': (np.array([100, 80, 50]), np.array([130, 255, 255])),
    'Red': [(np.array([0, 120, 50]), np.array([10, 255, 255])),
            (np.array([160, 120, 50]), np.array([180, 255, 255]))],
    'Green': (np.array([40, 80, 40]), np.array([90, 255, 255])),

    # Expanded V for brighter whites, low S for desaturation
    'White': (np.array([0, 0, 190]), np.array([180, 60, 255])),

    # Lowered V and S for dark gray/black tones
    'Gray': (np.array([0, 0, 0]), np.array([180, 130, 90]))
}

SHAPE_IDS = {
    0: "Triangle",
    1: "Square",
    2: "Hexagon",
    3: "Circle",
    4: "Star"
}

# Camera Const
maxCamerasToUse = 1
exitCode = 0


def main():
    if USE_ROBOT:

        # Connect the robot socket
        client_socket = connect_rob()

        # Create a list of possible colors
        color_names = list(COLOR_RANGES.keys())  # ['Blue', 'Gray', 'White', 'Red', 'Green']

        # Send Color choosing prompt to Robot
        defined_color = communicate_rob(
            "Select your colors, e.g. 124: 1-Blue, 2-Red, 3-Green, 4-White, 5-Black", client_socket)

        # Change the list
        colors_list = [color_names[int(char) - 1] for char in defined_color]

        # Send Shape choosing prompt
        defined_shape = communicate_rob(
            "Select your shapes, e.g. 124: 1-Triangle, 2-Square, 3-Hexagon, 4-Circle, 5-Star", client_socket)
        shapes_list = [int(char) - 1 for char in defined_shape]

        communicate_rob("Go away for picture", client_socket)

        # Get contour properties using vision algorithm
        contour_properties = get_object_data(colors_list, shapes_list)  # returns a vector with a dict of contours with
        cv2.waitKey()

        # Print Data as a table
        print_contour_summary(contour_properties)

        ##########################
        '''Create a function which go through the contour_properties and places first all triangles, then circles etc. 
            Or calls another function which send the robot the correct position for the part and also tells him where to
            put it, or simply, that is is e.g. a triangle and the robot program figures the correct location then'''

        for contour in contour_properties:
            shape = contour["Shape"]  # Shape-ID (0=Triangle, 1=Square, 2=Hexagon, 4=Hexagon, 4-Circle, 5-Star)
            center = contour["Center"]  # Center [x, y]
            orientation = contour["Orientation"]  # Orientation (in Grad)

            # Manual pos optimiser
            center = manual_position_optimiser(center)

            # assemble string
            # center = [42.0, 7.7]
            response = f"{center[0] * 10},{center[1] * 10},{orientation},{shape}"  # Times 10 cause robot uses mm

            # Send to robot
            communicate_rob(response, client_socket)

            # wait till robot is ready, sends a 1
            # while not int(rob_response) == 1:
            #    current_time = time.time()
            #    if current_time - start_time >= 10:  # Every 10 seconds
            #        print("Waiting for Robot to finish")
            #        start_time = current_time  # reset time

            communicate_rob('Done?', client_socket)

            # Quit Socket
            # disconnect_rob(client_socket)

            # === OPTION 1: Remove only first occurrence
            # if shape in shapes_list:
            #    shapes_list.remove(shape)  # Only first match
            # === OPTION 2: Remove ALL occurrences
            # shapes_list = [num for num in shapes_list if num != shape]

            # cv2.destroyAllWindows()
            # return

        # if no shapes found, tell the robot
        communicate_rob("No shapes found", client_socket)
        disconnect_rob(client_socket)
        cv2.waitKey()
        #cv2.destroyAllWindows()

    else:
        defined_colors = ['Red', 'Blue', 'White', 'Green', 'Gray']
        #defined_colors = ['Blue']
        defined_shapes = [0, 1, 2, 3, 4]
        print('Inputs for Vision Alg.:', defined_colors, 'and', [SHAPE_IDS[s] for s in defined_shapes])
        contour_properties = get_object_data(defined_colors, defined_shapes)
        print_contour_summary(contour_properties)
        cv2.waitKey()
        cv2.destroyAllWindows()


###############################
# Robot Communication Functions
def connect_rob():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set socket listening port. To find the robot address use cmd and type ipconfig in the terminal, there you see
    # the ip of the robot. you may have to open the port in the firewall as well. Note that the ip address is
    # 127.0.0.1 if you run via RobotStudio
    server_socket.bind((IP_ABB_ROBOT, 5000))

    # Set up the server
    # listen to incoming client connection
    server_socket.listen()
    print("Looking for client")

    # accept and store incoming socket connection
    (client_socket, client_ip) = server_socket.accept()
    print(f"Robot at address {client_ip} connected.")
    print("If you would like to end the program, enter 'quit'.")
    return client_socket


def disconnect_rob(client_socket):
    client_socket.close()


def communicate_rob(message, client_socket):
    while True:
        if message is None:
            UserInput = input("Please type your message: ")
        else:
            UserInput = message

        if UserInput.lower() == "quit":
            server_message = UserInput
            client_socket.send(server_message.encode("UTF-8"))
            print("Goodbye!")
            # wait for answer and print in terminal
            if client_socket.recv:
                client_message = client_socket.recv(4094)
                client_message = client_message.decode("latin-1")
                print("The received message is:", client_message)
                return client_message

            break

        else:
            client_socket.send(UserInput.encode("UTF-8"))
            print(f"Message sent to client: {message}")
            # Wait for answer and print in terminal
            if client_socket.recv:
                print("The received message is:")
                client_message = client_socket.recv(4094)
                client_message = client_message.decode("latin-1")
                print("!!", client_message)
                return client_message


###############################
# Vision functions

def detect_aruco_markers(frame):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return frame, corners


def calculate_pixel_to_cm_ratio(marker_corners, marker_real_size_cm):
    if marker_corners and len(marker_corners[0][0]) >= 2:
        marker_width_pixels = np.linalg.norm(marker_corners[0][0][0] - marker_corners[0][0][1])
        pixel_to_cm_ratio = marker_real_size_cm / marker_width_pixels
        return pixel_to_cm_ratio
    else:
        print("Error: Insufficient or invalid marker corners for ratio calculation.")
        return None


def get_object_data(defined_color=None, defined_shapes=None):
    try:
        converter = configure_converter()
        cameras = create_cameras(exposure_time=15000.0, frame_rate=30.0)

        if USE_CAMERA:
            frame = grab_images(cameras, converter)

            # Load the calibration file (npz)
            calibration_data = np.load('camera_calibration.npz')

            # Extract camera matrix, distortion coefficients, etc.
            camera_matrix = calibration_data['camera_matrix']
            dist_coeffs = calibration_data['dist_coeffs']
            # If you have other calibration data, you can extract them similarly:
            # For example, 'rvecs' and 'tvecs' for rotation and translation vectors
            # rvecs = calibration_data['rvecs']
            # tvecs = calibration_data['tvecs']


            # Use the camera matrix and distortion coefficients to undistort the image
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)


        else:
            # image_path = r"C:\Users\phili\Downloads\WhatsApp Image 2025-04-28 at 13.34.47.jpeg"
            # image_path = 'Image6.jpg'
            # frame = cv2.imread(image_path)
            frame = cv2.imread(IMAGE_NAME)

        if frame is None:
            print("Error: Image could not be loaded ...")
            exit()

        # frame[0:850, 0:900] = 0
        # Detect Aruco markers
        frame, aruco_corners = detect_aruco_markers(frame)

        # If no aruco is detected, exit function
        if not aruco_corners:
            print('No Aruco Marker detected!')
            return []

        # Proceed if markers are detected
        pixel_to_cm = calculate_pixel_to_cm_ratio(aruco_corners, 10)
        print("Pixel to cm ratio:", pixel_to_cm)

        # Set irrelevant arreas to black
        frame[:, :850] = [0, 0, 0]  # If the image is RGB (3 channels)

        # Filter the Image for the correct colored objects
        frame_masked, all_frame_masks, color_id_map = mask_image_with_color_ranges_hsv(frame, COLOR_RANGES,
                                                                                       debug=True,
                                                                                       return_individual=True,
                                                                                       cleanup=True)

        # Find contours in the masked image
        contours, _ = cv2.findContours(frame_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ADD COMMENT HERE
        frame, contour_properties = filter_objects_by_size(
            frame, contours, pixel_to_cm, COLOR_RANGES,
            color_id_map=color_id_map,
            defined_colors=defined_color,
            defined_shapes=defined_shapes
        )

        # Crate window in wanted ratio
        scale_percent = 100
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height))

        cv2.namedWindow("Bild", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bild", width, height)  # <-- important
        cv2.imshow("Bild", frame)

    except genicam.GenericException as e:
        print("An exception occurred.", e)
        sys.exit(1)
    except OSError as e:
        print("Disk space error:", e)
        sys.exit(1)

    return contour_properties


def mask_image_with_color_ranges_hsv(img, color_ranges, cleanup=False, debug=False, return_individual=False):
    """
    Create a combined mask and per-color masks using HSV. Also returns color ID map for dominant color.

    Args:
        img (np.ndarray): Input BGR image.
        color_ranges (dict): Color name to (lower HSV, upper HSV) range or list of ranges.
        cleanup (bool): Apply morphological closing if True.
        debug (bool): Show the final mask if True.
        return_individual (bool): Return per-color masks and color_id_map if True.

    Returns:
        combined_mask (np.ndarray): The combined binary mask.
        dict (optional): Per-color masks.
        np.ndarray (optional): Map of pixel-wise dominant color index (uint8).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    individual_masks = {}
    color_id_map = np.zeros(img.shape[:2], dtype=np.uint8)

    for idx, (color_name, bounds) in enumerate(color_ranges.items(), start=1):
        if isinstance(bounds, list):
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in bounds:
                mask |= cv2.inRange(hsv, lower, upper)
        else:
            lower, upper = bounds
            mask = cv2.inRange(hsv, lower, upper)

        if return_individual:
            individual_masks[color_name] = mask

        combined_mask = cv2.bitwise_or(combined_mask, mask)
        color_id_map[mask > 0] = idx

    if cleanup:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imshow('Combined HSV Mask', combined_mask)

    if return_individual:
        return combined_mask, individual_masks, color_id_map
    else:
        return combined_mask


def filter_objects_by_size(frame, contours, pixel_to_cm_ratio, color_ranges, color_id_map, defined_colors=None,
                           defined_shapes=None):
    if contours is None:
        print('No contours available in function filtered objects by size')
        return frame

    output_Matrix = []  # all properties of the detected contours will be stored here

    for idx, contour in enumerate(contours):

        # Get Data from Contour
        rect = cv2.minAreaRect(contour)
        if rect is None or rect[1][0] == 0 or rect[1][1] == 0:
            continue

        box = cv2.boxPoints(rect)
        if box is None or len(box) == 0:
            continue

        box = box.astype(np.int32)

        # --- Size detection part ---
        # Calculate the width and height in pixels
        width_px = np.linalg.norm(box[0] - box[1])
        height_px = np.linalg.norm(box[1] - box[2])

        # Convert width and height from pixels to centimeters
        width_cm = width_px * pixel_to_cm_ratio
        height_cm = height_px * pixel_to_cm_ratio

        # Check bounds and skip if not within limits
        if width_cm < 4 or width_cm > 7 or height_cm < 4 or height_cm > 7:
            continue

        # --- Color detection part ---
        mask_single = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_single, [contour], -1, 255, -1)

        if color_id_map is not None:
            color_pixels = color_id_map[mask_single > 0]
            if len(color_pixels) == 0:
                continue
            most_common_id = np.bincount(color_pixels).argmax()
            if most_common_id == 0:
                continue
            detected_color = list(color_ranges.keys())[most_common_id - 1]
        else:
            detected_color = 'Unknown'

        if detected_color not in defined_colors:
            continue

        # --- Shape detection ---
        shape_match, shape = is_shape(defined_shapes, contour, False)
        if shape is None:
            continue

        # --- Visualisation ---
        # Get values
        orientation = 90 - rect[2]  # Get the orientation of the box
        centroid = rect[0][0], rect[0][1]

        # Use specific functions for special shapes
        if shape == 0:  # Triangle
            orientation, centroid = get_equilateral_triangle_orientation(contour, True)
        if shape == 2:  # Hexagon
            orientation = get_angle_of_hexagon(contour)
        if shape == 1:  # Square
            orientation = orientation % 90 - 90  # Adjust so that no robot singularities occur
        if shape == 4: # Star
            orientation, centroid = get_star_angle(contour)


        # Get the center of the object as int values
        x, y = (int(centroid[0]), int(centroid[1]))

        # Draw elements
        cv2.circle(frame, (x, y), 5, PINK, -1)  # Draw the center of the object
        cv2.drawContours(frame, [box], -1, GREEN, 2)  # Draw the contur

        # Add text
        cv2.putText(frame, f"{idx}", (x + 20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
        cv2.putText(frame, f"{SHAPE_IDS[shape]}", (x + 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, .6, BLACK, 2)
        cv2.putText(frame, f"{detected_color}", (x + 20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, .6, BLACK, 2)
        cv2.putText(frame, f"{rect[0][0] * pixel_to_cm_ratio:.1f} | {rect[0][1] * pixel_to_cm_ratio:.1f}",
                    (x + 20, y + 70), cv2.FONT_HERSHEY_SIMPLEX, .6, BLACK, 2)
        cv2.putText(frame, f"{orientation:.1f}", (x + 20, y + 90), cv2.FONT_HERSHEY_SIMPLEX, .6, BLACK, 2)

        # Add Data to output matrix
        output_Matrix.append({"Shape": shape, "Orientation": orientation, "Measurements": [width_cm, height_cm],
                              "Center": [centroid[0] * pixel_to_cm_ratio,
                                         centroid[1] * pixel_to_cm_ratio], "ID": idx, "Color": detected_color})

    return frame, output_Matrix


def is_shape(wanted_shapes_list, contour, debug=False):
    detected_Contour = classify_contour(contour, debug=debug)
    # Is the detected contour in wanted? or is no shape selected
    if debug:
        print('isShape detected', SHAPE_IDS[detected_Contour])

    if detected_Contour in wanted_shapes_list or wanted_shapes_list is None:
        return True, detected_Contour
    else:
        return False, None


def classify_contour(contour, debug=False):
    """
    Classifies a contour based on the number of its approximated polygon vertices and circularity.

    Args:
        contour (np.ndarray): A single contour array (from cv2.findContours).
        debug (BOOL)        : prints out detected contour when activated. Standard False

    Returns:
        int: Shape ID:
             0 = Triangle,
             1 = Square,
             2 = Hexagon,
             3 = Circle (if circularity > 0.4),
             4 = Star (if circularity <= 0.4),
            -1 = Invalid or unclassified shape.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return 0  # Triangle
    elif vertices == 4:
        return 1  # Square
    elif 5 <= vertices <= 7:
        return 2  # Hexagon
    elif vertices > 7:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if debug:
            print("Circularity:", circularity)
            print('vertices: ', vertices)

        if circularity > 0.4:
            return 3  # Circle
        else:
            return 4  # Star
    else:
        return -1  # Invalid shape


def get_orientation_of_box_contour(contours):
    """
    Extracts the orientation angle from a list of rotated rectangle contours.

    Args:
        contours (list): A list of rotated rectangle tuples from cv2.minAreaRect().
                         Each item should be a tuple (center, size, angle).

    Returns:
        float: The orientation angle (in degrees) of the last contour in the list.
    """
    orientation = []
    for contour in contours:
        orientation = contour[2]

    return orientation


def print_contour_summary(output_matrix):
    """
    Prints a formatted summary of detected contours from the vision pipeline.

    Args:
        output_matrix (list): List of dicts with keys 'Shape', 'Orientation', 'Measurements', 'Center'.
    """
    for obj in output_matrix:
        shape_name = SHAPE_IDS.get(obj["Shape"], "Unknown")
        center_x, center_y = [round(c, 2) for c in obj["Center"]]
        orientation = round(obj["Orientation"], 2)
        width, height = [round(s, 2) for s in obj["Measurements"]]
        object_id = obj["ID"]
        color = obj["Color"]

        print(
            f"ID: {object_id:<5}| Shape: {shape_name:<8} |Color: {color:<8}| Center: ({center_x:>5.2f}, "
            f"{center_y:>5.2f}) |"f"Size: {width:>4.2f}×{height:>4.2f} cm | Orientation: {orientation:>5.2f}°")


def get_equilateral_triangle_orientation(contour, debug=False):
    """
    Calculates orientation of an equilateral triangle by measuring angle
    from the centroid to the furthest vertex.

    Args:
        contour (np.ndarray): Triangle contour (must have 3 vertices).
        debug (bool): Print debug info if True.

    Returns:
        float: Orientation angle in degrees (0° = right, CCW).
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 3:
        return None

    # Calculate centroid
    M = cv2.moments(approx)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    centroid = np.array([cx, cy])

    # Find the furthest vertex from centroid
    vertices = approx.reshape(-1, 2)
    dists = np.linalg.norm(vertices - centroid, axis=1)
    furthest_vertex = vertices[np.argmax(dists)]

    # Compute angle
    dx = furthest_vertex[0] - cx
    dy = furthest_vertex[1] - cy
    angle = np.degrees(np.arctan2(-dy, dx))  # negative for image Y-axis direction
    angle = ((angle + 180) % 120) - 120

    if debug:
        print(f"Centroid: ({cx:.1f}, {cy:.1f}), Furthest vertex: {furthest_vertex}, Angle: {angle:.2f}°")

    return angle, centroid

def get_star_angle(contour):
    """
    Computes a consistent orientation angle for a star contour.
    
    Args:
        contour (np.ndarray): Contour points of the star (Nx1x2 array).
    
    Returns:
        float: Angle in degrees (0° to 360°), where 0° points right.
    """
    # Flatten contour
    contour = contour.reshape(-1, 2)

    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    center = np.array([cx, cy])

    # Find furthest vertex from center (likely a star tip)
    dists = np.linalg.norm(contour - center, axis=1)
    furthest_point = contour[np.argmax(dists)]

    # Compute angle from center to furthest point
    vec = furthest_point - center
    angle = np.degrees(np.arctan2(-vec[1], vec[0])) % 72  # negative y for image coords

    return angle, center

def get_angle_of_hexagon(points):
    # Ensure input is a NumPy array
    points = np.array(points, dtype=np.float32)
    points = points.reshape(-1, 2)

    # Compute the center (centroid) of the polygon
    center = np.mean(points, axis=0)

    # Find the point furthest from the center
    distances = np.linalg.norm(points - center, axis=1)
    max_index = np.argmax(distances)
    furthest_point = points[max_index]

    # Vector from center to furthest point
    vec = furthest_point - center

    # Calculate the angle of this vector with respect to the Y-axis
    # For the Y-axis, we take the vector (0, 1) as the reference
    angle_rad = math.atan2(vec[0], vec[1])  # Swap x and y to calculate relative to the Y-axis
    angle_deg = math.degrees(angle_rad)

    angle_deg = angle_deg % 60

    return angle_deg


def manual_position_optimiser(coords):
    if coords[1] > 30:
        coords[1] += -.1
    if coords[1] < 20:
        coords[1] += .2

    if coords[0] > 30:
        coords[0] += -.2
        #coords[1] += .4

    return coords



##########################################
# Camera implementation
###########################################

def configure_converter():
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def create_cameras(exposure_time, frame_rate):
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Using device: ", cam.GetDeviceInfo().GetModelName())
        # Set the exposure time
        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(frame_rate)
        cam.Close()
    return cameras


def grab_images(cameras, converter):
    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    image1 = []
    frame_counter = 0
    # image_counter = 0

    while cameras.IsGrabbing():
        grabResult1 = cameras[0].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # grabResult2 = cameras[1].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult1.GrabSucceeded():
            frame_counter += 1
            image1 = converter.Convert(grabResult1).GetArray()
            break
    cameras.StopGrabbing()

    # Calc needed window size (z.B. 800px width, proportional height)
    target_width = 1500
    aspect_ratio = image1.shape[1] / image1.shape[0]  # width / height
    target_height = int(target_width / aspect_ratio)

    print(f"aspect ratio: {aspect_ratio}")
    # scale image (with aspect ratio)
    cv2.imwrite(r"output.png", image1)
    #resized = cv2.resize(image1, (target_width, target_height))
    Picture = image1
    return image1


##########################################

if __name__ == "__main__":
    main()

###########################################
