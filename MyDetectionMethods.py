import cv2 as cv
import numpy as np


class MyDetectionMethods:
    def applyCannyEdgeFilter(self, image, threshold1=100, threshold2=200, debug=False):
        """
        Apply Canny edge detection and return contours.

        Args:
            image (numpy.ndarray): Input image.
            threshold1 (int): First threshold for Canny edge detection.
            threshold2 (int): Second threshold for Canny edge detection.

        Returns:
            tuple: Contours and hierarchy.
        """
        # If the image is already single-channel, no need to convert
        if len(image.shape) == 3:  # If it's a color image
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert color to grayscale
        else:
            gray = image  # If it's already single-channel (grayscale or binary), use as-is

        # Apply Canny edge detection
        cannyEdgesImage = cv.Canny(gray, threshold1, threshold2)

        # Find contours in the edge-detected image
        contours, hierarchy = cv.findContours(cannyEdgesImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            cv.imshow('Canny Edge Image', cannyEdgesImage)

        return contours, hierarchy, cannyEdgesImage

    def applyBinaryEdgeFilter(self, image, threshold1=100, threshold2=255, debug=False, useAT=False):
        """
        Apply binary thresholding and return contours.

        Args:
            image (numpy.ndarray): Input image.
            threshold1 (int): Threshold value.
            threshold2 (int): Maximum value to use with THRESH_BINARY.
            debug (BOOL): If debug is on, binarized image gets opened.

        Returns:
            tuple: Contours (list), processed binary image (numpy.ndarray).
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale

        if useAT:
            thresholded_image = cv.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                                     thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)
        else:
            _, thresholded_image = cv.threshold(gray, threshold1, threshold2,
                                                cv.THRESH_BINARY)  # Apply binary thresholding

        # Debug option to visualize the binary image
        if debug:
            cv.imshow('Binary Image', thresholded_image)  # Show the binary image

        # Find contours from the binary image
        contours, _ = cv.findContours(thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        return contours, thresholded_image  # Return contours and binary image

    def find_centroid(self, contours):
        """
        Finds the centroid of the largest contour from a list of contours.
        Handles both a single contour or a list of contours.

        Parameters:
            contours (list or numpy.ndarray): A list of contours or a single contour (array of points).

        Returns:
            tuple: Centroid coordinates (x, y), or None if centroid calculation fails.
        """
        # If a single contour is passed, convert it to a list of contours
        if isinstance(contours, np.ndarray):
            contours = [contours]

        # Check if contours is empty
        if len(contours) == 0:
            print("No contours found.")
            return None, None

        # Find the largest contour (if there are multiple)
        largest_contour = max(contours, key=cv.contourArea)  # Find the largest contour

        # Ensure the contour has a non-zero area
        area = cv.contourArea(largest_contour)
        if area == 0:
            print("Largest contour has zero area.")
            return None, None

        # Calculate moments of the largest contour
        M = cv.moments(largest_contour)

        # Ensure we avoid division by zero
        if M['m00'] != 0:
            # Calculate the centroid coordinates
            cX = int(M['m10'] / M['m00'])  # X coordinate of centroid
            cY = int(M['m01'] / M['m00'])  # Y coordinate of centroid
            return cX, cY
        else:
            print("Centroid calculation failed (m00 == 0).")
            return None, None  # Return None if centroid calculation fails

    def find_orientation_slow(self, image, centroid):
        """
        Calculate the orientation of an object in a binary or edge-detected image.

        This function computes the orientation of an object using the method of moments.
        It calculates the angle of the major axis of the object with respect to the horizontal axis.

        Parameters:
        image (numpy.ndarray): A 2D image where the object is represented by non-zero pixels.
                               This can be a binary image or an edge-detected image (e.g., from Canny).
        centroid (tuple): The (x, y) coordinates of the object's centroid.

        Returns:
        float: The orientation angle in degrees. The angle is measured
               counterclockwise from the horizontal axis.

        Note:
        - For binary images, the object should be represented by white pixels (255) on a black background (0).
        - For edge images (e.g., Canny output), any non-zero pixel is considered part of the object.
        - The centroid should be pre-calculated and passed to this function.
        """
        a = 0
        b = 0
        c = 0
        rows, cols = image.shape
        for i in range(rows):  # 'i' is y (from rows)
            for j in range(cols):  # 'j' is x (from columns)
                if image[i, j] != 0:  # Consider any non-zero pixel as part of the object
                    a += (i - centroid[1]) ** 2
                    b += 2 * (i - centroid[1]) * (j - centroid[0])
                    c += (j - centroid[0]) ** 2

        orientation = 0.5 * np.arctan2(b, a - c)
        return np.rad2deg(orientation)

    def find_orientation(self, contour, centroid):
        """
        Calculate the orientation of an object based on the contour using image moments.

        Parameters:
            contour (list): The list of points that define the object's contour.
            centroid (tuple): The (x, y) coordinates of the object's centroid.

        Returns:
            float: The orientation angle in degrees.
        """
        # Create a blank mask with the same size as the image
        mask = np.zeros((500, 500), dtype=np.uint8)  # Modify size if necessary

        # Ensure the contour is in the correct format: (n, 1, 2)
        contour = np.array(contour, dtype=np.int32)
        contour = contour.reshape((-1, 1, 2))  # Ensure the shape is (n, 1, 2)

        # Draw the contour on the mask (fill the contour with white)
        cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

        # Calculate moments for the masked object region
        moments = cv.moments(mask)

        # Check if moments are valid to avoid division by zero
        if moments['mu02'] == 0 or moments['mu11'] == 0:
            return 0.0  # Return 0° if orientation cannot be determined

        # Compute the orientation angle using central moments
        angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])

        # Convert from radians to degrees
        return np.rad2deg(angle)  # Convert to degrees

    def maskImageWithColorRanges(self, img, color_ranges, cleanup=True, debug=False, return_individual=False):
        """
        Create a binary mask from multiple color ranges.

        Args:
            img (np.ndarray): Input BGR image.
            color_ranges (dict): Color name to (lower, upper) BGR range mapping.
            cleanup (bool): Apply morphological closing if True.
            debug (bool): Show masked image if True.
            return_individual (bool): Return individual masks per color if True.

        Returns:
            np.ndarray: Combined binary mask.
            dict (optional): Individual color masks if return_individual is True.
        """

        # Initialize empty mask
        img_masked = np.zeros(img.shape[:2], dtype=np.uint8)  # select height & width from img shape, creates new black
        individual_masks = {}

        # Loop through all color ranges dynamically
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv.inRange(img, lower, upper)
            img_masked = cv.bitwise_or(img_masked, mask)
            if return_individual:
                individual_masks[color_name] = mask

        # Morphological clean-up
        if cleanup:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            img_masked = cv.morphologyEx(img_masked, cv.MORPH_CLOSE, kernel)

        if debug:
            cv.imshow('Masked Image', img_masked)

        if return_individual:
            return img_masked, individual_masks
        else:
            return img_masked

    def mask_image_with_color_ranges_bgr(self, img, color_ranges, cleanup=True, debug=False, return_individual=False):
        """
        Create a binary mask from multiple color ranges.

        Args:
            img (np.ndarray): Input BGR image.
            color_ranges (dict): Color name to (lower, upper) BGR range mapping.
            cleanup (bool): Apply morphological closing if True.
            debug (bool): Show masked image if True.
            return_individual (bool): Return individual masks per color if True.

        Returns:
            np.ndarray: Combined binary mask.
            dict (optional): Individual color masks if return_individual is True.

            V1
        """

        # Initialize empty mask
        img_masked = np.zeros(img.shape[:2],
                              dtype=np.uint8)  # select height & width from img shape, creates new black
        individual_masks = {}

        # Loop through all color ranges dynamically
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(img, lower, upper)
            img_masked = cv2.bitwise_or(img_masked, mask)
            if return_individual:
                individual_masks[color_name] = mask

        # Morphological clean-up
        if cleanup:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel)

        if debug:
            cv2.imshow('Masked Image', img_masked)

        if return_individual:
            return img_masked, individual_masks
        else:
            return img_masked