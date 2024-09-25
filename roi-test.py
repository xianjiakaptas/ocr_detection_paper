import cv2
import numpy as np
import os

def detect_and_warp_paper(image, output_path, filename):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for detecting white color in HSV
    lower_white = np.array([0, 0, 200])  # Lower bound for white in HSV
    upper_white = np.array([180, 50, 255])  # Upper bound for white in HSV
    
    # Create a mask for white regions in the image
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find contours in the mask (white regions)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour which is most likely the paper
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the largest contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            # If the contour has 4 points, proceed with perspective transformation
            pts = approx.reshape(4, 2)
            print(f"Quadrilateral detected in {filename}, performing perspective transformation.")
        else:
            # Use the minimum area bounding box if the contour is not a quadrilateral
            print(f"Contour in {filename} is not a quadrilateral. Using minimum area bounding box.")
            rect = cv2.minAreaRect(largest_contour)
            pts = cv2.boxPoints(rect)
            pts = np.int0(pts)  # Convert to integer

        # Sort the points in order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        # Get the dimensions of the new transformed image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Set up the destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Get the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Save the warped (extracted and aligned) image
        output_filename = os.path.join(output_path, f'warped_{filename}')
        cv2.imwrite(output_filename, warped)
        print(f"Warped image saved to: {output_filename}")

        # Also save the contour-drawn image for reference
        image_with_contour = image.copy()
        cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 3)
        contour_filename = os.path.join(output_path, f'contour_{filename}')
        # cv2.imwrite(contour_filename, image_with_contour)
        print(f"Contour image saved to: {contour_filename}")
    
    else:
        print(f"No contours found in {filename}")
    
    return None  # Return None if no paper-like region is found


# Example usage
input_folder = './radio_meter/'
output_folder = './warped_results/'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        image_path = os.path.join(input_folder, filename)

        # Load the image
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            continue

        # Detect, extract, and warp the largest contour
        detect_and_warp_paper(image, output_folder, filename)
