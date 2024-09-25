import os
import cv2
from PIL import Image
import pytesseract
import re
import skimage.restoration

# Set the path to the folder containing the images
image_folder = "./warped_results"
output_folder = "./output_images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def clean_ocr_text(text):
    """Clean up the OCR text by removing unwanted characters but keeping parentheses and hyphens."""
    clean_text = re.sub(r'[^\w\s()-]', '', text)  # Keeps letters, digits, whitespace, parentheses, and hyphens
    return clean_text

def perform_ocr_without_preprocessing(image_folder, output_folder):
    # List all image files in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(image_folder, filename)
            print(f"Processing {filename}...")

            # Load the original image
            original_image = cv2.imread(image_path)

            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # denoised = skimage.restoration.denoise_wavelet(gray, multichannel=False)
            # Enhance contrast
            contrast_enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=5)
            


            # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 7, 2)

           
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # dilated = cv2.dilate(thresh, kernel, iterations=1)

            # cv2.imwrite(f"./output_images/constrast_{filename}.png", thresh)

            # contrast_enhanced = cv2.convertScaleAbs(dilated, alpha=1.5, beta=0)

            # Convert the original image from OpenCV format (BGR) to PIL format (RGB)
            pil_image = Image.fromarray(contrast_enhanced)

            # Perform OCR using Tesseract with custom configuration (LSTM engine only)
            # custom_config = r'--psm 3 --oem 3 -c tessedit_write_images=true'

            # Set the language explicitly to English using 'lang' parameter
            # text = pytesseract.image_to_string(pil_image, config=custom_config, lang='eng')
            text = pytesseract.image_to_string(pil_image)

            # Clean the OCR text (removes unwanted characters but keeps hyphens)
            clean_text = clean_ocr_text(text)

            # Split the text into lines to preserve line breaks
            lines = clean_text.split("\n")

            # Set the initial position for text display
            y_offset = 30
            line_height = 30  # Vertical spacing between lines

            # Write the cleaned OCR text line by line on the original image
            for line in lines:
                if line.strip():  # Avoid drawing empty lines
                    # Set the text color to red for better visibility (BGR format: (0, 0, 255))
                    cv2.putText(original_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    y_offset += line_height  # Move down for the next line

            # Save the final image with OCR text drawn on the original image
            output_image_path = os.path.join(output_folder, f"ocr_{filename}")
            cv2.imwrite(output_image_path, original_image)

            print(f"Saved final OCR image on original for {filename}")

# Run the function to process images and save the results
perform_ocr_without_preprocessing(image_folder, output_folder)
