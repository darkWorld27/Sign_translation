import cv2
import os
from PIL import Image

# Function to convert images to grayscale
def convert_images_to_grayscale(input_folder, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        print(f"Found file: {filename}")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing file: {filename}")
            try:
                # Open an image file
                img = Image.open(os.path.join(input_folder, filename)).convert('L')
                # Save the grayscale image to the output folder
                output_file_path = os.path.join(output_folder, filename)
                img.save(output_file_path)
                if os.path.exists(output_file_path):
                    print(f"Successfully saved {output_file_path}")
                else:
                    print(f"Failed to save {output_file_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipping file: {filename} (unsupported extension)")

# Input and output folder paths
input_folder = r"K:\Sign-Language-To-Text-and-Speech-Conversion-master (2)\Sign-Language-To-Text-and-Speech-Conversion-master\Sign-Language-To-Text-and-Speech-Conversion-master\dataset\Indian\T"
output_folder = r"K:\Sign-Language-To-Text-and-Speech-Conversion-master (2)\Sign-Language-To-Text-and-Speech-Conversion-master\Sign-Language-To-Text-and-Speech-Conversion-master\gray\T"

# Convert images to grayscale
convert_images_to_grayscale(input_folder, output_folder)
