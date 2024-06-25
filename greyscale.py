import os
import cv2

def convert_images_to_grayscale_and_rgb(source_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for dirpath, dnames, fnames in os.walk(source_dir):
        for f in fnames:
            if f.endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(src_path, source_dir)
                dest_path = os.path.join(dest_dir, rel_path)

                # Ensure the destination directory exists
                dest_dir_path = os.path.dirname(dest_path)
                if not os.path.exists(dest_dir_path):
                    os.makedirs(dest_dir_path)

                # Read the image file using OpenCV
                img = cv2.imread(src_path)
                if img is None:
                    print(f"Failed to load image {src_path}")
                    continue

                try:
                    # Convert image to 8-bit grayscale
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Convert grayscale image to RGB
                    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

                    # Save the RGB image
                    cv2.imwrite(dest_path, rgb_img)
                    print(f"Converted and saved: {dest_path}")
                except Exception as e:
                    print(f"Could not process image {src_path}: {e}")

if __name__ == "__main__":
    source_directory = "/Users/vladpavlovich/Downloads/FaceDataSet/Original Images/Original Images/"
    destination_directory = "/Users/vladpavlovich/Downloads/FaceDataSet/Grayscale Images/"

    convert_images_to_grayscale_and_rgb(source_directory, destination_directory)
