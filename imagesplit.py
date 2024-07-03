import os
import shutil
import random

# Define the source and destination directories
source_dir = "/Users/vladpavlovich/Desktop/MultipleTrue"
inputs_dir = "/Users/vladpavlovich/Desktop/MultipleInputs"
verification_dir = "/Users/vladpavlovich/Desktop/MultipleVerification"

# Ensure the destination directories exist
os.makedirs(inputs_dir, exist_ok=True)
os.makedirs(verification_dir, exist_ok=True)

# Get the list of people (subdirectories) in the source directory
people_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]


# Function to split the images and move them
def split_images(person_dir):
    # Get the list of images in the person's directory
    images = [f for f in os.listdir(os.path.join(source_dir, person_dir)) if
              os.path.isfile(os.path.join(source_dir, person_dir, f))]
    # Shuffle the images for random distribution
    random.shuffle(images)

    # Calculate the number of images for each destination
    num_images = len(images)
    num_per_group = num_images // 3

    # Split the images into three parts
    images_inputs = images[:num_per_group]
    images_verification = images[num_per_group:num_per_group * 2]
    images_true = images[num_per_group * 2:]

    # Create corresponding directories in the destination folders
    os.makedirs(os.path.join(inputs_dir, person_dir), exist_ok=True)
    os.makedirs(os.path.join(verification_dir, person_dir), exist_ok=True)

    # Move the images
    for img in images_inputs:
        shutil.move(os.path.join(source_dir, person_dir, img), os.path.join(inputs_dir, person_dir, img))

    for img in images_verification:
        shutil.move(os.path.join(source_dir, person_dir, img), os.path.join(verification_dir, person_dir, img))

    # Move remaining images back to original folder to retain its structure
    for img in images_true:
        shutil.move(os.path.join(source_dir, person_dir, img), os.path.join(source_dir, person_dir, img))


# Process each person's directory
for person_dir in people_dirs:
    split_images(person_dir)

print("Images have been successfully split and moved.")
