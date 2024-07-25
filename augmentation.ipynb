import os
import uuid
import cv2
import numpy as np
import tensorflow as tf

def data_aug(img):
    # Generate one augmented image
    img_aug = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
    img_aug = tf.image.stateless_random_contrast(img_aug, lower=0.6, upper=1, seed=(1, 3))
    img_aug = tf.image.stateless_random_flip_left_right(img_aug,
                                                        seed=(np.random.randint(100), np.random.randint(100)))
    img_aug = tf.image.stateless_random_jpeg_quality(img_aug, min_jpeg_quality=90, max_jpeg_quality=100,
                                                     seed=(np.random.randint(100), np.random.randint(100)))
    img_aug = tf.image.stateless_random_saturation(img_aug, lower=0.9, upper=1,
                                                   seed=(np.random.randint(100), np.random.randint(100)))
    return img_aug

def augment_images(src_path):
    for root, _, files in os.walk(src_path):
        for file_name in files:
            img_path = os.path.join(root, file_name)
            print(f"Reading image: {img_path}")
            img = cv2.imread(img_path)
            if img is not None:
                print(f"Image loaded successfully: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0  # Normalize the image
                augmented_image = data_aug(img)  # Generate one augmented image
                save_path = os.path.join(root, f"{uuid.uuid4().hex}.jpg")
                image_np = np.clip(augmented_image.numpy() * 255, 0, 255).astype(np.uint8)
                success = cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                if success:
                    print(f"Saved augmented image to: {save_path}")
                else:
                    print(f"Failed to save augmented image to: {save_path}")
            else:
                print(f"Failed to load image: {img_path}")

# Adjust these paths to point to your main folders
TRUE_PATH = '/Users/vladpavlovich/Desktop/MultipleTrue 2'
ANCHOR_PATH = '/Users/vladpavlovich/Desktop/MultipleVerification 2'

print("Starting data augmentation for true images...")
augment_images(TRUE_PATH)

print("Starting data augmentation for anchor images...")
augment_images(ANCHOR_PATH)
