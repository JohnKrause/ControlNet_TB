import numpy as np
import cv2
import random

def scale_distortion(image):
    scale_factor = random.uniform(0.9, 1.1)
    height, width, _ = image.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_img = cv2.resize(image, (new_width, new_height))
    
    if scale_factor > 1:
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        result = resized_img[y_offset:y_offset+height, x_offset:x_offset+width]
    else:
        result = np.zeros_like(image)
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
    
    return result

def scale_translation(image):
    tx, ty = random.randint(-5, 5), random.randint(-5, 5)
    height, width, _ = image.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, (width, height))

def scale_vignette(image):
    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    center_x += random.randint(-20, 20)
    center_y += random.randint(-20, 20)
    diameter = random.randint(600, 768)
    radius = diameter // 2
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    try:
        image = cv2.bitwise_and(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    except Exception as e:
        raise Exception(f"{e}: \n Shape image:{image.shape}, Shape mask:{mask.shape}")
    return image

def scale_pixel_values(image):
    scale_factor = random.uniform(0.8, 1)
    scaled_image = np.clip(image * scale_factor, 0, 255).astype(np.uint8)
    return scaled_image


def random_distortion(image):
    distortion_functions = [scale_distortion, scale_translation, scale_vignette, scale_pixel_values]
    num_distortions = random.randint(0, 4)
    selected_distortions = random.sample(distortion_functions, num_distortions)

    for distortion in selected_distortions:
        image = distortion(image)

    return image

# Load your image using cv2.imread('path/to/your/image.png')
# Example usage: distorted_image = random_distortion(image)
