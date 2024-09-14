import cv2
import numpy as np

def downscale_image(image_path, output_path, target_resolution=(1500, 1500)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image from {image_path}")

    h, w = img.shape[:2]
    
    scaling_factor = min(target_resolution[0] / w, target_resolution[1] / h)
    
    if scaling_factor < 1:
        new_dimensions = (int(w * scaling_factor), int(h * scaling_factor))
        downscaled_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
        
        enhanced_img = downscaled_img
    else:
        enhanced_img = img
        print("Image is already at the target resolution or smaller")

    cv2.imwrite(output_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return enhanced_img

# Example usage:
# downscale_image('path/to/input/image.jpg', 'path/to/output/image.jpg')
downscale_image("../images/41-NCxNuBxL.jpg", "test.jpg")


