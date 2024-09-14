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
        downscaled_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LANCZOS4)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_img = cv2.filter2D(downscaled_img, -1, kernel)
        
        lab = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        enhanced_img = img

    cv2.imwrite(output_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return enhanced_img

# Example usage:
# downscale_image('path/to/input/image.jpg', 'path/to/output/image.jpg')


