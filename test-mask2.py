"""
Shadow detection that properly masks SHADOWS (dark regions cast on surfaces).

Shadows are characterized by:
1. Being darker than surrounding areas
2. Low saturation (grayish)
3. Appearing on surfaces with some brightness
4. Having relatively soft edges
"""

import numpy as np
import cv2
from skimage.color import rgb2lab, rgb2hsv
from skimage.morphology import disk, closing, opening, remove_small_objects
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter


def detect_shadows_correct(img_path, sensitivity=0.5):
    """
    Detect actual shadow regions (dark areas on surfaces).
    
    Parameters:
        img_path: path to image
        sensitivity: 0-1, higher = more aggressive shadow detection
    """
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Image could not be loaded.")
    
    # Handle alpha channel
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Get dimensions
    h, w = img_float.shape[:2]
    
    # 1. Convert to LAB color space
    lab = rgb2lab(img_float)
    L = lab[:, :, 0]  # Lightness channel (0-100)
    A = lab[:, :, 1]  # A channel
    B = lab[:, :, 2]  # B channel
    
    # Normalize L to 0-1
    L_norm = L / 100.0
    
    # 2. Convert to HSV to get saturation
    hsv = rgb2hsv(img_float)
    S = hsv[:, :, 1]  # Saturation
    V = hsv[:, :, 2]  # Value
    
    # 3. Compute local statistics to find areas darker than surroundings
    window_size = 25
    local_mean = uniform_filter(L_norm, size=window_size)
    local_std = np.sqrt(np.maximum(
        uniform_filter(L_norm**2, size=window_size) - local_mean**2, 
        0
    ))
    
    # Shadow score based on being darker than local average
    # Normalized difference: (local_mean - pixel) / (local_std + epsilon)
    darkness_score = np.zeros_like(L_norm)
    valid_mask = local_std > 0.01
    darkness_score[valid_mask] = (local_mean[valid_mask] - L_norm[valid_mask]) / (local_std[valid_mask] + 0.01)
    darkness_score = np.clip(darkness_score, 0, None)  # Only positive values (darker than average)
    
    # 4. Shadows have low saturation (appear grayish)
    low_saturation_score = 1.0 - S
    
    # 5. Exclude very dark objects (shadows appear on relatively bright surfaces)
    # If the local mean is too dark, it's probably a dark object, not a shadow
    surface_brightness = local_mean
    bright_surface_mask = surface_brightness > 0.25
    
    # 6. Exclude sky (very bright upper regions)
    sky_mask = np.zeros((h, w), dtype=bool)
    sky_threshold = 0.8
    top_region = int(h * 0.5)  # Only check top 50%
    sky_mask[:top_region] = L_norm[:top_region] > sky_threshold
    
    # Expand sky mask
    sky_mask = closing(sky_mask.astype(np.uint8), disk(20)).astype(bool)
    
    # 7. Combine features
    # Weight the different components
    shadow_score = (
        darkness_score * 0.6 +           # Being darker than surroundings
        low_saturation_score * 0.3 +     # Low saturation
        (L_norm < 0.5).astype(float) * 0.1  # Overall darkness
    )
    
    # Apply constraints
    shadow_score[~bright_surface_mask] *= 0.2  # Penalize dark surfaces
    shadow_score[sky_mask] = 0  # Exclude sky
    
    # 8. Threshold to create binary mask
    # Adjust threshold based on sensitivity
    base_threshold = 0.5
    threshold = base_threshold * (1.0 - sensitivity * 0.3)
    
    shadow_mask = shadow_score > threshold
    
    # 9. Morphological cleanup
    # Remove very small regions
    shadow_mask = remove_small_objects(shadow_mask, min_size=100)
    
    # Smooth the mask
    shadow_mask = closing(shadow_mask, disk(5))
    shadow_mask = opening(shadow_mask, disk(3))
    
    # Convert to uint8
    shadow_mask = shadow_mask.astype(np.uint8) * 255
    
    return shadow_mask, img_bgr, shadow_score


def overlay_mask(img, mask, color=(0, 0, 255), alpha=0.6):
    """Overlay shadow mask on original image."""
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Create colored overlay
    overlay = img.copy()
    overlay[mask_binary == 1] = color
    
    # Blend
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return result


def test_shadow_detection(img_path, sensitivity=0.5):
    """Test shadow detection with visual output using OpenCV windows."""
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Image could not be loaded.")
    
    try:
        # Detect shadows
        shadow_mask, img, shadow_score = detect_shadows_correct(img_path, sensitivity)
        
        # Create visualizations
        overlay = overlay_mask(img, shadow_mask, color=(0, 0, 255), alpha=0.6)
        
        # Shadow score heatmap (convert to 0-255 and apply colormap)
        score_norm = (shadow_score / shadow_score.max() * 255).astype(np.uint8) if shadow_score.max() > 0 else (shadow_score * 255).astype(np.uint8)
        score_colored = cv2.applyColorMap(score_norm, cv2.COLORMAP_JET)
        
        # Display windows
        cv2.imshow('Original Image', img)
        cv2.imshow('Shadow Overlay (Red = Shadows)', overlay)
        cv2.imshow('Shadow Score (Hotter = More Likely Shadow)', score_colored)
        cv2.imshow('Final Shadow Mask (White = Shadow)', shadow_mask)
        
        print("✓ Shadow detection complete!")
        print("Press any key to close windows...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"✗ Error processing {img_path}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with your image
    # Run with default sensitivity (0.5)
    # Increase sensitivity (0.7-0.8) to detect more shadows
    # Decrease sensitivity (0.3-0.4) to be more conservative
    test_shadow_detection("data/images/6.jpg", sensitivity=0.5)