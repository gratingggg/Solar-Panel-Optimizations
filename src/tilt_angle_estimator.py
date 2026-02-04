"""
Tilt Angle Estimator using Computer Vision
Estimates solar panel tilt angle from uploaded images
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import CV_CONFIG


class TiltAngleEstimator:
    """Estimate solar panel tilt angle from images"""
    
    def __init__(self):
        self.canny_threshold1 = CV_CONFIG['canny_threshold1']
        self.canny_threshold2 = CV_CONFIG['canny_threshold2']
        self.min_angle = CV_CONFIG['min_angle']
        self.max_angle = CV_CONFIG['max_angle']
    
    def estimate_tilt_from_image(self, image_path: str) -> Tuple[float, float]:
        """
        Estimate tilt angle from solar panel image
        
        Args:
            image_path: Path to panel image
            
        Returns:
            Tuple of (estimated_angle, confidence_score)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not read image from {image_path}")
                return self._get_default_angle(), 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("No contours found in image")
                return self._get_default_angle(), 0.0
            
            # Find the largest contour (assumed to be the panel)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit a rectangle to the contour
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Adjust angle to be between 0 and 90
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Ensure angle is positive and within valid range
            angle = abs(angle)
            angle = np.clip(angle, self.min_angle, self.max_angle)
            
            # Calculate confidence based on contour area relative to image size
            image_area = gray.shape[0] * gray.shape[1]
            contour_area = cv2.contourArea(largest_contour)
            confidence = min(contour_area / image_area * 2, 1.0)  # Scale to 0-1
            
            print(f"Estimated tilt angle: {angle:.1f}° (confidence: {confidence:.2f})")
            
            return float(angle), float(confidence)
            
        except Exception as e:
            print(f"Error estimating tilt angle: {e}")
            return self._get_default_angle(), 0.0
    
    def _get_default_angle(self, latitude: Optional[float] = None) -> float:
        """
        Get default tilt angle based on latitude
        
        Args:
            latitude: Geographic latitude (optional)
            
        Returns:
            Default tilt angle in degrees
        """
        if latitude is not None and CV_CONFIG['default_angle_latitude_based']:
            # Rule of thumb: optimal tilt ≈ latitude
            return abs(latitude)
        else:
            # Default to 30 degrees if no latitude provided
            return 30.0
    
    def detect_panel_edges(self, image_path: str) -> Optional[np.ndarray]:
        """
        Detect edges in panel image (for visualization)
        
        Args:
            image_path: Path to panel image
            
        Returns:
            Edge image or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            
            return edges
            
        except Exception as e:
            print(f"Error detecting edges: {e}")
            return None
    
    def visualize_detection(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """
        Visualize the tilt angle detection
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Get edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Draw largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)
                
                # Draw fitted rectangle
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                
                # Add angle text
                angle, confidence = self.estimate_tilt_from_image(image_path)
                text = f"Angle: {angle:.1f}° (Conf: {confidence:.2f})"
                cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
            
            # Save or display
            if output_path:
                cv2.imwrite(output_path, image)
                print(f"Visualization saved to {output_path}")
            else:
                cv2.imshow('Tilt Angle Detection', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return False


if __name__ == "__main__":
    # Test the estimator
    estimator = TiltAngleEstimator()
    
    # Test with default angle
    print("Testing default angle calculation...")
    default_angle = estimator._get_default_angle(latitude=23.0)
    print(f"Default angle for latitude 23°: {default_angle}°")
    
    # Note: To test with actual images, you would need sample panel images
    print("\nTo test with images, provide a solar panel image path:")
    print("angle, confidence = estimator.estimate_tilt_from_image('path/to/panel.jpg')")
