import cv2
import numpy as np
import torch
import concurrent.futures
import json
import os
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from ultralytics import YOLO

class ALPRSystem:
    def __init__(
        self,
        plate_detector_path: str,
        state_classifier_path: str,
        char_detector_path: str,
        char_classifier_path: str,
        vehicle_detector_path: str,
        vehicle_classifier_path: str,
        enable_state_detection: bool = True,
        enable_vehicle_detection: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Add confidence thresholds for all models
        plate_detector_confidence: float = 0.45,
        state_classifier_confidence: float = 0.45,
        char_detector_confidence: float = 0.40,
        char_classifier_confidence: float = 0.40,
        vehicle_detector_confidence: float = 0.45,
        vehicle_classifier_confidence: float = 0.45,
        # Add option to set a fixed aspect ratio for license plates
        plate_aspect_ratio: Optional[float] = None,
        # Add option for corner dilation
        corner_dilation_pixels: int = 5
    ):
        """
        Initialize the ALPR system with the required YOLOv8 models.
        
        Args:
            plate_detector_path: Path to the YOLOv8 keypoint detection model for license plates
            state_classifier_path: Path to the YOLOv8 classification model for license plate states
            char_detector_path: Path to the YOLOv8 detection model for characters
            char_classifier_path: Path to the YOLOv8 classification model for OCR
            vehicle_detector_path: Path to the YOLOv8 detection model for vehicles
            vehicle_classifier_path: Path to the YOLOv8 classification model for vehicle make/model
            enable_state_detection: Whether to enable state identification
            enable_vehicle_detection: Whether to enable vehicle make/model detection
            device: Device to run the models on (cuda or cpu)
            plate_detector_confidence: Confidence threshold for plate detection
            state_classifier_confidence: Confidence threshold for state classification
            char_detector_confidence: Confidence threshold for character detection
            char_classifier_confidence: Confidence threshold for character classification
            vehicle_detector_confidence: Confidence threshold for vehicle detection
            vehicle_classifier_confidence: Confidence threshold for vehicle classification
            plate_aspect_ratio: If set, forces the warped license plate to have this aspect ratio (width/height)
                                while keeping the height fixed
            corner_dilation_pixels: Number of pixels to dilate the license plate corners from
                                   the center to ensure full plate coverage
        """
        # Load all YOLOv8 models with their respective resolutions
        self.plate_detector_model = YOLO(plate_detector_path, task='pose')
        self.state_classifier_model = YOLO(state_classifier_path, task='classify') if enable_state_detection else None
        self.char_detector_model = YOLO(char_detector_path, task='detect')
        self.char_classifier_model = YOLO(char_classifier_path, task='classify')
        self.vehicle_detector_model = YOLO(vehicle_detector_path, task='detect') if enable_vehicle_detection else None
        self.vehicle_classifier_model = YOLO(vehicle_classifier_path, task='classify') if enable_vehicle_detection else None
        
        self.enable_state_detection = enable_state_detection
        self.enable_vehicle_detection = enable_vehicle_detection
        self.device = device
        
        # Store confidence thresholds
        self.plate_detector_confidence = plate_detector_confidence
        self.state_classifier_confidence = state_classifier_confidence
        self.char_detector_confidence = char_detector_confidence
        self.char_classifier_confidence = char_classifier_confidence
        self.vehicle_detector_confidence = vehicle_detector_confidence
        self.vehicle_classifier_confidence = vehicle_classifier_confidence
        
        # Store the plate aspect ratio (width/height)
        self.plate_aspect_ratio = plate_aspect_ratio
        
        # Store corner dilation amount
        self.corner_dilation_pixels = corner_dilation_pixels
        
        # Define model resolutions as per requirements
        self.plate_detector_resolution = (640, 640)
        self.state_classifier_resolution = (224, 224)
        self.char_detector_resolution = (160, 160)
        self.char_classifier_resolution = (32, 32)
        self.vehicle_detector_resolution = (640, 640)
        self.vehicle_classifier_resolution = (224, 224)
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration details of the ALPR system.
        
        Returns:
            Dictionary containing configuration details
        """
        return {
            "device": self.device,
            "enable_state_detection": self.enable_state_detection,
            "enable_vehicle_detection": self.enable_vehicle_detection,
            "confidence_thresholds": {
                "plate_detector": self.plate_detector_confidence,
                "state_classifier": self.state_classifier_confidence,
                "char_detector": self.char_detector_confidence,
                "char_classifier": self.char_classifier_confidence,
                "vehicle_detector": self.vehicle_detector_confidence,
                "vehicle_classifier": self.vehicle_classifier_confidence
            },
            "resolutions": {
                "plate_detector": self.plate_detector_resolution,
                "state_classifier": self.state_classifier_resolution,
                "char_detector": self.char_detector_resolution,
                "char_classifier": self.char_classifier_resolution,
                "vehicle_detector": self.vehicle_detector_resolution,
                "vehicle_classifier": self.vehicle_classifier_resolution
            },
            "models_loaded": {
                "plate_detector": self.plate_detector_model is not None,
                "state_classifier": self.state_classifier_model is not None,
                "char_detector": self.char_detector_model is not None,
                "char_classifier": self.char_classifier_model is not None,
                "vehicle_detector": self.vehicle_detector_model is not None,
                "vehicle_classifier": self.vehicle_classifier_model is not None
            },
            "plate_aspect_ratio": self.plate_aspect_ratio,
            "corner_dilation_pixels": self.corner_dilation_pixels
        }
    
    def dilate_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Dilate the license plate corners by moving them outward from the centroid.
        
        Args:
            corners: Numpy array of shape (4, 2) containing the corner coordinates
            
        Returns:
            Dilated corners as a numpy array of the same shape
        """
        # Calculate the centroid
        centroid = np.mean(corners, axis=0)
        
        # Create a copy of the corners that we will modify
        dilated_corners = corners.copy()
        
        # For each corner, move it away from the centroid
        for i in range(len(corners)):
            # Vector from centroid to corner
            vector = corners[i] - centroid
            
            # Normalize the vector
            vector_length = np.sqrt(np.sum(vector**2))
            if vector_length > 0:  # Avoid division by zero
                unit_vector = vector / vector_length
                
                # Extend the corner by the dilation amount in the direction of the unit vector
                dilated_corners[i] = corners[i] + unit_vector * self.corner_dilation_pixels
        
        return dilated_corners
    
    def detect_license_plates(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect license plates (both day and night) in the image using keypoint detection model.
        Returns dictionary with 'day_plates' and 'night_plates' lists containing plate corner coordinates.
        """
        # Resize image for plate detector
        img_resized = cv2.resize(image, self.plate_detector_resolution)
        
        try:
            # Run YOLOv8 keypoint detection model to detect plate corners
            # Add confidence threshold parameter
            results = self.plate_detector_model(img_resized, conf=self.plate_detector_confidence, verbose=False)[0]
        except Exception as e:
            print(f"Error running plate detector model: {e}")
            # Return empty results to avoid breaking the pipeline
            return {"day_plates": [], "night_plates": []}
        
        # Process the results to extract plate corners
        day_plates = []
        night_plates = []
        
        # The model returns keypoints for each detected plate (4 corners)
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            for i, keypoints in enumerate(results.keypoints.data):
                if len(keypoints) >= 4:  # Ensure we have at least 4 keypoints (corners)
                    # Get the 4 corner points
                    corners = keypoints[:4].cpu().numpy()  # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    
                    # Scale the keypoints back to original image size
                    h, w = image.shape[:2]
                    scale_x = w / self.plate_detector_resolution[0]
                    scale_y = h / self.plate_detector_resolution[1]
                    
                    scaled_corners = []
                    for corner in corners:
                        # Handle different possible formats of keypoint data
                        try:
                            if len(corner) >= 3:  # Format may include confidence value or other data
                                x, y = corner[0], corner[1]
                            else:
                                x, y = corner
                            
                            scaled_corners.append([float(x * scale_x), float(y * scale_y)])
                        except Exception as e:
                            print(f"Error processing keypoint: {corner}, Error: {e}")
                            # Use a default value to avoid breaking the pipeline
                            scaled_corners.append([0.0, 0.0])
                    
                    # Convert to numpy array for dilation
                    scaled_corners_np = np.array(scaled_corners, dtype=np.float32)
                    
                    # Apply dilation to the corners
                    dilated_corners_np = self.dilate_corners(scaled_corners_np)
                    
                    # Convert back to list format
                    dilated_corners = dilated_corners_np.tolist()
                    
                    # Store both original and dilated corners for visualization
                    original_corners = scaled_corners.copy()
                    
                    # Get the detection box if available
                    detection_box = None
                    if hasattr(results.boxes, 'xyxy') and i < len(results.boxes.xyxy):
                        box = results.boxes.xyxy[i].cpu().numpy()
                        # Scale box coordinates
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x)-15
                        y1 = int(y1 * scale_y)-15
                        x2 = int(x2 * scale_x)+15
                        y2 = int(y2 * scale_y)+15
                        detection_box = [x1, y1, x2, y2]  # [x1, y1, x2, y2] format
                    
                    # Determine if it's a day plate or night plate based on the class
                    # Assuming class 0 is day plate and class 1 is night plate
                    if hasattr(results.boxes, 'cls') and i < len(results.boxes.cls):
                        plate_class = int(results.boxes.cls[i].item())
                        
                        plate_info = {
                            "corners": dilated_corners,  # Use dilated corners for processing
                            "original_corners": original_corners,  # Keep original corners for reference
                            "detection_box": detection_box,
                            "confidence": float(results.boxes.conf[i].item()) if hasattr(results.boxes, 'conf') else 0.0
                        }
                        
                        if plate_class == 0:  # Day plate
                            day_plates.append(plate_info)
                        else:  # Night plate
                            night_plates.append(plate_info)
        
        return {
            "day_plates": day_plates,
            "night_plates": night_plates
        }
    
    def four_point_transform(self, image: np.ndarray, corners: List[List[float]]) -> np.ndarray:
        """
        Apply a 4-point perspective transform to extract the license plate.
        If plate_aspect_ratio is set, the output will have that aspect ratio with fixed height.
        
        Args:
            image: Original image
            corners: List of 4 corner points [x, y]
            
        Returns:
            Warped image of the license plate
        """
        # Convert corners to numpy array
        try:
            corners = np.array(corners, dtype=np.float32)
            # Ensure we have exactly 4 points
            if corners.shape[0] != 4:
                print(f"Warning: Expected 4 corners but got {corners.shape[0]}. Adjusting...")
                if corners.shape[0] > 4:
                    corners = corners[:4]  # Take only first 4 points
                else:
                    # Not enough points, pad with zeros
                    padded_corners = np.zeros((4, 2), dtype=np.float32)
                    padded_corners[:corners.shape[0]] = corners
                    corners = padded_corners
        except Exception as e:
            print(f"Error converting corners to numpy array: {e}")
            # Create a fallback rectangle
            h, w = image.shape[:2]
            corners = np.array([
                [0, 0],
                [w-1, 0],
                [w-1, h-1],
                [0, h-1]
            ], dtype=np.float32)
        
        # Get the width and height of the transformed image
        # We'll sort the points to ensure consistent ordering: top-left, top-right, bottom-right, bottom-left
        rect = self.order_points(corners)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))
        
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))
        
        # Create output dimensions
        output_width = max_width
        output_height = max_height
        
        # Apply aspect ratio if specified (width/height)
        if self.plate_aspect_ratio is not None:
            # Keep height fixed and calculate width based on the desired aspect ratio
            output_width = int(output_height * self.plate_aspect_ratio)
        
        # Ensure dimensions are at least 1 pixel
        output_width = max(1, output_width)
        output_height = max(1, output_height)
        
        # Construct the set of destination points for the transform
        dst = np.array([
            [0, 0],                           # top-left
            [output_width - 1, 0],            # top-right
            [output_width - 1, output_height - 1],  # bottom-right
            [0, output_height - 1]            # bottom-left
        ], dtype=np.float32)
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (output_width, output_height))
        
        return warped
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the sequence: top-left, top-right, bottom-right, bottom-left
        """
        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # The top-left point will have the smallest sum
        # The bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Now compute the difference between the points
        # The top-right point will have the smallest difference
        # The bottom-left point will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def classify_state(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify the state of the license plate.
        """
        if not self.enable_state_detection or self.state_classifier_model is None:
            return "Unknown", 0.0
        
        # Resize plate image for state classifier
        plate_resized = cv2.resize(plate_image, self.state_classifier_resolution)
        
        # Run state classification model with confidence threshold
        results = self.state_classifier_model(plate_resized, conf=self.state_classifier_confidence, verbose=False)[0]
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            state_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to state name
            state_names = self.state_classifier_model.names
            state_name = state_names[state_idx]
            
            return state_name, confidence
        
        return "Unknown", 0.0
    
    def detect_characters(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters in the license plate image.
        """
        # Resize plate image for character detector
        plate_resized = cv2.resize(plate_image, self.char_detector_resolution)
        
        # Run character detection model with the configurable confidence threshold
        results = self.char_detector_model.predict(plate_resized, conf=self.char_detector_confidence, verbose=False)[0]
        
        # Process the results to extract character bounding boxes
        characters = []
        
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                
                # Scale the coordinates back to the original plate image size
                h, w = plate_image.shape[:2]
                scale_x = w / self.char_detector_resolution[0]
                scale_y = h / self.char_detector_resolution[1]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Ensure the box coordinates are within the image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2:
                    continue
                
                confidence = float(results.boxes.conf[i].item()) if hasattr(results.boxes, 'conf') else 0.0
                
                # Extract the character region
                char_img = plate_image[y1:y2, x1:x2]
                
                characters.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "image": char_img
                })
        
        return characters
    
    def organize_characters(self, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organize characters into a coherent structure, handling multiple lines and vertical characters.
        Returns a list of characters in reading order.
        """
        if not characters:
            return []
        
        # Extract bounding box coordinates
        boxes = np.array([[c["box"][0], c["box"][1], c["box"][2], c["box"][3]] for c in characters])
        
        # Calculate center points of all boxes
        centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])
        
        # Calculate heights and widths
        heights = boxes[:, 3] - boxes[:, 1]
        widths = boxes[:, 2] - boxes[:, 0]
        
        # Determine if there are multiple lines
        # We'll use a simple heuristic: if there are centers with y-coordinates that differ by more than
        # the average character height, then we have multiple lines
        avg_height = np.mean(heights)
        y_diffs = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1])
        multiple_lines = np.any(y_diffs > 1.5 * avg_height)
        
        # Determine if there are vertical characters
        # Vertical characters typically have height > width
        aspect_ratios = heights / widths
        vertical_chars = aspect_ratios > 1.5  # Characters with aspect ratio > 1.5 are considered vertical
        
        organized_chars = []
        
        if multiple_lines:
            # Cluster characters by y-coordinate (row)
            # Using a simple approach: characters with similar y-center are on the same line
            y_centers = centers[:, 1]
            sorted_indices = np.argsort(y_centers)
            
            # Group characters by line
            lines = []
            current_line = [sorted_indices[0]]
            
            for i in range(1, len(sorted_indices)):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                
                if abs(y_centers[idx] - y_centers[prev_idx]) > 0.5 * avg_height:
                    # Start a new line
                    lines.append(current_line)
                    current_line = [idx]
                else:
                    # Continue the current line
                    current_line.append(idx)
            
            lines.append(current_line)
            
            # Sort characters within each line by x-coordinate (left to right)
            for line in lines:
                line_chars = [characters[idx] for idx in sorted(line, key=lambda idx: centers[idx][0])]
                organized_chars.extend(line_chars)
        else:
            # Single line with possible vertical characters at the beginning or end
            
            # Check for vertical characters at the beginning
            start_vertical_indices = []
            for i, is_vertical in enumerate(vertical_chars):
                if is_vertical and centers[i][0] < np.median(centers[:, 0]):
                    start_vertical_indices.append(i)
            
            # Check for vertical characters at the end
            end_vertical_indices = []
            for i, is_vertical in enumerate(vertical_chars):
                if is_vertical and centers[i][0] > np.median(centers[:, 0]):
                    end_vertical_indices.append(i)
            
            # Remaining horizontal characters
            horizontal_indices = [i for i in range(len(characters)) 
                                if i not in start_vertical_indices and i not in end_vertical_indices]
            
            # Sort vertical characters at the beginning by x-coordinate
            start_vertical_indices.sort(key=lambda idx: centers[idx][0])
            
            # Sort horizontal characters by x-coordinate
            horizontal_indices.sort(key=lambda idx: centers[idx][0])
            
            # Sort vertical characters at the end by x-coordinate
            end_vertical_indices.sort(key=lambda idx: centers[idx][0])
            
            # Combine all indices in the correct order
            all_indices = start_vertical_indices + horizontal_indices + end_vertical_indices
            organized_chars = [characters[idx] for idx in all_indices]
        
        return organized_chars
    
    def classify_character(self, char_image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Classify a character using OCR and return top 5 predictions.
        
        Args:
            char_image: The character image to classify
            
        Returns:
            List of tuples containing (character, confidence) for top 5 predictions
        """
        if char_image.size == 0:
            return [("?", 0.0)]
        
        # Resize character image for classifier
        char_resized = cv2.resize(char_image, self.char_classifier_resolution)
        
        # Run character classification model with configurable confidence threshold
        results = self.char_classifier_model.predict(char_resized, conf=self.char_classifier_confidence, verbose=False)[0]
        
        top_predictions = []
        
        # Extract top5 predictions if available
        if hasattr(results, 'probs'):
            probs = results.probs
            
            # Try to access probability data
            if hasattr(probs, 'data'):
                try:
                    # Convert to tensor if it's not already
                    probs_tensor = probs.data
                    if not isinstance(probs_tensor, torch.Tensor):
                        probs_tensor = torch.tensor(probs_tensor)
                    
                    # Get top 5 predictions
                    values, indices = torch.topk(probs_tensor, min(5, len(probs_tensor)))
                    
                    # Convert to list of (char, confidence) tuples
                    char_names = self.char_classifier_model.names
                    for i in range(len(values)):
                        idx = int(indices[i].item())
                        conf = float(values[i].item())
                        
                        # Only include predictions with confidence > 0.02
                        if conf >= 0.02:
                            character = char_names[idx]
                            top_predictions.append((character, conf))
                except Exception as e:
                    print(f"Error getting top5 predictions: {e}")
                    
                    # Fallback to top1 if available
                    if hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                        idx = int(probs.top1)
                        conf = float(probs.top1conf.item())
                        
                        char_names = self.char_classifier_model.names
                        character = char_names[idx]
                        top_predictions.append((character, conf))
            elif hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                # Fallback to top1 if data attribute not available
                idx = int(probs.top1)
                conf = float(probs.top1conf.item())
                
                char_names = self.char_classifier_model.names
                character = char_names[idx]
                top_predictions.append((character, conf))
        
        # If no predictions were found or all had low confidence
        if not top_predictions:
            top_predictions.append(("?", 0.0))
        
        return top_predictions
    
    def _generate_top_plates(self, char_results: List[Dict[str, Any]], max_combinations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple possible license plate combinations using top character predictions.
        
        Args:
            char_results: List of character results with top_predictions
            max_combinations: Maximum number of combinations to return
            
        Returns:
            List of alternative plate combinations with plate number and confidence
        """
        if not char_results:
            return []
        
        # Identify positions with uncertain character predictions
        uncertain_positions = []
        for i, char_result in enumerate(char_results):
            top_preds = char_result.get("top_predictions", [])
            
            # If we have at least 2 predictions with good confidence
            if len(top_preds) >= 2 and top_preds[1][1] >= 0.02:
                confidence_diff = top_preds[0][1] - top_preds[1][1]
                uncertain_positions.append((i, confidence_diff))
        
        # Sort by smallest confidence difference (most uncertain first)
        uncertain_positions.sort(key=lambda x: x[1])
        
        # Create base plate using top1 predictions
        base_plate = ''.join(cr["char"] for cr in char_results)
        base_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
        
        # Start with the base plate
        combinations = [{"plate": base_plate, "confidence": base_confidence}]
        
        # Generate alternative plates by substituting at uncertain positions
        for pos_idx, _ in uncertain_positions[:min(3, len(uncertain_positions))]:
            char_result = char_results[pos_idx]
            top_preds = char_result.get("top_predictions", [])[1:3]  # Use 2nd and 3rd predictions
            
            # Generate new plates by substituting at this position
            new_combinations = []
            for existing in combinations:
                for alt_char, alt_conf in top_preds:
                    if alt_conf >= 0.02:
                        plate_chars = list(existing["plate"])
                        if pos_idx < len(plate_chars):
                            # Calculate new confidence
                            old_char_conf = char_results[pos_idx]["confidence"]
                            plate_chars[pos_idx] = alt_char
                            
                            # Adjust confidence by replacing the character's contribution
                            char_count = len(char_results)
                            new_conf = existing["confidence"] - (old_char_conf / char_count) + (alt_conf / char_count)
                            
                            new_plate = ''.join(plate_chars)
                            new_combinations.append({"plate": new_plate, "confidence": new_conf})
                
            combinations.extend(new_combinations)
            
            # If we have enough combinations, stop
            if len(combinations) >= max_combinations:
                break
        
        # Sort by confidence and take top N
        combinations.sort(key=lambda x: x["confidence"], reverse=True)
        return combinations[:max_combinations]
    
    def detect_vehicle(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the image.
        """
        if not self.enable_vehicle_detection or self.vehicle_detector_model is None:
            return []
        
        # Resize image for vehicle detector
        img_resized = cv2.resize(image, self.vehicle_detector_resolution)
        
        # Run vehicle detection model with configurable confidence threshold
        results = self.vehicle_detector_model(img_resized, conf=self.vehicle_detector_confidence, verbose=False)[0]
        
        # Process the results to extract vehicle bounding boxes
        vehicles = []
        
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                
                # Scale the coordinates back to the original image size
                h, w = image.shape[:2]
                scale_x = w / self.vehicle_detector_resolution[0]
                scale_y = h / self.vehicle_detector_resolution[1]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Ensure the box coordinates are within the image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2:
                    continue
                
                confidence = float(results.boxes.conf[i].item()) if hasattr(results.boxes, 'conf') else 0.0
                
                # Extract the vehicle region
                vehicle_img = image[y1:y2, x1:x2]
                
                vehicles.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "image": vehicle_img
                })
        
        return vehicles
    
    def classify_vehicle(self, vehicle_image: np.ndarray) -> Tuple[str, str, float]:
        """
        Classify vehicle make and model.
        """
        if not self.enable_vehicle_detection or self.vehicle_classifier_model is None or vehicle_image.size == 0:
            return "Unknown", "Unknown", 0.0
        
        # Resize vehicle image for classifier
        vehicle_resized = cv2.resize(vehicle_image, self.vehicle_classifier_resolution)
        
        # Run vehicle classification model with configurable confidence threshold
        results = self.vehicle_classifier_model(vehicle_resized, conf=self.vehicle_classifier_confidence, verbose=False)[0]
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            vehicle_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to make and model
            vehicle_names = self.vehicle_classifier_model.names
            make_model = vehicle_names[vehicle_idx]
            
            # Split make and model (assuming format "Make_Model")
            make, model = make_model.split("_", 1) if "_" in make_model else (make_model, "Unknown")
            
            return make, model, confidence
        
        return "Unknown", "Unknown", 0.0
    
    def process_plate(self, image: np.ndarray, plate_info: Dict[str, Any], is_day_plate: bool) -> Dict[str, Any]:
        """
        Process a single license plate.
        """
        try:
            # Extract corners
            plate_corners = plate_info["corners"]
            
            # Crop the license plate using 4-point transform
            plate_image = self.four_point_transform(image, plate_corners)
        except Exception as e:
            print(f"Error in four_point_transform: {e}")
            # Create a blank image as fallback
            plate_image = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Initialize result dictionary
        plate_result = {
            "corners": plate_corners,
            "is_day_plate": is_day_plate,
            "characters": [],
            "plate": "",
            "confidence": 0.0,
            "aspect_ratio": self.plate_aspect_ratio  # Store used aspect ratio for reference
        }
        
        # Include original corners if available
        if "original_corners" in plate_info:
            plate_result["original_corners"] = plate_info["original_corners"]
        
        # Include detection box if available
        if "detection_box" in plate_info:
            plate_result["detection_box"] = plate_info["detection_box"]
        
        # If it's a day plate, also determine the state
        if is_day_plate and self.enable_state_detection:
            state, state_confidence = self.classify_state(plate_image)
            plate_result["state"] = state
            plate_result["state_confidence"] = state_confidence
        
        # Detect characters in the plate
        characters = self.detect_characters(plate_image)
        
        # Organize characters (handle multiple lines and vertical characters)
        organized_chars = self.organize_characters(characters)
        
        # Classify each character
        char_results = []
        for char_info in organized_chars:
            top_chars = self.classify_character(char_info["image"])
            char_results.append({
                "char": top_chars[0][0] if top_chars else "?",  # Still use the top prediction as the main character
                "confidence": top_chars[0][1] if top_chars else 0.0,
                "top_predictions": top_chars,  # Store all top predictions
                "box": char_info["box"]
            })
            
        # Construct the license number by concatenating the characters
        license_number = ''.join(cr["char"] for cr in char_results)
        avg_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
        
        # Generate alternative plate combinations
        top_plates = self._generate_top_plates(char_results)
        
        plate_result["characters"] = char_results
        plate_result["plate"] = license_number
        plate_result["license_number"] = license_number
        plate_result["confidence"] = avg_confidence
        plate_result["top_plates"] = top_plates  # Add alternative plate combinations
        
        # Store plate dimensions for debugging
        if plate_image is not None:
            h, w = plate_image.shape[:2]
            plate_result["dimensions"] = {"width": w, "height": h, "actual_ratio": w/h if h > 0 else 0}
        
        return plate_result
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image to detect and recognize license plates, vehicle make/model.
        """
        # Create a copy of the image to avoid modifying the original
        image_copy = image.copy()
        
        # Detect license plates in the image
        plate_detection = self.detect_license_plates(image_copy)
        
        # Initialize results
        results = {
            "day_plates": [],
            "night_plates": [],
            "vehicles": []
        }
        
        # Process plates using multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit day plate processing tasks
            day_plate_futures = [
                executor.submit(self.process_plate, image_copy, plate, True)
                for plate in plate_detection["day_plates"]
            ]
            
            # Submit night plate processing tasks
            night_plate_futures = [
                executor.submit(self.process_plate, image_copy, plate, False)
                for plate in plate_detection["night_plates"]
            ]
            
            # Collect day plate results
            for future in concurrent.futures.as_completed(day_plate_futures):
                try:
                    results["day_plates"].append(future.result())
                except Exception as e:
                    print(f"Error processing day plate: {e}")
            
            # Collect night plate results
            for future in concurrent.futures.as_completed(night_plate_futures):
                try:
                    results["night_plates"].append(future.result())
                except Exception as e:
                    print(f"Error processing night plate: {e}")
        
        # If day plates were detected and vehicle detection is enabled,
        # detect and classify vehicles
        if results["day_plates"] and self.enable_vehicle_detection:
            vehicles = self.detect_vehicle(image_copy)
            
            vehicle_results = []
            for vehicle_info in vehicles:
                try:
                    make, model, confidence = self.classify_vehicle(vehicle_info["image"])
                    vehicle_results.append({
                        "box": vehicle_info["box"],
                        "make": make,
                        "model": model,
                        "confidence": confidence
                    })
                except Exception as e:
                    print(f"Error classifying vehicle: {e}")
            
            results["vehicles"] = vehicle_results
        
        # Remove image data from the results before JSON serialization
        for plate_type in ["day_plates", "night_plates"]:
            for plate in results[plate_type]:
                for char in plate.get("characters", []):
                    if "image" in char:
                        del char["image"]
        
        for vehicle in results.get("vehicles", []):
            if "image" in vehicle:
                del vehicle["image"]
        
        return results

def process_alpr(
    image_path: str,
    plate_detector_path: str,
    state_classifier_path: str,
    char_detector_path: str,
    char_classifier_path: str,
    vehicle_detector_path: str,
    vehicle_classifier_path: str,
    enable_state_detection: bool = True,
    enable_vehicle_detection: bool = True,
    output_path: str = None,
    visualization_dir: str = None,
    # Add confidence parameters with default values to the function
    plate_detector_confidence: float = 0.45,
    state_classifier_confidence: float = 0.45,
    char_detector_confidence: float = 0.40,
    char_classifier_confidence: float = 0.40,
    vehicle_detector_confidence: float = 0.45,
    vehicle_classifier_confidence: float = 0.45,
    # Add option for plate aspect ratio
    plate_aspect_ratio: Optional[float] = None,
    # Add option for corner dilation
    corner_dilation_pixels: int = 5
):
    """
    Process an image through the ALPR system and return the results.
    
    Args:
        image_path: Path to the image to process
        plate_detector_path: Path to the YOLOv8 keypoint detection model for license plates
        state_classifier_path: Path to the YOLOv8 classification model for license plate states
        char_detector_path: Path to the YOLOv8 detection model for characters
        char_classifier_path: Path to the YOLOv8 classification model for OCR
        vehicle_detector_path: Path to the YOLOv8 detection model for vehicles
        vehicle_classifier_path: Path to the YOLOv8 classification model for vehicle make/model
        enable_state_detection: Whether to enable state identification
        enable_vehicle_detection: Whether to enable vehicle make/model detection
        output_path: Path to save the JSON results (optional)
        visualization_dir: Directory to save visualization images (optional)
        plate_detector_confidence: Confidence threshold for plate detection
        state_classifier_confidence: Confidence threshold for state classification
        char_detector_confidence: Confidence threshold for character detection
        char_classifier_confidence: Confidence threshold for character classification
        vehicle_detector_confidence: Confidence threshold for vehicle detection
        vehicle_classifier_confidence: Confidence threshold for vehicle classification
        plate_aspect_ratio: If set, forces the warped license plate to have this aspect ratio (width/height)
                            while keeping the height fixed
        corner_dilation_pixels: Number of pixels to dilate the license plate corners from
                                the center to ensure full plate coverage
    
    Returns:
        Dictionary containing the ALPR results
    """
    try:
        # Initialize ALPR system with configurable confidence thresholds
        alpr = ALPRSystem(
            plate_detector_path=plate_detector_path,
            state_classifier_path=state_classifier_path,
            char_detector_path=char_detector_path,
            char_classifier_path=char_classifier_path,
            vehicle_detector_path=vehicle_detector_path,
            vehicle_classifier_path=vehicle_classifier_path,
            enable_state_detection=enable_state_detection,
            enable_vehicle_detection=enable_vehicle_detection,
            plate_detector_confidence=plate_detector_confidence,
            state_classifier_confidence=state_classifier_confidence,
            char_detector_confidence=char_detector_confidence,
            char_classifier_confidence=char_classifier_confidence,
            vehicle_detector_confidence=vehicle_detector_confidence,
            vehicle_classifier_confidence=vehicle_classifier_confidence,
            plate_aspect_ratio=plate_aspect_ratio,
            corner_dilation_pixels=corner_dilation_pixels
        )
    except Exception as e:
        print(f"Error initializing ALPR system: {e}")
        return None
    
    try:
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
    
    try:
        # Process the image
        results = alpr.process_image(image)
        
        # Convert results to JSON format
        json_results = json.dumps(results, indent=4)
        
        # Generate visualizations if requested
        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)
            
            # Save individual model visualizations
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # 1. License plate detection
            plate_det_path = os.path.join(visualization_dir, f"{base_filename}_plate_detection.jpg")
            plates = {"day_plates": [], "night_plates": []}
            for plate_type in ["day_plates", "night_plates"]:
                plates[plate_type] = [
                    {
                        "corners": plate["corners"],
                        "original_corners": plate.get("original_corners"),
                        "detection_box": plate.get("detection_box"),
                        "confidence": plate.get("confidence", 0.0),
                        "dimensions": plate.get("dimensions", {})
                    } 
                    for plate in results.get(plate_type, [])
                ]
            visualize_plate_detection(image, plates, plate_det_path)
            print(f"Saved plate detection visualization to {plate_det_path}")
            
            # 2. For each detected plate, save individual model visualizations
            for plate_type in ["day_plates", "night_plates"]:
                for i, plate in enumerate(results.get(plate_type, [])):
                    try:
                        # Extract the plate image
                        plate_image = alpr.four_point_transform(image, plate["corners"])
                        
                        # Character detection
                        if plate.get("characters", []):
                            char_det_path = os.path.join(visualization_dir, 
                                                      f"{base_filename}_{plate_type[:-1]}_{i+1}_char_detection.jpg")
                            visualize_character_detection(plate_image, plate["characters"], char_det_path)
                            print(f"Saved character detection visualization to {char_det_path}")
                        
                        # State classification (only for day plates)
                        if plate_type == "day_plates" and "state" in plate:
                            state_cls_path = os.path.join(visualization_dir, 
                                                       f"{base_filename}_{plate_type[:-1]}_{i+1}_state_classification.jpg")
                            visualize_state_classification(plate_image, plate["state"], 
                                                        plate["state_confidence"], state_cls_path)
                            print(f"Saved state classification visualization to {state_cls_path}")
                    except Exception as e:
                        print(f"Error creating plate visualization: {e}")
            
            # 3. Vehicle detection
            if results.get("vehicles", []):
                vehicle_det_path = os.path.join(visualization_dir, f"{base_filename}_vehicle_detection.jpg")
                visualize_vehicle_detection(image, results["vehicles"], vehicle_det_path)
                print(f"Saved vehicle detection visualization to {vehicle_det_path}")
            
            # 4. Combined results visualization
            combined_path = os.path.join(visualization_dir, f"{base_filename}_complete_results.jpg")
            visualize_results(image_path, results, combined_path)
            print(f"Saved combined results visualization to {combined_path}")
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
    # Save results to file if specified
    if output_path:
        with open(output_path, "w") as f:
            f.write(json_results)
    
    return results

# Visualization functions for each model
def visualize_plate_detection(image: np.ndarray, plates: Dict[str, List[Dict[str, Any]]], save_path: str = None) -> np.ndarray:
    """
    Create a visualization of license plate detection results.
    
    Args:
        image: Original image as numpy array
        plates: Dictionary with 'day_plates' and 'night_plates' lists
        save_path: Path to save the visualization image (optional)
    
    Returns:
        Visualization image as numpy array
    """
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Colors for visualization (BGR format)
    day_plate_color = (0, 255, 0)     # Green
    night_plate_color = (0, 165, 255) # Orange
    corner_point_color = (0, 0, 255)  # Red
    detection_box_color = (255, 255, 0) # Cyan
    original_corner_color = (255, 0, 255)  # Magenta
    text_color = (255, 255, 255)      # White
    
    # Draw day plates
    for i, plate in enumerate(plates.get('day_plates', [])):
        # Draw detection box if available
        if "detection_box" in plate and plate["detection_box"] is not None:
            x1, y1, x2, y2 = [int(coord) for coord in plate["detection_box"]]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), detection_box_color, 2)
        
        # Draw original corners if available (before dilation)
        if "original_corners" in plate and plate["original_corners"] is not None:
            orig_corners = np.array(plate['original_corners'], dtype=np.int32)
            cv2.polylines(vis_image, [orig_corners.reshape((-1, 1, 2))], True, original_corner_color, 1)
            
            # Draw magenta dots at each original corner point
            for corner in orig_corners:
                x, y = corner
                cv2.circle(vis_image, (int(x), int(y)), 3, original_corner_color, -1)
            
        # Draw plate corners (dilated) as a polygon
        corners = np.array(plate['corners'], dtype=np.int32)
        cv2.polylines(vis_image, [corners.reshape((-1, 1, 2))], True, day_plate_color, 2)
        
        # Draw red dots at each dilated corner point
        for corner in corners:
            x, y = corner
            cv2.circle(vis_image, (int(x), int(y)), 5, corner_point_color, -1)  # -1 thickness means filled circle
        
        # Add plate confidence and dimensions
        x, y = corners[0]
        y = max(y - 10, 10)  # Ensure text is visible
        plate_text = f"Day Plate {i+1} ({plate['confidence']:.2f})"
        cv2.putText(vis_image, plate_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add dimensions info if available
        if "dimensions" in plate:
            dims = plate["dimensions"]
            if "width" in dims and "height" in dims and "actual_ratio" in dims:
                dim_text = f"W:{dims['width']} H:{dims['height']} Ratio:{dims['actual_ratio']:.2f}"
                cv2.putText(vis_image, dim_text, (int(x), int(y) + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Draw night plates
    for i, plate in enumerate(plates.get('night_plates', [])):
        # Draw detection box if available
        if "detection_box" in plate and plate["detection_box"] is not None:
            x1, y1, x2, y2 = [int(coord) for coord in plate["detection_box"]]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), detection_box_color, 2)
            
        # Draw original corners if available (before dilation)
        if "original_corners" in plate and plate["original_corners"] is not None:
            orig_corners = np.array(plate['original_corners'], dtype=np.int32)
            cv2.polylines(vis_image, [orig_corners.reshape((-1, 1, 2))], True, original_corner_color, 1)
            
            # Draw magenta dots at each original corner point
            for corner in orig_corners:
                x, y = corner
                cv2.circle(vis_image, (int(x), int(y)), 3, original_corner_color, -1)
        
        # Draw plate corners (dilated) as a polygon
        corners = np.array(plate['corners'], dtype=np.int32)
        cv2.polylines(vis_image, [corners.reshape((-1, 1, 2))], True, night_plate_color, 2)
        
        # Draw red dots at each dilated corner point
        for corner in corners:
            x, y = corner
            cv2.circle(vis_image, (int(x), int(y)), 5, corner_point_color, -1)  # -1 thickness means filled circle
        
        # Add plate confidence and dimensions
        x, y = corners[0]
        y = max(y - 10, 10)  # Ensure text is visible
        plate_text = f"Night Plate {i+1} ({plate['confidence']:.2f})"
        cv2.putText(vis_image, plate_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add dimensions info if available
        if "dimensions" in plate:
            dims = plate["dimensions"]
            if "width" in dims and "height" in dims and "actual_ratio" in dims:
                dim_text = f"W:{dims['width']} H:{dims['height']} Ratio:{dims['actual_ratio']:.2f}"
                cv2.putText(vis_image, dim_text, (int(x), int(y) + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Add title
    cv2.putText(vis_image, "License Plate Detection", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Add legend
    legend_y = 60
    cv2.putText(vis_image, "Legend:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(vis_image, "Day Plate Outline", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, day_plate_color, 2)
    cv2.putText(vis_image, "Night Plate Outline", (350, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, night_plate_color, 2)
    
    legend_y += 30
    cv2.circle(vis_image, (140, legend_y-5), 5, corner_point_color, -1)
    cv2.putText(vis_image, "Dilated Corners", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, corner_point_color, 2)
    cv2.putText(vis_image, "Detection Box", (350, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_box_color, 2)
    
    legend_y += 30
    cv2.circle(vis_image, (140, legend_y-5), 3, original_corner_color, -1)
    cv2.putText(vis_image, "Original Corners", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, original_corner_color, 2)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image

def visualize_state_classification(plate_image: np.ndarray, state: str, confidence: float, save_path: str = None) -> np.ndarray:
    """
    Create a visualization of state classification results.
    
    Args:
        plate_image: License plate image as numpy array
        state: Predicted state name
        confidence: Confidence score
        save_path: Path to save the visualization image (optional)
    
    Returns:
        Visualization image as numpy array
    """
    # Resize plate image for better visualization if too small
    h, w = plate_image.shape[:2]
    if h < 100 or w < 200:
        scale = max(200 / w, 100 / h)
        plate_image = cv2.resize(plate_image, (int(w * scale), int(h * scale)))
    
    # Create a copy for visualization
    vis_image = plate_image.copy()
    
    # Add state classification information
    cv2.putText(vis_image, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis_image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add dimensions information
    h, w = plate_image.shape[:2]
    cv2.putText(vis_image, f"Dimensions: {w}x{h} (Ratio: {w/h:.2f})", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add title
    cv2.putText(vis_image, "State Classification", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image

def visualize_character_detection(plate_image: np.ndarray, characters: List[Dict[str, Any]], save_path: str = None) -> np.ndarray:
    """
    Create a visualization of character detection results.
    
    Args:
        plate_image: License plate image as numpy array
        characters: List of character dictionaries with 'box' and 'confidence'
        save_path: Path to save the visualization image (optional)
    
    Returns:
        Visualization image as numpy array
    """
    # Resize plate image for better visualization if too small
    h, w = plate_image.shape[:2]
    if h < 100 or w < 200:
        scale = max(200 / w, 100 / h)
        plate_image = cv2.resize(plate_image, (int(w * scale), int(h * scale)))
    
    # Create a copy for visualization
    vis_image = plate_image.copy()
    
    # Draw character boxes
    for i, char in enumerate(characters):
        x1, y1, x2, y2 = char['box']
        
        # If the plate was resized, scale the box coordinates
        if h < 100 or w < 200:
            scale = max(200 / w, 100 / h)
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw red dots at the corners of the box
        cv2.circle(vis_image, (x1, y1), 3, (0, 0, 255), -1)  # Top-left
        cv2.circle(vis_image, (x2, y1), 3, (0, 0, 255), -1)  # Top-right
        cv2.circle(vis_image, (x2, y2), 3, (0, 0, 255), -1)  # Bottom-right
        cv2.circle(vis_image, (x1, y2), 3, (0, 0, 255), -1)  # Bottom-left
        
        # Add character index and confidence
        cv2.putText(vis_image, f"{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    # Add plate dimensions
    h_orig, w_orig = plate_image.shape[:2]
    cv2.putText(vis_image, f"Plate: {w_orig}x{h_orig} (Ratio: {w_orig/h_orig:.2f})", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add title and count information
    cv2.putText(vis_image, "Character Detection", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_image, f"Found {len(characters)} characters", (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image

def visualize_character_classification(chars: List[Dict[str, Any]], save_path: str = None) -> np.ndarray:
    """
    Create a visualization of character classification results.
    
    Args:
        chars: List of character dictionaries with 'char', 'confidence', and 'image'
        save_path: Path to save the visualization image (optional)
    
    Returns:
        Visualization image as numpy array
    """
    if not chars:
        return None
    
    # Create a grid to display all character images with their classifications
    max_rows = 4
    max_cols = min(8, len(chars))
    rows = min(max_rows, (len(chars) + max_cols - 1) // max_cols)
    cols = min(max_cols, len(chars))
    
    # Determine the size of each character image
    char_size = 80
    padding = 10
    
    # Create the grid image
    grid_width = cols * (char_size + padding) + padding
    grid_height = rows * (char_size + padding) + padding + 40  # Extra space for title
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(grid_image, "Character Classification", (padding, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add each character to the grid
    for i, char_info in enumerate(chars[:rows*cols]):
        row = i // cols
        col = i % cols
        
        x = col * (char_size + padding) + padding
        y = row * (char_size + padding) + padding + 40  # Account for title
        
        # Resize character image to fit in the grid
        if 'image' in char_info and char_info['image'] is not None:
            char_img = char_info['image']
            char_img_resized = cv2.resize(char_img, (char_size, char_size))
            
            # Place character image in the grid
            grid_image[y:y+char_size, x:x+char_size] = char_img_resized
        
        # Add character and confidence text
        char_text = char_info['char']
        conf_text = f"{char_info['confidence']:.2f}"
        
        cv2.putText(grid_image, char_text, (x + 5, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(grid_image, conf_text, (x + 5, y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, grid_image)
    
    return grid_image

def visualize_vehicle_detection(image: np.ndarray, vehicles: List[Dict[str, Any]], save_path: str = None) -> np.ndarray:
    """
    Create a visualization of vehicle detection results.
    
    Args:
        image: Original image as numpy array
        vehicles: List of vehicle dictionaries with 'box' and 'confidence'
        save_path: Path to save the visualization image (optional)
    
    Returns:
        Visualization image as numpy array
    """
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw vehicle boxes
    for i, vehicle in enumerate(vehicles):
        x1, y1, x2, y2 = vehicle['box']
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw red dots at the corners of the box
        cv2.circle(vis_image, (x1, y1), 5, (0, 0, 255), -1)  # Top-left
        cv2.circle(vis_image, (x2, y1), 5, (0, 0, 255), -1)  # Top-right
        cv2.circle(vis_image, (x2, y2), 5, (0, 0, 255), -1)  # Bottom-right
        cv2.circle(vis_image, (x1, y2), 5, (0, 0, 255), -1)  # Bottom-left
        
        # Add vehicle index and confidence
        cv2.putText(vis_image, f"Vehicle {i+1} ({vehicle['confidence']:.2f})", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add title
    cv2.putText(vis_image, "Vehicle Detection", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Add count information
    cv2.putText(vis_image, f"Found {len(vehicles)} vehicles", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image

def visualize_vehicle_classification(vehicles: List[Dict[str, Any]], save_path: str = None) -> np.ndarray:
    """
    Create a visualization of vehicle classification results.
    
    Args:
        vehicles: List of vehicle dictionaries with 'make', 'model', 'confidence', and 'image'
        save_path: Path to save the visualization image (optional)
    
    Returns:
        Visualization image as numpy array
    """
    if not vehicles:
        return None
    
    # Determine the layout of the grid
    max_vehicles = min(4, len(vehicles))
    
    # Create a grid to display vehicle images with their classifications
    vehicle_width = 320
    vehicle_height = 240
    padding = 20
    
    # Create the grid image
    grid_width = max_vehicles * (vehicle_width + padding) + padding
    grid_height = vehicle_height + 2 * padding + 60  # Extra space for title and text
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(grid_image, "Vehicle Classification", (padding, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add each vehicle to the grid
    for i, vehicle_info in enumerate(vehicles[:max_vehicles]):
        x = i * (vehicle_width + padding) + padding
        y = padding + 40  # Account for title
        
        # Resize vehicle image to fit in the grid
        if 'image' in vehicle_info and vehicle_info['image'] is not None:
            vehicle_img = vehicle_info['image']
            vehicle_img_resized = cv2.resize(vehicle_img, (vehicle_width, vehicle_height))
            
            # Place vehicle image in the grid
            grid_image[y:y+vehicle_height, x:x+vehicle_width] = vehicle_img_resized
        
        # Add make, model, and confidence text
        make_model = f"{vehicle_info['make']} {vehicle_info['model']}"
        conf_text = f"Confidence: {vehicle_info['confidence']:.2f}"
        
        cv2.putText(grid_image, make_model, (x + 5, y + vehicle_height + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(grid_image, conf_text, (x + 5, y + vehicle_height + 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, grid_image)
    
    return grid_image

def visualize_results(image_path: str, results: Dict[str, Any], save_path: str = None, 
                     visualization_dir: str = None) -> np.ndarray:
    """
    Create a visualization of the ALPR results.
    
    Args:
        image_path: Path to the original image
        results: Results dictionary from process_alpr function
        save_path: Path to save the visualization image (optional)
        visualization_dir: Directory to save individual model visualizations (optional)
    
    Returns:
        Visualization image as numpy array
    """
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Colors for visualization (BGR format)
    day_plate_color = (0, 255, 0)     # Green
    night_plate_color = (0, 165, 255) # Orange
    vehicle_color = (255, 0, 0)       # Blue
    corner_point_color = (0, 0, 255)  # Red
    detection_box_color = (255, 255, 0) # Cyan
    original_corner_color = (255, 0, 255)  # Magenta
    text_color = (255, 255, 255)      # White
    
    # Draw day plates
    for i, plate in enumerate(results.get('day_plates', [])):
        # Draw detection box if available
        if "detection_box" in plate and plate["detection_box"] is not None:
            x1, y1, x2, y2 = [int(coord) for coord in plate["detection_box"]]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), detection_box_color, 2)
        
        # Draw original corners if available (before dilation)
        if "original_corners" in plate and plate["original_corners"] is not None:
            orig_corners = np.array(plate['original_corners'], dtype=np.int32)
            cv2.polylines(vis_image, [orig_corners.reshape((-1, 1, 2))], True, original_corner_color, 1)
            
            # Draw magenta dots at each original corner point
            for corner in orig_corners:
                x, y = corner
                cv2.circle(vis_image, (int(x), int(y)), 3, original_corner_color, -1)
        
        # Draw plate corners (dilated) as a polygon
        corners = np.array(plate['corners'], dtype=np.int32)
        cv2.polylines(vis_image, [corners.reshape((-1, 1, 2))], True, day_plate_color, 2)
        
        # Draw red dots at each dilated corner point
        for corner in corners:
            x, y = corner
            cv2.circle(vis_image, (int(x), int(y)), 5, corner_point_color, -1)  # -1 thickness means filled circle
        
        # Add plate number and confidence
        x, y = corners[0]
        y = max(y - 10, 10)  # Ensure text is visible
        plate_text = f"{plate['license_number']} ({plate['confidence']:.2f})"
        cv2.putText(vis_image, plate_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add dimensions info if available
        if "dimensions" in plate:
            dims = plate["dimensions"]
            if "width" in dims and "height" in dims and "actual_ratio" in dims:
                dim_text = f"Size: {dims['width']}x{dims['height']} (AR: {dims['actual_ratio']:.2f})"
                cv2.putText(vis_image, dim_text, (int(x), int(y) + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Add state information if available
        if 'state' in plate:
            state_y = int(y) + (50 if "dimensions" in plate else 25)
            state_text = f"State: {plate['state']} ({plate['state_confidence']:.2f})"
            cv2.putText(vis_image, state_text, (int(x), state_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Draw night plates
    for i, plate in enumerate(results.get('night_plates', [])):
        # Draw detection box if available
        if "detection_box" in plate and plate["detection_box"] is not None:
            x1, y1, x2, y2 = [int(coord) for coord in plate["detection_box"]]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), detection_box_color, 2)
        
        # Draw original corners if available (before dilation)
        if "original_corners" in plate and plate["original_corners"] is not None:
            orig_corners = np.array(plate['original_corners'], dtype=np.int32)
            cv2.polylines(vis_image, [orig_corners.reshape((-1, 1, 2))], True, original_corner_color, 1)
            
            # Draw magenta dots at each original corner point
            for corner in orig_corners:
                x, y = corner
                cv2.circle(vis_image, (int(x), int(y)), 3, original_corner_color, -1)
        
        # Draw plate corners (dilated) as a polygon
        corners = np.array(plate['corners'], dtype=np.int32)
        cv2.polylines(vis_image, [corners.reshape((-1, 1, 2))], True, night_plate_color, 2)
        
        # Draw red dots at each dilated corner point
        for corner in corners:
            x, y = corner
            cv2.circle(vis_image, (int(x), int(y)), 5, corner_point_color, -1)  # -1 thickness means filled circle
        
        # Add plate number and confidence
        x, y = corners[0]
        y = max(y - 10, 10)  # Ensure text is visible
        plate_text = f"{plate['license_number']} ({plate['confidence']:.2f})"
        cv2.putText(vis_image, plate_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add dimensions info if available
        if "dimensions" in plate:
            dims = plate["dimensions"]
            if "width" in dims and "height" in dims and "actual_ratio" in dims:
                dim_text = f"Size: {dims['width']}x{dims['height']} (AR: {dims['actual_ratio']:.2f})"
                cv2.putText(vis_image, dim_text, (int(x), int(y) + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Draw vehicles
    for i, vehicle in enumerate(results.get('vehicles', [])):
        # Draw vehicle box
        x1, y1, x2, y2 = vehicle['box']
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), vehicle_color, 2)
        
        # Draw red dots at the corners of the box
        cv2.circle(vis_image, (x1, y1), 5, corner_point_color, -1)  # Top-left
        cv2.circle(vis_image, (x2, y1), 5, corner_point_color, -1)  # Top-right
        cv2.circle(vis_image, (x2, y2), 5, corner_point_color, -1)  # Bottom-right
        cv2.circle(vis_image, (x1, y2), 5, corner_point_color, -1)  # Bottom-left
        
        # Add vehicle make and model
        vehicle_text = f"{vehicle['make']} {vehicle['model']} ({vehicle['confidence']:.2f})"
        cv2.putText(vis_image, vehicle_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis_image, "Legend:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(vis_image, "Day Plate", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, day_plate_color, 2)
    cv2.putText(vis_image, "Night Plate", (300, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, night_plate_color, 2)
    
    legend_y += 30
    cv2.circle(vis_image, (140, legend_y-5), 5, corner_point_color, -1)
    cv2.putText(vis_image, "Dilated Corners", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, corner_point_color, 2)
    cv2.putText(vis_image, "Detection Box", (300, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_box_color, 2)
    
    legend_y += 30
    cv2.circle(vis_image, (140, legend_y-5), 3, original_corner_color, -1)
    cv2.putText(vis_image, "Original Corners", (150, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, original_corner_color, 2)
    
    # Save visualization if requested
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    # Generate and save individual model visualizations if requested
    if visualization_dir:
        # Create the visualization directory if it doesn't exist
        os.makedirs(visualization_dir, exist_ok=True)
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. License plate detection visualization
        plate_det_path = os.path.join(visualization_dir, f"{base_filename}_plate_detection.jpg")
        visualize_plate_detection(image, {
            'day_plates': results.get('day_plates', []), 
            'night_plates': results.get('night_plates', [])
        }, plate_det_path)
        
        # 2. Process each plate for character detection, classification, and state classification
        for plate_type in ['day_plates', 'night_plates']:
            for i, plate in enumerate(results.get(plate_type, [])):
                # Get the warped plate image for visualization
                # We need to recreate it since it's not stored in the results
                try:
                    # This assumes we have access to the image and ALPRSystem is available
                    alpr = ALPRSystem(
                        plate_detector_path="",  # Placeholder paths, not actually used here
                        state_classifier_path="",
                        char_detector_path="",
                        char_classifier_path="",
                        vehicle_detector_path="",
                        vehicle_classifier_path="",
                        enable_state_detection=False,
                        enable_vehicle_detection=False,
                        plate_aspect_ratio=plate.get("aspect_ratio")  # Pass the same aspect ratio
                    )
                    
                    plate_image = alpr.four_point_transform(image, plate['corners'])
                    
                    # Character detection visualization
                    char_det_path = os.path.join(visualization_dir, 
                                               f"{base_filename}_{plate_type[:-1]}_{i+1}_char_detection.jpg")
                    visualize_character_detection(plate_image, plate.get('characters', []), char_det_path)
                    
                    # State classification visualization (only for day plates)
                    if plate_type == 'day_plates' and 'state' in plate:
                        state_cls_path = os.path.join(visualization_dir, 
                                                    f"{base_filename}_{plate_type[:-1]}_{i+1}_state_classification.jpg")
                        visualize_state_classification(plate_image, plate['state'], plate['state_confidence'], state_cls_path)
                except Exception as e:
                    print(f"Error creating plate visualization: {e}")
        
        # 3. Vehicle detection visualization
        if results.get('vehicles', []):
            vehicle_det_path = os.path.join(visualization_dir, f"{base_filename}_vehicle_detection.jpg")
            visualize_vehicle_detection(image, results.get('vehicles', []), vehicle_det_path)
            
            # 4. Vehicle classification visualization
            # Note: This requires the vehicle images which are removed from the results
            # We would need to process the image again to get these
    
    return vis_image


# Example of using the process_alpr function
if __name__ == "__main__":
    # Example file paths
    image_path = "test3.jpg"
    plate_detector_path = "models/plate_detector.pt"
    state_classifier_path = "models/state_classifier.pt"
    char_detector_path = "models/char_detector.pt"
    char_classifier_path = "models/char_classifier.pt"
    vehicle_detector_path = "models/vehicle_detector.pt"
    vehicle_classifier_path = "models/vehicle_classifier.pt"
    
    start_time = time.time()
    
    # Set output path for JSON results (optional)
    output_path = "results.json"
    
    # Create visualization directory
    visualization_dir = "visualization"
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Example: Process the image with all features, explicit aspect ratio, and corner dilation
    print("Running ALPR with all features, aspect ratio 3.0, and corner dilation...")
    results = process_alpr(
        image_path=image_path,
        plate_detector_path=plate_detector_path,
        state_classifier_path=state_classifier_path,
        char_detector_path=char_detector_path,
        char_classifier_path=char_classifier_path,
        vehicle_detector_path=vehicle_detector_path,
        vehicle_classifier_path=vehicle_classifier_path,
        enable_state_detection=True,
        enable_vehicle_detection=False,
        output_path=output_path,
        visualization_dir=visualization_dir,
        plate_detector_confidence=0.45,
        state_classifier_confidence=0.45,
        char_detector_confidence=0.40,
        char_classifier_confidence=0.40,
        vehicle_detector_confidence=0.45,
        vehicle_classifier_confidence=0.45,
        plate_aspect_ratio=4.0,  # US license plates typically have 3:1 width:height ratio
        corner_dilation_pixels=15  # 5 pixel corner dilation
    )
    
    # Initialize ALPR system to check configuration
    alpr = ALPRSystem(
        plate_detector_path=plate_detector_path,
        state_classifier_path=state_classifier_path,
        char_detector_path=char_detector_path,
        char_classifier_path=char_classifier_path,
        vehicle_detector_path=vehicle_detector_path,
        vehicle_classifier_path=vehicle_classifier_path,
        enable_state_detection=False,
        enable_vehicle_detection=False,
        plate_detector_confidence=0.80,
        state_classifier_confidence=0.70,
        char_detector_confidence=0.70,
        char_classifier_confidence=0.70,
        vehicle_detector_confidence=0.70,
        vehicle_classifier_confidence=0.70,
        plate_aspect_ratio=4.0,
        corner_dilation_pixels=15
    )
    
    # Print configuration
    config = alpr.get_config()
    print("ALPR System Configuration:")
    print(f"  - Device: {config['device']}")
    print(f"  - State detection enabled: {config['enable_state_detection']}")
    print(f"  - Vehicle detection enabled: {config['enable_vehicle_detection']}")
    print("  - Confidence thresholds:")
    for model_name, threshold in config['confidence_thresholds'].items():
        print(f"    - {model_name}: {threshold}")
    print("  - Models loaded:")
    for model_name, loaded in config['models_loaded'].items():
        print(f"    - {model_name}: {'✓' if loaded else '✗'}")
    print(f"  - Plate Aspect Ratio: {config['plate_aspect_ratio']}")
    print(f"  - Corner Dilation Pixels: {config['corner_dilation_pixels']}")
    print("\n")
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Process the results
    if results:
        print(f"Processing completed in {processing_time:.2f} seconds. Found:")
        print(f"  - {len(results['day_plates'])} day plates")
        print(f"  - {len(results['night_plates'])} night plates")
        print(f"  - {len(results['vehicles'])} vehicles")
        
        # Print license plate numbers and dimensions
        print("License plates:")
        for i, plate in enumerate(results['day_plates'] + results['night_plates']):
            plate_type = "Day" if i < len(results['day_plates']) else "Night"
            plate_info = f"  - {plate_type} Plate: {plate['license_number']} (Confidence: {plate['confidence']:.2f})"
            
            # Print dimensions if available
            if "dimensions" in plate:
                dims = plate["dimensions"]
                plate_info += f", Size: {dims['width']}x{dims['height']} (Ratio: {dims['actual_ratio']:.2f})"
            
            print(plate_info)
            
            # Print state for day plates
            if 'state' in plate:
                print(f"    State: {plate['state']} (Confidence: {plate['state_confidence']:.2f})")
        
        # Print vehicle information
        print("Vehicles:")
        for i, vehicle in enumerate(results['vehicles']):
            print(f"  - Vehicle {i+1}: {vehicle['make']} {vehicle['model']} (Confidence: {vehicle['confidence']:.2f})")
    else:
        print("Processing failed.")
        
    # Optional: Create visualizations of the results
    if results:
        print("\nCreating visualizations...")
        try:
            # Create visualization directory if it doesn't exist
            visualization_dir = "visualization"
            os.makedirs(visualization_dir, exist_ok=True)
            
            # Generate the main visualization
            vis_image = visualize_results(
                image_path=image_path, 
                results=results, 
                save_path=os.path.join(visualization_dir, "complete_results.jpg"),
                visualization_dir=visualization_dir
            )
            print(f"Visualizations saved to '{visualization_dir}' directory")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
