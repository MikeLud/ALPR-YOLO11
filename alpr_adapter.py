# Import general libraries
import os
import sys
import time
import json
import hashlib
import shutil
from typing import Dict, Any, Optional, List
from collections import OrderedDict
from pathlib import Path

# For PyTorch on Apple silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import CodeProject.AI SDK
from codeproject_ai_sdk import RequestData, ModuleRunner, LogMethod, JSON

# Import the method of the module we're wrapping
from PIL import Image, ImageDraw
import numpy as np
import cv2

from options import Options
from alpr_system_v205 import ALPRSystem, process_alpr

class LRUCache:
    """A simple LRU cache implementation for caching results"""
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        # Move to end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        # Move to end or add new item
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # Remove oldest if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class ALPR_adapter(ModuleRunner):

    def __init__(self):
        super().__init__()
        self.opts = Options()
        self.models_last_checked = None

        # These will be adjusted based on the hardware / packages found
        self.use_CUDA     = self.opts.use_CUDA
        self.use_MPS      = self.opts.use_MPS
        self.use_DirectML = self.opts.use_DirectML

        if self.use_CUDA and self.half_precision == True and \
           not self.system_info.hasTorchHalfPrecision:
            self.half_precision = False

        # Initialize the ALPR system once for use across requests
        self.alpr_system = None
        
        # Result cache for frequent license plates
        self.result_cache = LRUCache(self.opts.cache_size) if self.opts.enable_caching else None
        
        # Statistics tracking
        self._plates_detected = 0
        self._histogram = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def initialise(self):
        # CUDA takes precedence
        if self.use_CUDA:
            self.use_CUDA = self.system_info.hasTorchCuda
            # Potentially solve an issue around CUDNN_STATUS_ALLOC_FAILED errors
            try:
                import cudnn as cudnn
                if cudnn.is_available():
                    cudnn.benchmark = False
            except:
                pass

        # If no CUDA, maybe we're on an Apple Silicon Mac?
        if self.use_CUDA:
            self.use_MPS      = False
            self.use_DirectML = False
        else:
            self.use_MPS = self.system_info.hasTorchMPS

        # DirectML currently not supported
        self.use_DirectML = False

        self.can_use_GPU = self.system_info.hasTorchCuda or self.system_info.hasTorchMPS

        if self.use_CUDA:
            self.inference_device  = "GPU"
            self.inference_library = "CUDA"
        elif self.use_MPS:
            self.inference_device  = "GPU"
            self.inference_library = "MPS"
        elif self.use_DirectML:
            self.inference_device  = "GPU"
            self.inference_library = "DirectML"

        # Initialize statistics tracking
        self._plates_detected = 0
        self._histogram = {}

        # Initialize the ALPR system
        try:
            self.log(LogMethod.Info | LogMethod.Server,
            { 
                "filename": __file__,
                "loglevel": "information",
                "method": sys._getframe().f_code.co_name,
                "message": f"Initializing ALPR system with models from {self.opts.models_dir}"
            })
            
            # Convert device string for CUDA with TensorRT if enabled
            device = "cuda"
            if self.use_CUDA and self.opts.enable_tensorrt:
                try:
                    import torch
                    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                        device = "cuda:0"
                except ImportError:
                    pass
            elif not self.use_CUDA:
                device = "cpu"
            
            # Determine optimal resolution based on optimization level
            input_resolution = self.opts.input_resolution
            if self.opts.optimization_level >= 2:
                # Use lower resolution for higher speed
                input_resolution = min(input_resolution, 480)
            if self.opts.optimization_level >= 3:
                # Use even lower resolution for maximum speed
                input_resolution = min(input_resolution, 320)
            
            # Set up ONNX model paths if enabled
            if self.opts.enable_onnx:
                # Make sure ONNX directory exists
                os.makedirs(self.opts.onnx_model_path, exist_ok=True)
                
                # Convert PyTorch models to ONNX if not already converted
                self._convert_models_to_onnx()
                
                # Use ONNX model paths
                model_paths = {
                    "plate_detector": os.path.join(self.opts.onnx_model_path, "plate_detector.onnx"),
                    "state_classifier": os.path.join(self.opts.onnx_model_path, "state_classifier.onnx"),
                    "char_detector": os.path.join(self.opts.onnx_model_path, "char_detector.onnx"),
                    "char_classifier": os.path.join(self.opts.onnx_model_path, "char_classifier.onnx"),
                    "vehicle_detector": os.path.join(self.opts.onnx_model_path, "vehicle_detector.onnx"),
                    "vehicle_classifier": os.path.join(self.opts.onnx_model_path, "vehicle_classifier.onnx")
                }
                
                # Log ONNX usage
                self.log(LogMethod.Info | LogMethod.Server,
                {
                    "filename": __file__,
                    "loglevel": "information", 
                    "method": sys._getframe().f_code.co_name,
                    "message": f"Using ONNX models with provider: {self.opts.onnx_provider}"
                })
            else:
                # Use standard PyTorch model paths
                model_paths = {
                    "plate_detector": os.path.join(self.opts.models_dir, "plate_detector.pt"),
                    "state_classifier": os.path.join(self.opts.models_dir, "state_classifier.pt"),
                    "char_detector": os.path.join(self.opts.models_dir, "char_detector.pt"),
                    "char_classifier": os.path.join(self.opts.models_dir, "char_classifier.pt"),
                    "vehicle_detector": os.path.join(self.opts.models_dir, "vehicle_detector.pt"),
                    "vehicle_classifier": os.path.join(self.opts.models_dir, "vehicle_classifier.pt")
                }
            
            self.alpr_system = ALPRSystem(
                plate_detector_path=model_paths["plate_detector"],
                state_classifier_path=model_paths["state_classifier"],
                char_detector_path=model_paths["char_detector"],
                char_classifier_path=model_paths["char_classifier"],
                vehicle_detector_path=model_paths["vehicle_detector"],
                vehicle_classifier_path=model_paths["vehicle_classifier"],
                enable_state_detection=self.opts.enable_state_detection,
                enable_vehicle_detection=self.opts.enable_vehicle_detection,
                device=device,
                plate_detector_confidence=float(self.opts.plate_detector_confidence),
                state_classifier_confidence=float(self.opts.state_classifier_confidence),
                char_detector_confidence=float(self.opts.char_detector_confidence),
                char_classifier_confidence=float(self.opts.char_classifier_confidence),
                vehicle_detector_confidence=float(self.opts.vehicle_detector_confidence),
                vehicle_classifier_confidence=float(self.opts.vehicle_classifier_confidence),
                plate_aspect_ratio=float(self.opts.plate_aspect_ratio) if self.opts.plate_aspect_ratio else None,
                corner_dilation_pixels=int(self.opts.corner_dilation_pixels),
                # Optimization parameters
                half_precision=self.opts.half_precision and self.can_use_GPU,
                input_resolution=input_resolution,
                enable_tensorrt=self.opts.enable_tensorrt and self.use_CUDA,
                enable_onnx=self.opts.enable_onnx,
                onnx_provider=self.opts.onnx_provider,
                optimization_level=self.opts.optimization_level,
                enable_batch_processing=self.opts.enable_batch_processing,
                enable_adaptive_processing=self.opts.enable_adaptive_processing,
                max_thread_count=self.opts.max_thread_count
            )
            
            self.log(LogMethod.Info | LogMethod.Server,
            {
                "filename": __file__,
                "loglevel": "information", 
                "method": sys._getframe().f_code.co_name,
                "message": f"ALPR system initialized successfully with optimization level {self.opts.optimization_level}"
            })
            
        except Exception as ex:
            self.report_error(ex, __file__, f"Error initializing ALPR system: {str(ex)}")
            self.alpr_system = None

    def process(self, data: RequestData) -> JSON:
        
        response = None

        try:
            # The route to here is /v1/vision/alpr
            img = data.get_image(0)
                        
            # Get thresholds
            plate_threshold = float(data.get_value("min_confidence", "0.4"))
            
            # Only detect license plates
            response = self.detect_license_plate(img, plate_threshold)

        except Exception as ex:
            response = { "success": False, "error": f"Unknown command {data.command}" }
            self.report_error(None, __file__, f"Unknown command {data.command}")

        return response

    def _compute_image_hash(self, img) -> str:
        """
        Compute a hash of the image for caching purposes.
        Uses a perceptual hash that's resistant to minor image changes.
        """
        img_array = np.array(img)
        # Resize to a small size (32x32) to make hash more perceptual
        resized = cv2.resize(img_array, (32, 32), interpolation=cv2.INTER_AREA)
        # Convert to grayscale
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = resized
        # Compute hash
        flat = gray.flatten()
        # Get the average value
        avg = np.mean(flat)
        # Create a binary hash based on whether each pixel is above or below average
        bits = (flat > avg).astype(np.int8)
        # Convert to hexadecimal
        byte_array = np.packbits(bits)
        hash_hex = byte_array.tobytes().hex()
        return hash_hex

    def detect_license_plate(self, img, threshold):
        """
        Detect license plates in an image
        """
        if self.alpr_system is None:
            return {"success": False, "error": "ALPR system not initialized"}
        
        start_process_time = time.perf_counter()
        
        # Check cache if enabled
        if self.opts.enable_caching and self.result_cache is not None:
            # Compute a hash of the image
            img_hash = self._compute_image_hash(img)
            cached_result = self.result_cache.get(img_hash)
            if cached_result is not None:
                # If found in cache, update statistics and return
                self._cache_hits += 1
                cached_result["cached"] = True
                cached_result["processMs"] = int((time.perf_counter() - start_process_time) * 1000)
                return cached_result
            else:
                self._cache_misses += 1
                
        # Convert PIL Image to numpy array for OpenCV
        image_np = np.array(img)
        # Convert RGB to BGR (OpenCV format)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Color image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        try:
            start_inference_time = time.perf_counter()
            
            # Skip vehicle detection at higher optimization levels if not explicitly needed
            skip_vehicle_detection = False
            if self.opts.optimization_level >= 2 and self.opts.enable_adaptive_processing:
                skip_vehicle_detection = True
            
            # Process the image to find license plates
            plate_detection = self.alpr_system.detect_license_plates(image_np)
            
            # Process each plate
            results = {
                "day_plates": [],
                "night_plates": []
            }
            
            # Adaptive processing - only process plates with confidence above threshold
            # For maximum speed (optimization_level 3), increase threshold slightly
            effective_threshold = threshold
            if self.opts.optimization_level >= 3 and self.opts.enable_adaptive_processing:
                effective_threshold = max(threshold, 0.45)  # Slightly higher threshold for speed
            
            # Skip additional processing if no plates were found with good confidence
            day_plates_count = sum(1 for plate in plate_detection["day_plates"] if plate["confidence"] >= effective_threshold)
            night_plates_count = sum(1 for plate in plate_detection["night_plates"] if plate["confidence"] >= effective_threshold)
            
            if day_plates_count + night_plates_count == 0:
                inferenceMs = int((time.perf_counter() - start_inference_time) * 1000)
                result = {
                    "success": True,
                    "processMs": int((time.perf_counter() - start_process_time) * 1000),
                    "inferenceMs": inferenceMs,
                    "predictions": [],
                    "message": "No license plates detected",
                    "count": 0
                }
                
                # Store in cache if enabled
                if self.opts.enable_caching and self.result_cache is not None:
                    img_hash = self._compute_image_hash(img)
                    self.result_cache.put(img_hash, result.copy())
                
                return result
            
            # Process plates more efficiently
            for plate_type in ["day_plates", "night_plates"]:
                # Filter plates by threshold first to avoid unnecessary processing
                filtered_plates = [plate for plate in plate_detection[plate_type] if plate["confidence"] >= effective_threshold]
                
                # Batch process plates if enabled
                if self.opts.enable_batch_processing and len(filtered_plates) > 1:
                    batch_results = self.alpr_system.process_plates_batch(
                        image_np, 
                        filtered_plates, 
                        plate_type == "day_plates",
                        threshold=effective_threshold
                    )
                    results[plate_type].extend(batch_results)
                else:
                    # Process plates individually
                    for plate_info in filtered_plates:
                        plate_result = self.alpr_system.process_plate(
                            image_np, 
                            plate_info, 
                            plate_type == "day_plates",
                            skip_state_detection=(self.opts.optimization_level >= 3),
                            skip_character_recognition=(len(filtered_plates) > 5 and self.opts.optimization_level >= 3)
                        )
                        if plate_result["confidence"] >= threshold:
                            results[plate_type].append(plate_result)
            
            inferenceMs = int((time.perf_counter() - start_inference_time) * 1000)
            
            # Extract plate numbers and coordinates for client response
            plates = []
            for plate_type in ["day_plates", "night_plates"]:
                for plate in results[plate_type]:
                    # Only include plates with confidence above threshold
                    if plate["confidence"] >= threshold:
                        # Use detection_box if available, otherwise calculate from corners
                        if "detection_box" in plate and plate["detection_box"] is not None:
                            # If detection_box is available, use it directly
                            x1, y1, x2, y2 = plate["detection_box"]
                            plate_data = {
                                "confidence": plate["confidence"],
                                "is_day_plate": plate["is_day_plate"],
                                "label": plate["license_number"],
                                "plate": plate["license_number"],
                                "x_min": x1,
                                "y_min": y1,
                                "x_max": x2,
                                "y_max": y2
                            }
                        else:
                            # Otherwise, calculate the bounding box from the corners
                            corners = plate["corners"]
                            # Convert corners to numpy array if not already
                            corners_arr = np.array(corners)
                            x_min = np.min(corners_arr[:, 0])
                            y_min = np.min(corners_arr[:, 1])
                            x_max = np.max(corners_arr[:, 0])
                            y_max = np.max(corners_arr[:, 1])
                            
                            plate_data = {
                                "confidence": plate["confidence"],
                                "is_day_plate": plate["is_day_plate"],
                                "label": plate["license_number"],
                                "plate": plate["license_number"],
                                "x_min": float(x_min),
                                "y_min": float(y_min),
                                "x_max": float(x_max),
                                "y_max": float(y_max)
                            }
                        
                        if "state" in plate:
                            plate_data["state"] = plate["state"]
                            plate_data["state_confidence"] = plate["state_confidence"]
                        
                        # Add top plate alternatives
                        if "top_plates" in plate:
                            plate_data["top_plates"] = plate["top_plates"]
                            
                        plates.append(plate_data)
            
            # Update statistics
            self._plates_detected += len(plates)
            for plate in plates:
                license_num = plate["label"]
                if license_num not in self._histogram:
                    self._histogram[license_num] = 1
                else:
                    self._histogram[license_num] += 1
            
            # Create a response message
            if len(plates) > 0:
                message = f"Found {len(plates)} license plates"
                if len(plates) <= 3:
                    message += ": " + ", ".join([p["label"] for p in plates])
            else:
                message = "No license plates detected"
            
            result = {
                "success": True,
                "processMs": int((time.perf_counter() - start_process_time) * 1000),
                "inferenceMs": inferenceMs,
                "predictions": plates,
                "message": message,
                "count": len(plates)
            }
            
            # Store in cache if enabled
            if self.opts.enable_caching and self.result_cache is not None:
                img_hash = self._compute_image_hash(img)
                # Store a copy to avoid modification
                self.result_cache.put(img_hash, result.copy())
            
            return result
            
        except Exception as ex:
            self.report_error(ex, __file__, f"Error detecting license plates: {str(ex)}")
            return {"success": False, "error": f"Error detecting license plates: {str(ex)}"}
    
    def status(self) -> JSON:
        statusData = super().status()
        statusData["platesDetected"] = self._plates_detected
        statusData["histogram"] = self._histogram
        
        # Add cache statistics if caching is enabled
        if self.opts.enable_caching and self.result_cache is not None:
            statusData["cacheHits"] = self._cache_hits
            statusData["cacheMisses"] = self._cache_misses
            statusData["cacheHitRate"] = round(self._cache_hits / max(1, self._cache_hits + self._cache_misses) * 100, 2)
            statusData["cacheSize"] = len(self.result_cache.cache)
            statusData["cacheMaxSize"] = self.result_cache.capacity
        
        # Add optimization info
        statusData["optimizationLevel"] = self.opts.optimization_level
        statusData["inputResolution"] = self.opts.input_resolution
        statusData["halfPrecision"] = self.opts.half_precision
        statusData["tensorRT"] = self.opts.enable_tensorrt if self.use_CUDA else False
        statusData["onnx"] = self.opts.enable_onnx
        if self.opts.enable_onnx:
            statusData["onnxProvider"] = self.alpr_system.active_onnx_provider if hasattr(self.alpr_system, "active_onnx_provider") else self.opts.onnx_provider
        
        return statusData

    def _convert_models_to_onnx(self) -> None:
        """
        Converts PyTorch models to ONNX format if not already converted
        """
        try:
            models_to_convert = [
                ("plate_detector.pt", "plate_detector.onnx", "pose"),
                ("state_classifier.pt", "state_classifier.onnx", "classify"),
                ("char_detector.pt", "char_detector.onnx", "detect"),
                ("char_classifier.pt", "char_classifier.onnx", "classify"),
                ("vehicle_detector.pt", "vehicle_detector.onnx", "detect"),
                ("vehicle_classifier.pt", "vehicle_classifier.onnx", "classify")
            ]
            
            self.log(LogMethod.Info | LogMethod.Server,
            {
                "filename": __file__,
                "loglevel": "information", 
                "method": sys._getframe().f_code.co_name,
                "message": f"Checking for ONNX model conversions..."
            })
            
            # Check if ONNX models exist and PyTorch models are newer
            for pt_filename, onnx_filename, task in models_to_convert:
                pt_path = os.path.join(self.opts.models_dir, pt_filename)
                onnx_path = os.path.join(self.opts.onnx_model_path, onnx_filename)
                
                # Skip if PyTorch model doesn't exist
                if not os.path.exists(pt_path):
                    continue
                
                # Get file modification times
                pt_mtime = os.path.getmtime(pt_path) if os.path.exists(pt_path) else 0
                onnx_mtime = os.path.getmtime(onnx_path) if os.path.exists(onnx_path) else 0
                
                # Only convert if ONNX doesn't exist or PyTorch model is newer
                if not os.path.exists(onnx_path) or pt_mtime > onnx_mtime:
                    self.log(LogMethod.Info | LogMethod.Server,
                    {
                        "filename": __file__,
                        "loglevel": "information", 
                        "method": sys._getframe().f_code.co_name,
                        "message": f"Converting {pt_filename} to ONNX format..."
                    })
                    
                    # Import here to avoid importing if not needed
                    from ultralytics import YOLO
                    
                    # Load the model
                    model = YOLO(pt_path, task=task)
                    
                    # Export to ONNX
                    success = model.export(format="onnx", 
                                          dynamic=True, 
                                          simplify=True, 
                                          opset=12,
                                          output=onnx_path)
                    
                    if success:
                        self.log(LogMethod.Info | LogMethod.Server,
                        {
                            "filename": __file__,
                            "loglevel": "information", 
                            "method": sys._getframe().f_code.co_name,
                            "message": f"Successfully converted {pt_filename} to ONNX"
                        })
                    else:
                        self.log(LogMethod.Error | LogMethod.Server,
                        {
                            "filename": __file__,
                            "loglevel": "error", 
                            "method": sys._getframe().f_code.co_name,
                            "message": f"Failed to convert {pt_filename} to ONNX"
                        })
        except Exception as ex:
            self.report_error(ex, __file__, f"Error converting models to ONNX: {str(ex)}")
    
    def selftest(self) -> JSON:
        # If we don't have any test images, just return success if we could initialize the system
        if self.alpr_system is None:
            return {
                "success": False,
                "message": "ALPR system failed to initialize"
            }
        
        test_file = os.path.join("test", "license_plate_test.jpg")
        if not os.path.exists(test_file):
            return {
                "success": True,
                "message": "ALPR system initialized successfully (no test image available)"
            }
        
        # Test with an actual image
        request_data = RequestData()
        request_data.queue = self.queue_name
        request_data.command = "detect"
        request_data.add_file(test_file)
        request_data.add_value("operation", "plate")
        request_data.add_value("min_confidence", 0.4)
        
        result = self.process(request_data)
        print(f"Info: Self-test for {self.module_id}. Success: {result['success']}")
        
        if result['success']:
            message = "ALPR system test successful"
            if result.get('count', 0) > 0:
                message += f" - detected {result['count']} license plates"
        else:
            message = "ALPR system test failed"
            
        return { "success": result['success'], "message": message }

if __name__ == "__main__":
    ALPR_adapter().start_loop()
