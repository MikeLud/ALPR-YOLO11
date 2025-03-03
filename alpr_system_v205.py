import cv2
import numpy as np
import torch
import concurrent.futures
import json
import os
import time
import platform
from typing import List, Dict, Tuple, Optional, Union, Any
from ultralytics import YOLO
from pathlib import Path

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
        corner_dilation_pixels: int = 5,
        # Performance optimization parameters
        half_precision: bool = False,
        input_resolution: int = 640,
        enable_tensorrt: bool = False,
        enable_onnx: bool = False,
        onnx_provider: str = "auto",
        optimization_level: int = 1,
        enable_batch_processing: bool = True,
        enable_adaptive_processing: bool = True,
        max_thread_count: int = 4
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
            half_precision: Whether to use FP16 (half) precision
            input_resolution: Input resolution for the models (lower = faster)
            enable_tensorrt: Whether to enable TensorRT for NVIDIA GPUs
            enable_onnx: Whether to use ONNX models instead of PyTorch
            onnx_provider: Which ONNX execution provider to use
            optimization_level: Level of optimization (1=balanced, 2=speed, 3=max speed)
            enable_batch_processing: Whether to enable batch processing for multiple detections
            enable_adaptive_processing: Whether to enable adaptive processing based on scene complexity
            max_thread_count: Maximum number of threads to use for parallel processing
        """
        # Save optimization parameters
        self.half_precision = half_precision
        self.input_resolution = input_resolution
        self.enable_tensorrt = enable_tensorrt
        self.enable_onnx = enable_onnx
        self.onnx_provider = onnx_provider
        self.active_onnx_provider = None  # Will be set during initialization if ONNX is used
        self.optimization_level = optimization_level
        self.enable_batch_processing = enable_batch_processing
        self.enable_adaptive_processing = enable_adaptive_processing
        self.max_thread_count = max_thread_count
        
        # TensorRT configuration if enabled
        if enable_tensorrt and device.startswith("cuda") and torch.cuda.is_available():
            try:
                # Try to set up TensorRT
                YOLO.device = device  # Ensure device is set
                # Enable TensorRT by setting environment variable
                os.environ["CUDA_MODULE_LOADING"] = "LAZY"
            except Exception as e:
                print(f"Warning: TensorRT setup failed - {e}")
                self.enable_tensorrt = False
        
        # Calculate model resolutions based on input_resolution
        self.plate_detector_resolution = (input_resolution, input_resolution)
        self.state_classifier_resolution = (224, 224)  # Keep standard for classification
        self.char_detector_resolution = (max(160, input_resolution // 4), max(160, input_resolution // 4))
        self.char_classifier_resolution = (32, 32)  # Keep standard for character classification
        self.vehicle_detector_resolution = (input_resolution, input_resolution)
        self.vehicle_classifier_resolution = (224, 224)  # Keep standard for classification
        
        # Determine if we're using ONNX models
        if self.enable_onnx:
            try:
                # Import ONNX runtime here to avoid importing if not using ONNX
                import onnxruntime as ort
                
                # Determine available providers
                available_providers = ort.get_available_providers()
                print(f"Available ONNX Runtime providers: {available_providers}")
                
                # Set the active provider based on user selection and availability
                selected_provider = self._get_onnx_provider(self.onnx_provider, available_providers)
                self.active_onnx_provider = selected_provider
                
                print(f"Using ONNX Runtime with provider: {selected_provider}")
                
                # Configure ONNX session options
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Optimize based on optimization_level
                if self.optimization_level >= 2:
                    session_options.enable_profiling = False
                    session_options.enable_mem_pattern = True
                    session_options.enable_cpu_mem_arena = True
                    session_options.intra_op_num_threads = self.max_thread_count
                
                # Create ONNX inference sessions instead of loading YOLO models
                self.plate_detector_model = self._create_onnx_session(plate_detector_path, session_options, [selected_provider], task='pose')
                self.state_classifier_model = None  # Lazy load
                self.char_detector_model = self._create_onnx_session(char_detector_path, session_options, [selected_provider], task='detect')
                self.char_classifier_model = self._create_onnx_session(char_classifier_path, session_options, [selected_provider], task='classify')
                self.vehicle_detector_model = None  # Lazy load
                self.vehicle_classifier_model = None  # Lazy load
                
                # Store paths for lazy loading
                self.state_classifier_path = state_classifier_path if enable_state_detection else None
                self.vehicle_detector_path = vehicle_detector_path if enable_vehicle_detection else None
                self.vehicle_classifier_path = vehicle_classifier_path if enable_vehicle_detection else None
                
                # Model load status tracking
                self._models_loaded = {
                    "plate_detector": self.plate_detector_model is not None,
                    "state_classifier": False,
                    "char_detector": self.char_detector_model is not None,
                    "char_classifier": self.char_classifier_model is not None,
                    "vehicle_detector": False,
                    "vehicle_classifier": False
                }
                
                # We'll use a different execution path for ONNX
                self._using_onnx = True
                
                # Set the "names" dictionary for each model (needed for class mapping)
                self._load_model_names()
                
                return
                
            except ImportError:
                print("ONNX Runtime not installed. Falling back to PyTorch models.")
                self.enable_onnx = False
            except Exception as e:
                print(f"Error initializing ONNX: {e}")
                self.enable_onnx = False
        
        # If we're here, we're not using ONNX
        self._using_onnx = False
        
        # Prepare model loading arguments
        yolo_args = {"task": "pose", "verbose": False}
        # Don't use half=True parameter as it's not supported in YOLOv8
        
        # Load plate detector with task='pose' for keypoint detection
        try:
            self.plate_detector_model = YOLO(plate_detector_path, **yolo_args)
            
            # Apply half precision after loading if needed
            if half_precision and (device.startswith("cuda") or device == "mps"):
                self.plate_detector_model.to(device)
                if hasattr(self.plate_detector_model, 'model'):
                    self.plate_detector_model.model = self.plate_detector_model.model.half()
                    
            if enable_tensorrt and device.startswith("cuda"):
                self.plate_detector_model.to(device)
                if self.optimization_level >= 2:
                    # Export to TensorRT - only at higher optimization levels since it takes time
                    try:
                        export_name = os.path.join(os.path.dirname(plate_detector_path), "plate_detector_trt.engine")
                        if not os.path.exists(export_name):
                            self.plate_detector_model.export(format="engine", device=0)
                    except Exception as e:
                        print(f"TensorRT export failed: {e}")
        except Exception as e:
            print(f"Error loading plate detector: {e}")
            self.plate_detector_model = None
        
        # Load other models with appropriate settings
        cls_args = {"task": "classify", "verbose": False}
        # Don't use half=True parameter
        
        detect_args = {"task": "detect", "verbose": False}
        # Don't use half=True parameter
        
        # Load remaining models only when needed (lazy loading)
        self.state_classifier_model = None
        self.char_detector_model = None
        self.char_classifier_model = None
        self.vehicle_detector_model = None
        self.vehicle_classifier_model = None
        
        # Load character detector since it's almost always needed
        try:
            self.char_detector_model = YOLO(char_detector_path, **detect_args)
            
            # Apply half precision after loading if needed
            if half_precision and (device.startswith("cuda") or device == "mps"):
                self.char_detector_model.to(device)
                if hasattr(self.char_detector_model, 'model'):
                    self.char_detector_model.model = self.char_detector_model.model.half()
                    
            if enable_tensorrt and device.startswith("cuda"):
                self.char_detector_model.to(device)
        except Exception as e:
            print(f"Error loading character detector: {e}")
        
        # Load character classifier since it's almost always needed
        try:
            self.char_classifier_model = YOLO(char_classifier_path, **cls_args)
            
            # Apply half precision after loading if needed
            if half_precision and (device.startswith("cuda") or device == "mps"):
                self.char_classifier_model.to(device)
                if hasattr(self.char_classifier_model, 'model'):
                    self.char_classifier_model.model = self.char_classifier_model.model.half()
                    
            if enable_tensorrt and device.startswith("cuda"):
                self.char_classifier_model.to(device)
        except Exception as e:
            print(f"Error loading character classifier: {e}")
        
        # Store paths for lazy loading later
        self.state_classifier_path = state_classifier_path if enable_state_detection else None
        self.vehicle_detector_path = vehicle_detector_path if enable_vehicle_detection else None
        self.vehicle_classifier_path = vehicle_classifier_path if enable_vehicle_detection else None
        
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
        
        # Create a ThreadPoolExecutor with a maximum number of workers
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_count)
        
        # Model load status tracking
        self._models_loaded = {
            "plate_detector": self.plate_detector_model is not None,
            "state_classifier": False,
            "char_detector": self.char_detector_model is not None,
            "char_classifier": self.char_classifier_model is not None,
            "vehicle_detector": False,
            "vehicle_classifier": False
        }
        
    def _get_onnx_provider(self, requested_provider: str, available_providers: List[str]) -> str:
        """
        Determines the appropriate ONNX provider based on user selection and availability
        
        Args:
            requested_provider: The provider requested by the user
            available_providers: List of available providers
            
        Returns:
            The provider to use
        """
        # Map user-friendly names to ONNX provider names
        provider_map = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider", 
            "dml": "DmlExecutionProvider",  # DirectML for Windows
            "coreml": "CoreMLExecutionProvider",  # Apple CoreML
            "tensorrt": "TensorrtExecutionProvider"
        }
        
        # Auto detection - find the best available provider
        if requested_provider == "auto":
            # Try GPU providers first
            for gpu_provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider", 
                              "DmlExecutionProvider", "CoreMLExecutionProvider"]:
                if gpu_provider in available_providers:
                    return gpu_provider
            
            # Fall back to CPU
            return "CPUExecutionProvider"
        
        # Get the actual provider name from the map
        actual_provider = provider_map.get(requested_provider.lower(), requested_provider)
        
        # Check if requested provider is available
        if actual_provider in available_providers:
            return actual_provider
        
        # If requested provider is not available, fall back to CPU
        print(f"Requested ONNX provider '{actual_provider}' not available. Falling back to CPU.")
        return "CPUExecutionProvider"
    
    def _create_onnx_session(self, model_path: str, session_options, providers: List[str], task: str = 'detect'):
        """
        Create an ONNX Runtime session for inference
        
        Args:
            model_path: Path to the ONNX model file
            session_options: ONNX session options
            providers: List of execution providers
            task: Model task type
            
        Returns:
            ONNX Runtime session or None if creation failed
        """
        try:
            import onnxruntime as ort
            
            # Verify file exists
            if not os.path.exists(model_path):
                print(f"ONNX model not found: {model_path}")
                return None
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Store the task type with the session
            session.task = task
            
            return session
        except Exception as e:
            print(f"Error creating ONNX session for {model_path}: {e}")
            return None
    
    def _load_model_names(self):
        """
        Load class names for ONNX models from corresponding JSON files
        """
        try:
            # For each model, check if there's a names.json file alongside it
            model_types = [
                "plate_detector", "state_classifier", "char_detector", 
                "char_classifier", "vehicle_detector", "vehicle_classifier"
            ]
            
            self.model_names = {}
            
            for model_type in model_types:
                # Get model path
                model_path = getattr(self, f"{model_type}_path", None)
                if model_path:
                    # Find names.json in the same directory
                    names_path = os.path.join(os.path.dirname(model_path), f"{model_type}_names.json")
                    if os.path.exists(names_path):
                        with open(names_path, 'r') as f:
                            self.model_names[model_type] = json.load(f)
                    else:
                        # If no names file, create a simple numeric mapping
                        # This is a fallback and may not be accurate
                        if model_type == "char_classifier":
                            # Common characters for license plates
                            chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            self.model_names[model_type] = {str(i): char for i, char in enumerate(chars)}
                        else:
                            # Generic class names
                            self.model_names[model_type] = {str(i): f"class_{i}" for i in range(100)}
        except Exception as e:
            print(f"Error loading model class names: {e}")
    
    def _lazy_load_model(self, model_type: str) -> bool:
        """
        Lazy-load a model when needed
        
        Args:
            model_type: The type of model to load ('state_classifier', 'vehicle_detector', 'vehicle_classifier')
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        if self._using_onnx:
            # ONNX model lazy loading
            try:
                if model_type == "state_classifier" and self.enable_state_detection and self.state_classifier_path:
                    if self.state_classifier_model is None and os.path.exists(self.state_classifier_path):
                        import onnxruntime as ort
                        session_options = ort.SessionOptions()
                        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        if self.optimization_level >= 2:
                            session_options.enable_profiling = False
                            session_options.enable_mem_pattern = True
                            session_options.enable_cpu_mem_arena = True
                            session_options.intra_op_num_threads = self.max_thread_count
                        self.state_classifier_model = self._create_onnx_session(
                            self.state_classifier_path, 
                            session_options, 
                            [self.active_onnx_provider], 
                            task='classify'
                        )
                        self._models_loaded["state_classifier"] = self.state_classifier_model is not None
                        return self.state_classifier_model is not None
                    return self.state_classifier_model is not None
                
                elif model_type == "vehicle_detector" and self.enable_vehicle_detection and self.vehicle_detector_path:
                    if self.vehicle_detector_model is None and os.path.exists(self.vehicle_detector_path):
                        import onnxruntime as ort
                        session_options = ort.SessionOptions()
                        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        if self.optimization_level >= 2:
                            session_options.enable_profiling = False
                            session_options.enable_mem_pattern = True
                            session_options.enable_cpu_mem_arena = True
                            session_options.intra_op_num_threads = self.max_thread_count
                        self.vehicle_detector_model = self._create_onnx_session(
                            self.vehicle_detector_path, 
                            session_options, 
                            [self.active_onnx_provider], 
                            task='detect'
                        )
                        self._models_loaded["vehicle_detector"] = self.vehicle_detector_model is not None
                        return self.vehicle_detector_model is not None
                    return self.vehicle_detector_model is not None
                
                elif model_type == "vehicle_classifier" and self.enable_vehicle_detection and self.vehicle_classifier_path:
                    if self.vehicle_classifier_model is None and os.path.exists(self.vehicle_classifier_path):
                        import onnxruntime as ort
                        session_options = ort.SessionOptions()
                        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        if self.optimization_level >= 2:
                            session_options.enable_profiling = False
                            session_options.enable_mem_pattern = True
                            session_options.enable_cpu_mem_arena = True
                            session_options.intra_op_num_threads = self.max_thread_count
                        self.vehicle_classifier_model = self._create_onnx_session(
                            self.vehicle_classifier_path, 
                            session_options, 
                            [self.active_onnx_provider], 
                            task='classify'
                        )
                        self._models_loaded["vehicle_classifier"] = self.vehicle_classifier_model is not None
                        return self.vehicle_classifier_model is not None
                    return self.vehicle_classifier_model is not None
                
            except Exception as e:
                print(f"Error lazy-loading ONNX model {model_type}: {e}")
                return False
            
            return False
        
        # PyTorch model lazy loading
        if model_type == "state_classifier" and self.enable_state_detection and self.state_classifier_path:
            if self.state_classifier_model is None:
                try:
                    cls_args = {"task": "classify", "verbose": False}
                    # Don't use half=True parameter
                    
                    self.state_classifier_model = YOLO(self.state_classifier_path, **cls_args)
                    
                    # Apply half precision after loading if needed
                    if self.half_precision and (self.device.startswith("cuda") or self.device == "mps"):
                        self.state_classifier_model.to(self.device)
                        if hasattr(self.state_classifier_model, 'model'):
                            self.state_classifier_model.model = self.state_classifier_model.model.half()
                    
                    if self.enable_tensorrt and self.device.startswith("cuda"):
                        self.state_classifier_model.to(self.device)
                    self._models_loaded["state_classifier"] = True
                    return True
                except Exception as e:
                    print(f"Error lazy-loading state classifier: {e}")
                    return False
            return True
            
        elif model_type == "vehicle_detector" and self.enable_vehicle_detection and self.vehicle_detector_path:
            if self.vehicle_detector_model is None:
                try:
                    detect_args = {"task": "detect", "verbose": False}
                    # Don't use half=True parameter
                    
                    self.vehicle_detector_model = YOLO(self.vehicle_detector_path, **detect_args)
                    
                    # Apply half precision after loading if needed
                    if self.half_precision and (self.device.startswith("cuda") or self.device == "mps"):
                        self.vehicle_detector_model.to(self.device)
                        if hasattr(self.vehicle_detector_model, 'model'):
                            self.vehicle_detector_model.model = self.vehicle_detector_model.model.half()
                    
                    if self.enable_tensorrt and self.device.startswith("cuda"):
                        self.vehicle_detector_model.to(self.device)
                    self._models_loaded["vehicle_detector"] = True
                    return True
                except Exception as e:
                    print(f"Error lazy-loading vehicle detector: {e}")
                    return False
            return True
            
        elif model_type == "vehicle_classifier" and self.enable_vehicle_detection and self.vehicle_classifier_path:
            if self.vehicle_classifier_model is None:
                try:
                    cls_args = {"task": "classify", "verbose": False}
                    # Don't use half=True parameter
                    
                    self.vehicle_classifier_model = YOLO(self.vehicle_classifier_path, **cls_args)
                    
                    # Apply half precision after loading if needed
                    if self.half_precision and (self.device.startswith("cuda") or self.device == "mps"):
                        self.vehicle_classifier_model.to(self.device)
                        if hasattr(self.vehicle_classifier_model, 'model'):
                            self.vehicle_classifier_model.model = self.vehicle_classifier_model.model.half()
                    
                    if self.enable_tensorrt and self.device.startswith("cuda"):
                        self.vehicle_classifier_model.to(self.device)
                    self._models_loaded["vehicle_classifier"] = True
                    return True
                except Exception as e:
                    print(f"Error lazy-loading vehicle classifier: {e}")
                    return False
            return True
            
        return False
        
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
            "models_loaded": self._models_loaded,
            "plate_aspect_ratio": self.plate_aspect_ratio,
            "corner_dilation_pixels": self.corner_dilation_pixels,
            "using_onnx": self._using_onnx,
            "onnx_provider": self.active_onnx_provider if self._using_onnx else None,
            "optimization_level": self.optimization_level,
            "enable_batch_processing": self.enable_batch_processing,
            "enable_adaptive_processing": self.enable_adaptive_processing
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
        if self.plate_detector_model is None:
            return {"day_plates": [], "night_plates": []}
        
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Adaptive resolution scaling for better performance
        resolution = self.plate_detector_resolution
        if self.enable_adaptive_processing:
            # For very small images, use the input resolution directly
            if orig_h <= self.input_resolution and orig_w <= self.input_resolution:
                resolution = (orig_w, orig_h)
            # For very large images, use a lower resolution
            elif max(orig_h, orig_w) > 1920 and self.optimization_level >= 2:
                scale_factor = min(1.0, 1920 / max(orig_h, orig_w))
                resolution = (int(orig_w * scale_factor), int(orig_h * scale_factor))
        
        # Resize image for plate detector
        img_resized = cv2.resize(image, resolution)
        
        try:
            if self._using_onnx:
                # Using ONNX Runtime for inference
                results = self._run_onnx_inference(self.plate_detector_model, img_resized, task='pose', 
                                                 threshold=self.plate_detector_confidence)
            else:
                # Using PyTorch/YOLOv8 for inference
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
                    scale_x = w / resolution[0]
                    scale_y = h / resolution[1]
                    
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
    
    def _run_onnx_inference(self, model, image: np.ndarray, task: str = 'detect', threshold: float = 0.4):
        """
        Run inference using ONNX Runtime
        
        Args:
            model: ONNX Runtime session
            image: Input image
            task: Task type (detect, classify, pose)
            threshold: Confidence threshold
            
        Returns:
            Results in a format similar to YOLOv8 output
        """
        if model is None:
            return None
            
        try:
            # Preprocess image
            input_name = model.get_inputs()[0].name
            
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3 and task != 'classify':
                # YOLOv8 models expect RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize and transpose
            img = image.astype(np.float32) / 255.0
            
            if task == 'classify':
                # Classification models expect different input shape
                img = cv2.resize(img, (224, 224))
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                input_tensor = np.expand_dims(img, axis=0)
            else:
                # Detection models
                img = cv2.resize(img, (640, 640))
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                input_tensor = np.expand_dims(img, axis=0)
            
            # Run inference
            outputs = model.run(None, {input_name: input_tensor})
            
            # Process outputs based on task
            if task == 'classify':
                # Return a class object with probs
                class_scores = outputs[0][0]
                top_idx = np.argmax(class_scores)
                top_conf = class_scores[top_idx]
                
                # Create a results object similar to YOLOv8
                class OnnxClassifyResults:
                    class probs:
                        def __init__(self, scores, top_idx, top_conf):
                            self.data = torch.tensor(scores)
                            self.top1 = top_idx
                            self.top1conf = torch.tensor(top_conf)
                
                results = [OnnxClassifyResults()]
                results[0].probs = OnnxClassifyResults.probs(class_scores, top_idx, top_conf)
                
                return results
                
            elif task == 'pose':
                # Process keypoint outputs for pose detection
                # The output format depends on the model, but typically contains:
                # - boxes: [batch, num_boxes, 4] - coordinates
                # - scores: [batch, num_boxes] - confidence scores
                # - classes: [batch, num_boxes] - class IDs
                # - keypoints: [batch, num_boxes, num_keypoints, 3] - x, y, conf for each keypoint
                
                # Extract boxes, scores, classes from outputs
                boxes = outputs[0][0]  # First batch, boxes
                scores = outputs[1][0]  # First batch, scores
                classes = outputs[2][0]  # First batch, classes
                keypoints = outputs[3][0]  # First batch, keypoints
                
                # Filter by confidence threshold
                mask = scores > threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
                filtered_classes = classes[mask]
                filtered_keypoints = keypoints[mask]
                
                # Create a results object similar to YOLOv8
                class OnnxPoseResults:
                    class boxes:
                        def __init__(self, boxes, scores, classes):
                            self.xyxy = torch.tensor(boxes)
                            self.conf = torch.tensor(scores)
                            self.cls = torch.tensor(classes)
                    
                    class keypoints:
                        def __init__(self, kpts):
                            self.data = torch.tensor(kpts)
                
                results = [OnnxPoseResults()]
                results[0].boxes = OnnxPoseResults.boxes(filtered_boxes, filtered_scores, filtered_classes)
                results[0].keypoints = OnnxPoseResults.keypoints(filtered_keypoints)
                
                return results
                
            else:  # detect
                # Process detection outputs
                # Extract boxes, scores, classes from outputs
                # Output format depends on model, but typically:
                # - outputs[0]: [batch, num_boxes, 5+num_classes] where 5 = x, y, w, h, conf
                detections = outputs[0][0]  # First batch
                
                # Filter by confidence threshold
                mask = detections[:, 4] > threshold
                filtered_dets = detections[mask]
                
                if len(filtered_dets) == 0:
                    # No detections, return empty results
                    class OnnxEmptyResults:
                        class boxes:
                            def __init__(self):
                                self.xyxy = torch.tensor([])
                                self.conf = torch.tensor([])
                                self.cls = torch.tensor([])
                    
                    results = [OnnxEmptyResults()]
                    results[0].boxes = OnnxEmptyResults.boxes()
                    return results
                
                # Extract boxes (convert xywh to xyxy), scores, classes
                boxes = filtered_dets[:, :4]  # xywh format
                scores = filtered_dets[:, 4]
                
                # Get class with highest confidence for each detection
                class_scores = filtered_dets[:, 5:]
                classes = np.argmax(class_scores, axis=1)
                
                # Convert xywh to xyxy format
                x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                xyxy = np.column_stack((x1, y1, x2, y2))
                
                # Create a results object similar to YOLOv8
                class OnnxDetectResults:
                    class boxes:
                        def __init__(self, boxes, scores, classes):
                            self.xyxy = torch.tensor(boxes)
                            self.conf = torch.tensor(scores)
                            self.cls = torch.tensor(classes)
                
                results = [OnnxDetectResults()]
                results[0].boxes = OnnxDetectResults.boxes(xyxy, scores, classes)
                
                return results
                
        except Exception as e:
            print(f"Error running ONNX inference: {e}")
            return None
    
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
        if not self.enable_state_detection:
            return "Unknown", 0.0
            
        # Lazy load state classifier if needed
        if not self._lazy_load_model("state_classifier"):
            return "Unknown", 0.0
        
        # Resize plate image for state classifier
        plate_resized = cv2.resize(plate_image, self.state_classifier_resolution)
        
        try:
            if self._using_onnx:
                # Using ONNX Runtime for inference
                results = self._run_onnx_inference(self.state_classifier_model, plate_resized, 
                                                 task='classify', threshold=self.state_classifier_confidence)
                if results is None:
                    return "Unknown", 0.0
                    
                result = results[0]
            else:
                # Using PyTorch/YOLOv8 for inference
                result = self.state_classifier_model(plate_resized, conf=self.state_classifier_confidence, verbose=False)[0]
            
            # Get the predicted class and confidence
            if hasattr(result, 'probs') and hasattr(result.probs, 'top1'):
                state_idx = int(result.probs.top1)
                confidence = float(result.probs.top1conf.item())
                
                # Convert class index to state name
                if self._using_onnx:
                    state_name = self.model_names.get("state_classifier", {}).get(str(state_idx), f"state_{state_idx}")
                else:
                    state_names = self.state_classifier_model.names
                    state_name = state_names[state_idx]
                
                return state_name, confidence
        except Exception as e:
            print(f"Error classifying state: {e}")
        
        return "Unknown", 0.0
    
    def detect_characters(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters in the license plate image.
        """
        if self.char_detector_model is None:
            return []
            
        # Resize plate image for character detector
        plate_resized = cv2.resize(plate_image, self.char_detector_resolution)
        
        try:
            if self._using_onnx:
                # Using ONNX Runtime for inference
                results = self._run_onnx_inference(self.char_detector_model, plate_resized, 
                                                 task='detect', threshold=self.char_detector_confidence)
                if results is None:
                    return []
                results = results[0]
            else:
                # Using PyTorch/YOLOv8 for inference
                results = self.char_detector_model.predict(plate_resized, conf=self.char_detector_confidence, verbose=False)[0]
        except Exception as e:
            print(f"Error detecting characters: {e}")
            return []
        
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
        if char_image.size == 0 or self.char_classifier_model is None:
            return [("?", 0.0)]
        
        # Resize character image for classifier
        char_resized = cv2.resize(char_image, self.char_classifier_resolution)
        
        try:
            if self._using_onnx:
                # Using ONNX Runtime for inference
                results = self._run_onnx_inference(self.char_classifier_model, char_resized, 
                                                 task='classify', threshold=self.char_classifier_confidence)
                if results is None:
                    return [("?", 0.0)]
                results = results[0]
            else:
                # Using PyTorch/YOLOv8 for inference
                results = self.char_classifier_model.predict(char_resized, conf=self.char_classifier_confidence, verbose=False)[0]
        except Exception as e:
            print(f"Error classifying character: {e}")
            return [("?", 0.0)]
        
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
                    if self._using_onnx:
                        char_names = self.model_names.get("char_classifier", {})
                        for i in range(len(values)):
                            idx = int(indices[i].item())
                            conf = float(values[i].item())
                            
                            # Only include predictions with confidence > 0.02
                            if conf >= 0.02:
                                character = char_names.get(str(idx), f"{idx}")
                                top_predictions.append((character, conf))
                    else:
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
                        
                        if self._using_onnx:
                            char_names = self.model_names.get("char_classifier", {})
                            character = char_names.get(str(idx), f"{idx}")
                        else:
                            char_names = self.char_classifier_model.names
                            character = char_names[idx]
                        top_predictions.append((character, conf))
            elif hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                # Fallback to top1 if data attribute not available
                idx = int(probs.top1)
                conf = float(probs.top1conf.item())
                
                if self._using_onnx:
                    char_names = self.model_names.get("char_classifier", {})
                    character = char_names.get(str(idx), f"{idx}")
                else:
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
    
    def process_plates_batch(self, image: np.ndarray, plates: List[Dict[str, Any]], is_day_plates: bool, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Process multiple plates in batch for better efficiency
        
        Args:
            image: Original image
            plates: List of plates to process
            is_day_plates: Whether these are day plates
            threshold: Confidence threshold
            
        Returns:
            List of processed plate results
        """
        if not plates:
            return []
        
        # Step 1: Extract all plate images for batch processing
        plate_images = []
        for plate_info in plates:
            try:
                plate_corners = plate_info["corners"]
                plate_image = self.four_point_transform(image, plate_corners)
                plate_images.append(plate_image)
            except Exception as e:
                print(f"Error extracting plate image: {e}")
                # Create a blank fallback image
                plate_images.append(np.zeros((100, 300, 3), dtype=np.uint8))
        
        # Step 2: For day plates, batch process state classification if enabled
        state_results = []
        if is_day_plates and self.enable_state_detection and self.optimization_level < 3:
            # Lazy load state classifier if needed
            if self._lazy_load_model("state_classifier"):
                try:
                    # Prepare batch of resized plate images for state classifier
                    batch_state_images = [cv2.resize(img, self.state_classifier_resolution) for img in plate_images]
                    
                    if self._using_onnx:
                        # Process each image individually for ONNX (could be optimized with batched ONNX)
                        for img in batch_state_images:
                            results = self._run_onnx_inference(self.state_classifier_model, img, 
                                                             task='classify', 
                                                             threshold=self.state_classifier_confidence)
                            
                            if results is None or not hasattr(results[0].probs, 'top1'):
                                state_results.append(("Unknown", 0.0))
                                continue
                                
                            state_idx = int(results[0].probs.top1)
                            confidence = float(results[0].probs.top1conf.item())
                            state_name = self.model_names.get("state_classifier", {}).get(str(state_idx), f"state_{state_idx}")
                            state_results.append((state_name, confidence))
                    else:
                        # Run state classifier on batch
                        state_batch_results = self.state_classifier_model(
                            batch_state_images, 
                            conf=self.state_classifier_confidence, 
                            verbose=False
                        )
                        
                        # Process state classification results
                        for result in state_batch_results:
                            if hasattr(result, 'probs') and hasattr(result.probs, 'top1'):
                                state_idx = int(result.probs.top1)
                                confidence = float(result.probs.top1conf.item())
                                state_names = self.state_classifier_model.names
                                state_name = state_names[state_idx]
                                state_results.append((state_name, confidence))
                            else:
                                state_results.append(("Unknown", 0.0))
                except Exception as e:
                    print(f"Error in batch state classification: {e}")
                    # Fill with default values
                    state_results = [("Unknown", 0.0) for _ in plates]
            else:
                # Fill with default values if model not available
                state_results = [("Unknown", 0.0) for _ in plates]
        
        # Step 3: Batch process character detection
        char_results = []
        try:
            # Process each plate individually for character detection
            # (Character detection doesn't benefit much from batching due to different plate sizes)
            futures = []
            for plate_img in plate_images:
                future = self._executor.submit(self.detect_characters, plate_img)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                char_results.append(future.result())
        except Exception as e:
            print(f"Error in batch character detection: {e}")
            # Fill with empty results
            char_results = [[] for _ in plates]
        
        # Step 4: Process the character results and construct final results
        processed_results = []
        for i, plate_info in enumerate(plates):
            if i >= len(char_results):
                continue
                
            plate_chars = char_results[i] if i < len(char_results) else []
            
            # Initialize result dictionary
            plate_result = {
                "corners": plate_info["corners"],
                "is_day_plate": is_day_plates,
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
            
            # Add state for day plates if available
            if is_day_plates and i < len(state_results):
                state, state_confidence = state_results[i]
                plate_result["state"] = state
                plate_result["state_confidence"] = state_confidence
            
            # Process characters more efficiently
            organized_chars = self.organize_characters(plate_chars)
            
            # Prepare batches of character images for classification
            char_images = []
            char_indices = []
            
            for char_idx, char_info in enumerate(organized_chars):
                if "image" in char_info and char_info["image"] is not None:
                    # Resize for classifier
                    char_img = cv2.resize(char_info["image"], self.char_classifier_resolution)
                    char_images.append(char_img)
                    char_indices.append(char_idx)
            
            # Batch classify characters if there are any
            char_classifications = []
            if char_images:
                try:
                    if self._using_onnx:
                        # Process each image individually for ONNX
                        for img in char_images:
                            results = self._run_onnx_inference(self.char_classifier_model, img, 
                                                           task='classify', 
                                                           threshold=self.char_classifier_confidence)
                            
                            if results is None:
                                char_classifications.append([("?", 0.0)])
                                continue
                                
                            # Extract top predictions
                            top_predictions = []
                            probs = results[0].probs
                            if hasattr(probs, 'data'):
                                probs_tensor = probs.data
                                values, indices = torch.topk(probs_tensor, min(5, len(probs_tensor)))
                                
                                # Get character names
                                char_names = self.model_names.get("char_classifier", {})
                                for j in range(len(values)):
                                    idx = int(indices[j].item())
                                    conf = float(values[j].item())
                                    
                                    if conf >= 0.02:
                                        character = char_names.get(str(idx), str(idx))
                                        top_predictions.append((character, conf))
                            
                            # If no predictions, add placeholder
                            if not top_predictions:
                                top_predictions.append(("?", 0.0))
                                
                            char_classifications.append(top_predictions)
                    else:
                        # Use PyTorch batch inference
                        batch_results = self.char_classifier_model(
                            char_images, 
                            conf=self.char_classifier_confidence, 
                            verbose=False
                        )
                        
                        for result in batch_results:
                            top_predictions = []
                            # Extract top predictions if available
                            if hasattr(result, 'probs'):
                                probs = result.probs
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
                            
                            # If no predictions were found
                            if not top_predictions:
                                top_predictions.append(("?", 0.0))
                                
                            char_classifications.append(top_predictions)
                except Exception as e:
                    print(f"Error in batch character classification: {e}")
                    # Fill with defaults
                    char_classifications = [[("?", 0.0)] for _ in char_images]
            
            # Map classifications back to the original characters
            char_results = []
            for j, char_info in enumerate(organized_chars):
                if j in char_indices:
                    idx = char_indices.index(j)
                    if idx < len(char_classifications):
                        top_chars = char_classifications[idx]
                    else:
                        top_chars = [("?", 0.0)]
                else:
                    top_chars = [("?", 0.0)]
                
                char_results.append({
                    "char": top_chars[0][0] if top_chars else "?",
                    "confidence": top_chars[0][1] if top_chars else 0.0,
                    "top_predictions": top_chars,
                    "box": char_info["box"] if "box" in char_info else [0, 0, 0, 0]
                })
            
            # Construct the license number
            license_number = ''.join(cr["char"] for cr in char_results)
            avg_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
            
            # Generate alternative plate combinations
            top_plates = self._generate_top_plates(char_results, max_combinations=5 if self.optimization_level < 3 else 3)
            
            plate_result["characters"] = char_results
            plate_result["plate"] = license_number
            plate_result["license_number"] = license_number
            plate_result["confidence"] = avg_confidence
            plate_result["top_plates"] = top_plates
            
            # Store plate dimensions for debugging
            if i < len(plate_images) and plate_images[i] is not None:
                h, w = plate_images[i].shape[:2]
                plate_result["dimensions"] = {"width": w, "height": h, "actual_ratio": w/h if h > 0 else 0}
            
            # Only include plates with confidence above threshold
            if avg_confidence >= threshold:
                processed_results.append(plate_result)
        
        return processed_results
    
    def process_plate(self, image: np.ndarray, plate_info: Dict[str, Any], is_day_plate: bool, 
                      skip_state_detection: bool = False, skip_character_recognition: bool = False) -> Dict[str, Any]:
        """
        Process a single license plate with optional optimization flags.
        
        Args:
            image: Original image
            plate_info: Plate detection information
            is_day_plate: Whether this is a day plate
            skip_state_detection: Whether to skip state detection for speed
            skip_character_recognition: Whether to skip character recognition for speed
        
        Returns:
            Plate result dictionary
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
            "confidence": plate_info.get("confidence", 0.0),  # Use detection confidence as initial value
            "aspect_ratio": self.plate_aspect_ratio  # Store used aspect ratio for reference
        }
        
        # Include original corners if available
        if "original_corners" in plate_info:
            plate_result["original_corners"] = plate_info["original_corners"]
        
        # Include detection box if available
        if "detection_box" in plate_info:
            plate_result["detection_box"] = plate_info["detection_box"]
        
        # If it's a day plate, also determine the state (unless skipping for speed)
        if is_day_plate and self.enable_state_detection and not skip_state_detection:
            if self._lazy_load_model("state_classifier"):
                state, state_confidence = self.classify_state(plate_image)
                plate_result["state"] = state
                plate_result["state_confidence"] = state_confidence
        
        # Skip character recognition if requested (for speed in high optimization mode)
        if skip_character_recognition:
            # In this case, we'll just use a placeholder license number based on the plate shape
            h, w = plate_image.shape[:2]
            # Estimate how many characters would fit based on aspect ratio
            char_count = max(4, min(8, int(w / h * 1.8)))
            plate_result["plate"] = "?" * char_count
            plate_result["license_number"] = "?" * char_count
            
            # Store plate dimensions for debugging
            if plate_image is not None:
                h, w = plate_image.shape[:2]
                plate_result["dimensions"] = {"width": w, "height": h, "actual_ratio": w/h if h > 0 else 0}
                
            return plate_result
        
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
        
        # Generate alternative plate combinations (fewer for high optimization)
        max_combinations = 5
        if self.optimization_level >= 2:
            max_combinations = 3
        if self.optimization_level >= 3:
            max_combinations = 2
            
        top_plates = self._generate_top_plates(char_results, max_combinations=max_combinations)
        
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
    
    def detect_vehicle(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the image.
        """
        if not self.enable_vehicle_detection:
            return []
            
        # Lazy load vehicle detector if needed
        if not self._lazy_load_model("vehicle_detector"):
            return []
        
        # Resize image for vehicle detector
        img_resized = cv2.resize(image, self.vehicle_detector_resolution)
        
        try:
            if self._using_onnx:
                # Using ONNX Runtime for inference
                results = self._run_onnx_inference(self.vehicle_detector_model, img_resized, 
                                                 task='detect', threshold=self.vehicle_detector_confidence)
                if results is None:
                    return []
                    
                result = results[0]
            else:
                # Using PyTorch/YOLOv8 for inference
                result = self.vehicle_detector_model(img_resized, conf=self.vehicle_detector_confidence, verbose=False)[0]
        except Exception as e:
            print(f"Error detecting vehicles: {e}")
            return []
        
        # Process the results to extract vehicle bounding boxes
        vehicles = []
        
        if hasattr(result, 'boxes') and hasattr(result.boxes, 'xyxy'):
            for i, box in enumerate(result.boxes.xyxy):
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
                
                confidence = float(result.boxes.conf[i].item()) if hasattr(result.boxes, 'conf') else 0.0
                
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
        if not self.enable_vehicle_detection or vehicle_image.size == 0:
            return "Unknown", "Unknown", 0.0
        
        # Lazy load vehicle classifier if needed
        if not self._lazy_load_model("vehicle_classifier"):
            return "Unknown", "Unknown", 0.0
        
        # Resize vehicle image for classifier
        vehicle_resized = cv2.resize(vehicle_image, self.vehicle_classifier_resolution)
        
        try:
            if self._using_onnx:
                # Using ONNX Runtime for inference
                results = self._run_onnx_inference(self.vehicle_classifier_model, vehicle_resized, 
                                                 task='classify', threshold=self.vehicle_classifier_confidence)
                if results is None:
                    return "Unknown", "Unknown", 0.0
                    
                result = results[0]
            else:
                # Using PyTorch/YOLOv8 for inference
                result = self.vehicle_classifier_model(vehicle_resized, conf=self.vehicle_classifier_confidence, verbose=False)[0]
        except Exception as e:
            print(f"Error classifying vehicle: {e}")
            return "Unknown", "Unknown", 0.0
        
        # Get the predicted class and confidence
        if hasattr(result, 'probs') and hasattr(result.probs, 'top1'):
            vehicle_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf.item())
            
            # Convert class index to make and model
            if self._using_onnx:
                vehicle_names = self.model_names.get("vehicle_classifier", {})
                make_model = vehicle_names.get(str(vehicle_idx), f"Unknown_{vehicle_idx}")
            else:
                vehicle_names = self.vehicle_classifier_model.names
                make_model = vehicle_names[vehicle_idx]
            
            # Split make and model (assuming format "Make_Model")
            make, model = make_model.split("_", 1) if "_" in make_model else (make_model, "Unknown")
            
            return make, model, confidence
        
        return "Unknown", "Unknown", 0.0
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image to detect and recognize license plates, vehicle make/model.
        
        Args:
            image: Input image
            
        Returns:
            Results dictionary
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
        
        # Fast-path for no plates detected
        day_plates_count = len(plate_detection["day_plates"])
        night_plates_count = len(plate_detection["night_plates"])
        
        if day_plates_count + night_plates_count == 0:
            return results
        
        # Determine optimal processing strategy based on plate count and optimization level
        use_batch_processing = self.enable_batch_processing and (day_plates_count > 1 or night_plates_count > 1)
        
        # Process plates using optimal strategy
        if use_batch_processing:
            # Process day plates in batch if there are multiple
            if day_plates_count > 0:
                day_results = self.process_plates_batch(
                    image_copy, 
                    plate_detection["day_plates"], 
                    True,
                    threshold=self.plate_detector_confidence
                )
                results["day_plates"] = day_results
                
            # Process night plates in batch if there are multiple
            if night_plates_count > 0:
                night_results = self.process_plates_batch(
                    image_copy, 
                    plate_detection["night_plates"], 
                    False,
                    threshold=self.plate_detector_confidence
                )
                results["night_plates"] = night_results
        else:
            # Process plates individually using multi-threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_count) as executor:
                # Submit day plate processing tasks
                day_plate_futures = [
                    executor.submit(
                        self.process_plate, 
                        image_copy, 
                        plate, 
                        True,
                        # Skip state detection at highest optimization level
                        skip_state_detection=(self.optimization_level >= 3),
                        # Skip character recognition for large numbers of plates at highest optimization
                        skip_character_recognition=(day_plates_count > 5 and self.optimization_level >= 3)
                    )
                    for plate in plate_detection["day_plates"]
                ]
                
                # Submit night plate processing tasks
                night_plate_futures = [
                    executor.submit(
                        self.process_plate, 
                        image_copy, 
                        plate, 
                        False,
                        # Skip state detection (night plates don't have states anyway)
                        skip_state_detection=True,
                        # Skip character recognition for large numbers of plates at highest optimization
                        skip_character_recognition=(night_plates_count > 5 and self.optimization_level >= 3)
                    )
                    for plate in plate_detection["night_plates"]
                ]
                
                # Collect day plate results
                for future in concurrent.futures.as_completed(day_plate_futures):
                    try:
                        plate_result = future.result()
                        # Only include plates with sufficient confidence
                        if plate_result["confidence"] >= self.plate_detector_confidence:
                            results["day_plates"].append(plate_result)
                    except Exception as e:
                        print(f"Error processing day plate: {e}")
                
                # Collect night plate results
                for future in concurrent.futures.as_completed(night_plate_futures):
                    try:
                        plate_result = future.result()
                        # Only include plates with sufficient confidence
                        if plate_result["confidence"] >= self.plate_detector_confidence:
                            results["night_plates"].append(plate_result)
                    except Exception as e:
                        print(f"Error processing night plate: {e}")
        
        # Skip vehicle detection in higher optimization levels or if no day plates detected
        skip_vehicle_detection = (self.optimization_level >= 2) or not results["day_plates"]
        
        # If day plates were detected and vehicle detection is enabled,
        # detect and classify vehicles (unless skipping for optimization)
        if results["day_plates"] and self.enable_vehicle_detection and not skip_vehicle_detection:
            # Lazy load vehicle detector if needed
            if self._lazy_load_model("vehicle_detector"):
                vehicles = self.detect_vehicle(image_copy)
                
                # Skip vehicle classification at highest optimization level
                skip_classification = (self.optimization_level >= 3)
                
                vehicle_results = []
                for vehicle_info in vehicles:
                    try:
                        if skip_classification:
                            # Skip classification, just use detection
                            vehicle_results.append({
                                "box": vehicle_info["box"],
                                "make": "Unknown",
                                "model": "Unknown",
                                "confidence": vehicle_info["confidence"]
                            })
                        else:
                            # Lazy load vehicle classifier if needed
                            if self._lazy_load_model("vehicle_classifier"):
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
    corner_dilation_pixels: int = 5,
    # Add ONNX options
    enable_onnx: bool = False,
    onnx_provider: str = "auto"
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
        enable_onnx: Whether to use ONNX models instead of PyTorch
        onnx_provider: Which ONNX execution provider to use
    
    Returns:
        Dictionary containing the ALPR results
    """
    try:
        # Determine if we should use ONNX models
        if enable_onnx:
            try:
                import onnxruntime as ort
                print(f"Available ONNX providers: {ort.get_available_providers()}")
                
                # Convert models to ONNX if needed (this would typically be done ahead of time)
                # For this function, we'll assume the ONNX models already exist
                
                # Check if ONNX models exist at the same location but with .onnx extension
                onnx_models_exist = all([
                    os.path.exists(path.replace('.pt', '.onnx')) for path in 
                    [plate_detector_path, state_classifier_path, char_detector_path, 
                     char_classifier_path, vehicle_detector_path, vehicle_classifier_path]
                    if path and enable_state_detection
                ])
                
                if not onnx_models_exist:
                    print("ONNX models not found. Using PyTorch models.")
                    enable_onnx = False
                else:
                    # Update paths to use ONNX models
                    if plate_detector_path and plate_detector_path.endswith('.pt'):
                        plate_detector_path = plate_detector_path.replace('.pt', '.onnx')
                    if state_classifier_path and state_classifier_path.endswith('.pt'):
                        state_classifier_path = state_classifier_path.replace('.pt', '.onnx')
                    if char_detector_path and char_detector_path.endswith('.pt'):
                        char_detector_path = char_detector_path.replace('.pt', '.onnx')
                    if char_classifier_path and char_classifier_path.endswith('.pt'):
                        char_classifier_path = char_classifier_path.replace('.pt', '.onnx')
                    if vehicle_detector_path and vehicle_detector_path.endswith('.pt'):
                        vehicle_detector_path = vehicle_detector_path.replace('.pt', '.onnx')
                    if vehicle_classifier_path and vehicle_classifier_path.endswith('.pt'):
                        vehicle_classifier_path = vehicle_classifier_path.replace('.pt', '.onnx')
            except ImportError:
                print("ONNX Runtime not installed. Using PyTorch models.")
                enable_onnx = False
        
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
            device="cuda" if torch.cuda.is_available() else "cpu",
            plate_detector_confidence=plate_detector_confidence,
            state_classifier_confidence=state_classifier_confidence,
            char_detector_confidence=char_detector_confidence,
            char_classifier_confidence=char_classifier_confidence,
            vehicle_detector_confidence=vehicle_detector_confidence,
            vehicle_classifier_confidence=vehicle_classifier_confidence,
            plate_aspect_ratio=plate_aspect_ratio,
            corner_dilation_pixels=corner_dilation_pixels,
            enable_onnx=enable_onnx,
            onnx_provider=onnx_provider
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

# Visualization functions (same as before - these have not been modified for ONNX support)
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
        plate_aspect_ratio=4.0,  # US license plates typically have 4:1 width:height ratio
        corner_dilation_pixels=15,  # 15 pixel corner dilation
        enable_onnx=False,  # Set to True to use ONNX models if available
        onnx_provider="auto"  # Auto-select the best available provider
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
        corner_dilation_pixels=15,
        enable_onnx=False
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
    print(f"  - Using ONNX: {config.get('using_onnx', False)}")
    if config.get('using_onnx', False):
        print(f"  - ONNX Provider: {config.get('onnx_provider', 'Unknown')}")
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