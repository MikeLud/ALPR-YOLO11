# Import general libraries
import os
import sys
import time
import json

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

class ALPR_adapter(ModuleRunner):

    def __init__(self):
        super().__init__()
        self.opts = Options()
        self.models_last_checked = None

        # These will be adjusted based on the hardware / packages found
        self.use_CUDA     = self.opts.use_CUDA
        self.use_MPS      = self.opts.use_MPS
        self.use_DirectML = self.opts.use_DirectML

        if self.use_CUDA and self.half_precision == 'enable' and \
           not self.system_info.hasTorchHalfPrecision:
            self.half_precision = 'disable'

        # Initialize the ALPR system once for use across requests
        self.alpr_system = None

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
            
            self.alpr_system = ALPRSystem(
                plate_detector_path=os.path.join(self.opts.models_dir, "plate_detector.pt"),
                state_classifier_path=os.path.join(self.opts.models_dir, "state_classifier.pt"),
                char_detector_path=os.path.join(self.opts.models_dir, "char_detector.pt"),
                char_classifier_path=os.path.join(self.opts.models_dir, "char_classifier.pt"),
                vehicle_detector_path=os.path.join(self.opts.models_dir, "vehicle_detector.pt"),
                vehicle_classifier_path=os.path.join(self.opts.models_dir, "vehicle_classifier.pt"),
                enable_state_detection=self.opts.enable_state_detection,
                enable_vehicle_detection=self.opts.enable_vehicle_detection,
                device="cuda" if self.use_CUDA else "cpu",
                plate_detector_confidence=float(self.opts.plate_detector_confidence),
                state_classifier_confidence=float(self.opts.state_classifier_confidence),
                char_detector_confidence=float(self.opts.char_detector_confidence),
                char_classifier_confidence=float(self.opts.char_classifier_confidence),
                vehicle_detector_confidence=float(self.opts.vehicle_detector_confidence),
                vehicle_classifier_confidence=float(self.opts.vehicle_classifier_confidence),
                plate_aspect_ratio=float(self.opts.plate_aspect_ratio) if self.opts.plate_aspect_ratio else None,
                corner_dilation_pixels=int(self.opts.corner_dilation_pixels)
            )
            
            self.log(LogMethod.Info | LogMethod.Server,
            {
                "filename": __file__,
                "loglevel": "information", 
                "method": sys._getframe().f_code.co_name,
                "message": f"ALPR system initialized successfully"
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

            with open("log.txt", "a") as text_file:
                json.dump(response, text_file, indent=2)
                text_file.write("\n\n")
            

        except Exception as ex:
            response = { "success": False, "error": f"Unknown command {data.command}" }
            self.report_error(None, __file__, f"Unknown command {data.command}")

        return response

    def detect_license_plate(self, img, threshold):
        """
        Detect license plates in an image
        """
        if self.alpr_system is None:
            return {"success": False, "error": "ALPR system not initialized"}
        
        start_process_time = time.perf_counter()
        
        # Convert PIL Image to numpy array for OpenCV
        image_np = np.array(img)
        # Convert RGB to BGR (OpenCV format)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Color image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        try:
            start_inference_time = time.perf_counter()
            
            # Process the image to find license plates
            plate_detection = self.alpr_system.detect_license_plates(image_np)
            
            # Process each plate
            results = {
                "day_plates": [],
                "night_plates": []
            }
            
            for plate_type in ["day_plates", "night_plates"]:
                for plate_info in plate_detection[plate_type]:
                    if plate_info["confidence"] >= threshold:
                        plate_result = self.alpr_system.process_plate(image_np, plate_info, plate_type == "day_plates")
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
                
            return {
                "success": True,
                "processMs": int((time.perf_counter() - start_process_time) * 1000),
                "inferenceMs": inferenceMs,
                "predictions": plates,
                "message": message,
                "count": len(plates)
            }
            
        except Exception as ex:
            self.report_error(ex, __file__, f"Error detecting license plates: {str(ex)}")
            return {"success": False, "error": f"Error detecting license plates: {str(ex)}"}
    
    def status(self) -> JSON:
        statusData = super().status()
        statusData["platesDetected"] = self._plates_detected
        statusData["histogram"] = self._histogram
        return statusData

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
