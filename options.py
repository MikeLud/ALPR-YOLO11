import os
from codeproject_ai_sdk import ModuleOptions

class Options:

    def __init__(self):
        # -------------------------------------------------------------------------
        # Setup values

        self._show_env_variables = True

        self.app_dir            = os.path.normpath(ModuleOptions.getEnvVariable("APPDIR", os.getcwd()))
        self.models_dir         = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{self.app_dir}/models"))
        
        # ALPR specific settings
        self.enable_state_detection    = ModuleOptions.getEnvVariable("ENABLE_STATE_DETECTION", "True").lower() == "true"
        self.enable_vehicle_detection  = ModuleOptions.getEnvVariable("ENABLE_VEHICLE_DETECTION", "True").lower() == "true"
        
        # Confidence thresholds
        self.plate_detector_confidence    = ModuleOptions.getEnvVariable("PLATE_DETECTOR_CONFIDENCE", "0.45")
        self.state_classifier_confidence  = ModuleOptions.getEnvVariable("STATE_CLASSIFIER_CONFIDENCE", "0.45")
        self.char_detector_confidence     = ModuleOptions.getEnvVariable("CHAR_DETECTOR_CONFIDENCE", "0.40")
        self.char_classifier_confidence   = ModuleOptions.getEnvVariable("CHAR_CLASSIFIER_CONFIDENCE", "0.40")
        self.vehicle_detector_confidence  = ModuleOptions.getEnvVariable("VEHICLE_DETECTOR_CONFIDENCE", "0.45")
        self.vehicle_classifier_confidence = ModuleOptions.getEnvVariable("VEHICLE_CLASSIFIER_CONFIDENCE", "0.45")
        
        # License plate aspect ratio and corner dilation
        self.plate_aspect_ratio      = ModuleOptions.getEnvVariable("PLATE_ASPECT_RATIO", "4.0")
        self.corner_dilation_pixels  = ModuleOptions.getEnvVariable("CORNER_DILATION_PIXELS", "5")

        # GPU settings
        self.use_CUDA           = ModuleOptions.getEnvVariable("USE_CUDA", "True").lower() == "true"
        self.use_MPS            = True  # only if available...
        self.use_DirectML       = True  # only if available...
        
        # Performance optimization settings
        self.optimization_level = int(ModuleOptions.getEnvVariable("OPTIMIZATION_LEVEL", "1"))
        self.enable_tensorrt    = ModuleOptions.getEnvVariable("ENABLE_TENSORRT", "False").lower() == "true"
        self.enable_onnx        = ModuleOptions.getEnvVariable("ENABLE_ONNX", "False").lower() == "true"
        self.onnx_provider      = ModuleOptions.getEnvVariable("ONNX_PROVIDER", "auto")
        self.half_precision     = ModuleOptions.getEnvVariable("HALF_PRECISION", "True").lower() == "true"
        self.input_resolution   = int(ModuleOptions.getEnvVariable("INPUT_RESOLUTION", "640"))
        self.enable_caching     = ModuleOptions.getEnvVariable("ENABLE_CACHING", "True").lower() == "true"
        self.cache_size         = int(ModuleOptions.getEnvVariable("CACHE_SIZE", "100"))
        self.enable_adaptive_processing = ModuleOptions.getEnvVariable("ENABLE_ADAPTIVE_PROCESSING", "True").lower() == "true"
        self.enable_batch_processing = ModuleOptions.getEnvVariable("ENABLE_BATCH_PROCESSING", "True").lower() == "true"
        self.max_thread_count   = int(ModuleOptions.getEnvVariable("MAX_THREAD_COUNT", "4"))
        self.onnx_model_path    = os.path.normpath(ModuleOptions.getEnvVariable("ONNX_MODEL_PATH", f"{self.models_dir}/onnx"))

        # -------------------------------------------------------------------------
        # dump the important variables

        if self._show_env_variables:
            print(f"Debug: APPDIR:      {self.app_dir}")
            print(f"Debug: MODELS_DIR:  {self.models_dir}")
            print(f"Debug: USE_CUDA:    {self.use_CUDA}")
            print(f"Debug: ENABLE_STATE_DETECTION: {self.enable_state_detection}")
            print(f"Debug: ENABLE_VEHICLE_DETECTION: {self.enable_vehicle_detection}")
            print(f"Debug: OPTIMIZATION_LEVEL: {self.optimization_level}")
            print(f"Debug: ENABLE_TENSORRT: {self.enable_tensorrt}")
            print(f"Debug: ENABLE_ONNX: {self.enable_onnx}")
            print(f"Debug: ONNX_PROVIDER: {self.onnx_provider}")
            print(f"Debug: HALF_PRECISION: {self.half_precision}")
            print(f"Debug: INPUT_RESOLUTION: {self.input_resolution}")
