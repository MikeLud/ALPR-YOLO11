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

        # -------------------------------------------------------------------------
        # dump the important variables

        if self._show_env_variables:
            print(f"Debug: APPDIR:      {self.app_dir}")
            print(f"Debug: MODELS_DIR:  {self.models_dir}")
            print(f"Debug: USE_CUDA:    {self.use_CUDA}")
            print(f"Debug: ENABLE_STATE_DETECTION: {self.enable_state_detection}")
            print(f"Debug: ENABLE_VEHICLE_DETECTION: {self.enable_vehicle_detection}")
