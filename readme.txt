# Automatic License Plate Recognition (ALPR) Module for CodeProject.AI Server

This is an Automatic License Plate Recognition (ALPR) module for [CodeProject.AI Server](https://www.codeproject.com/Articles/5322557/CodeProject-AI-Server-AI-the-easy-way). The module can detect license plates in images, recognize characters, identify states, and detect vehicles with make/model classifications.

## Features

- License plate detection for both day and night plates
- Character recognition on license plates
- State classification for license plates (when enabled)
- Vehicle detection and make/model classification (when enabled)
- Support for GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon)
- Configurable confidence thresholds and plate aspect ratios

## API Endpoint

The module provides a single API endpoint with different operation modes:

```
POST /v1/vision/alpr
```

Parameters:
- `operation`: The type of analysis to perform
  - `plate`: Detect only license plates
  - `vehicle`: Detect only vehicles and their make/model
  - `full`: Complete analysis (both license plates and vehicles)
- `min_confidence`: Minimum confidence threshold for detections (0.0 to 1.0)

## Technical Details

This module uses YOLOv8 models for various detection and recognition tasks:

- **plate_detector.pt**: Detects license plates in the image
- **state_classifier.pt**: Identifies the US state for a license plate
- **char_detector.pt**: Detects individual characters on the license plate
- **char_classifier.pt**: Recognizes each character (OCR)
- **vehicle_detector.pt**: Detects vehicles in the image
- **vehicle_classifier.pt**: Identifies vehicle make and model

## Configuration

The module supports several configuration options through environment variables:

- `ENABLE_STATE_DETECTION`: Enable/disable state identification
- `ENABLE_VEHICLE_DETECTION`: Enable/disable vehicle detection
- `PLATE_ASPECT_RATIO`: Set a specific aspect ratio for license plates
- `CORNER_DILATION_PIXELS`: Configure corner dilation for license plate extraction
- Various confidence thresholds for different detection components

## Requirements

- Python 3.8+ 
- PyTorch
- OpenCV
- Ultralytics YOLOv8
