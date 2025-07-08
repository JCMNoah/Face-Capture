# Configuration Guide

This guide explains all configuration options for the Face Overlay system and how to optimize them for your setup.

## Quick Configuration

```bash
python main.py config
```

## Configuration File: `face_overlay_config.json`

### Basic Settings

#### `game_region`
**What it does**: Defines the screen area to capture for character detection.
```json
"game_region": {
    "top": 200,
    "left": 700, 
    "width": 1000,
    "height": 1000
}
```
- **top/left**: Position of capture area on screen
- **width/height**: Size of capture area
- **Tip**: Smaller regions = better performance

#### `fps`
**What it does**: Target framerate for the virtual camera output.
```json
"fps": 60
```
- **Range**: 15-60
- **Recommended**: 30 for stability, 60 for smoothness
- **Note**: Higher FPS requires more processing power

#### `webcam_face_size`
**What it does**: Size of your face overlay in pixels.
```json
"webcam_face_size": 150
```
- **Range**: 50-500
- **Recommended**: 150-250
- **Tip**: Larger faces need more processing

### Detection Settings

#### `face_threshold`
**What it does**: Sensitivity for ML model character detection.
```json
"face_threshold": 0.5
```
- **Range**: 0.0-1.0
- **Lower values**: More sensitive (detects more faces)
- **Higher values**: Less sensitive (fewer false positives)
- **Recommended**: 0.3-0.7

#### `face_detection_confidence`
**What it does**: Confidence threshold for MediaPipe face detection.
```json
"face_detection_confidence": 0.7
```
- **Range**: 0.1-1.0
- **Lower values**: Detects more faces (including unclear ones)
- **Higher values**: Only detects clear, obvious faces
- **Recommended**: 0.5-0.8

#### `use_hybrid_detection`
**What it does**: Uses both ML model and MediaPipe for better accuracy.
```json
"use_hybrid_detection": true
```
- **true**: Better accuracy, slightly slower
- **false**: Faster, ML model only
- **Recommended**: true

#### `mediapipe_confidence`
**What it does**: Confidence threshold for MediaPipe in hybrid mode.
```json
"mediapipe_confidence": 0.3
```
- **Range**: 0.1-1.0
- **Lower values**: More sensitive MediaPipe detection
- **Recommended**: 0.2-0.4

#### `detection_smoothing`
**What it does**: Number of frames to average for stable detection.
```json
"detection_smoothing": 3
```
- **Range**: 1-10
- **Higher values**: More stable, less flickering
- **Lower values**: More responsive, may flicker
- **Recommended**: 3-5

### Performance Settings

#### `optimize_performance`
**What it does**: Enables various performance optimizations.
```json
"optimize_performance": true
```
- **true**: Better performance, may reduce quality slightly
- **false**: Best quality, may be slower
- **Recommended**: true

#### `prioritize_camera_smoothness`
**What it does**: Prioritizes smooth camera over detection frequency.
```json
"prioritize_camera_smoothness": false
```
- **true**: Smooth 60 FPS camera, detection runs less often
- **false**: Detection runs frequently, may impact camera smoothness
- **Use when**: Camera feels laggy or stuttery

#### `skip_frames`
**What it does**: Process ML detection every N frames (when smoothness = false).
```json
"skip_frames": 1
```
- **1**: Every frame (best quality, slowest)
- **2**: Every other frame (good balance)
- **3-5**: Every 3rd-5th frame (faster, less accurate)

#### `detection_interval`
**What it does**: Seconds between ML detections (when smoothness = true).
```json
"detection_interval": 0.1
```
- **Range**: 0.05-1.0
- **0.05**: Very frequent detection (20 times per second)
- **0.1**: Frequent detection (10 times per second)
- **0.5**: Moderate detection (2 times per second)

#### `cache_detection_time`
**What it does**: How long to use cached detection results.
```json
"cache_detection_time": 2.0
```
- **Range**: 1.0-5.0
- **Shorter**: More responsive to changes
- **Longer**: Better performance, may miss quick changes

#### `webcam_fps`
**What it does**: Webcam capture framerate (separate from output FPS).
```json
"webcam_fps": 30
```
- **Range**: 15-60
- **Recommended**: 30 (good balance)
- **Note**: Lower = better performance

### Display Settings

#### `mirror_webcam`
**What it does**: Horizontally flips your webcam feed.
```json
"mirror_webcam": true
```
- **true**: Natural mirror effect
- **false**: Non-mirrored (may look backwards)

#### `show_debug`
**What it does**: Shows performance and detection information.
```json
"show_debug": true
```
- **true**: Helpful for troubleshooting
- **false**: Cleaner output, slightly better performance

#### `disable_game_background`
**What it does**: Shows black background instead of game screen.
```json
"disable_game_background": true
```
- **true**: Face on black background
- **false**: Face overlaid on game screen

#### `use_transparency`
**What it does**: Uses green screen for true transparency in OBS.
```json
"use_transparency": true
```
- **true**: Green background for chroma key
- **false**: Solid background
- **Note**: Use with OBS Chroma Key filter

## Performance Optimization Presets

### Maximum Performance (Smooth Camera)
```json
{
    "fps": 60,
    "prioritize_camera_smoothness": true,
    "detection_interval": 0.2,
    "cache_detection_time": 3.0,
    "webcam_fps": 30,
    "optimize_performance": true,
    "show_debug": false
}
```

### Balanced Performance
```json
{
    "fps": 30,
    "prioritize_camera_smoothness": false,
    "skip_frames": 2,
    "webcam_fps": 30,
    "optimize_performance": true
}
```

### Maximum Quality
```json
{
    "fps": 30,
    "prioritize_camera_smoothness": false,
    "skip_frames": 1,
    "use_hybrid_detection": true,
    "detection_smoothing": 5,
    "optimize_performance": false
}
```

## Troubleshooting

### Camera is Laggy/Stuttery
1. Set `prioritize_camera_smoothness: true`
2. Increase `detection_interval` to 0.2-0.5
3. Lower `webcam_fps` to 20-25
4. Reduce `game_region` size

### Detection is Inaccurate
1. Lower `face_threshold` to 0.3-0.4
2. Enable `use_hybrid_detection: true`
3. Lower `mediapipe_confidence` to 0.2-0.3
4. Collect more training data

### Face Position is Wrong
1. Ensure `use_hybrid_detection: true`
2. Adjust `face_detection_confidence`
3. Check game region covers character faces properly

### System is Too Slow
1. Enable `optimize_performance: true`
2. Increase `skip_frames` to 3-5
3. Lower `fps` to 30
4. Reduce `webcam_face_size`
5. Make `game_region` smaller

## Quick Setup Commands

```bash
# Configure for smooth camera
python main.py config
# Option 16: true (prioritize camera smoothness)

# Configure for maximum performance  
python main.py config
# Option 15: true (optimize performance)
# Option 13: 3 (skip frames)

# Configure for best quality
python main.py config
# Option 9: true (hybrid detection)
# Option 11: 5 (detection smoothing)
```
