# Face Overlay for OBS Streaming

A real-time face detection and overlay system that detects character faces on screen and replaces them with your webcam face for streaming.

## Features

- **Real-time face detection** on game/application screens
- **Webcam face overlay** with automatic positioning
- **Machine learning model** for character face detection
- **Virtual camera output** for OBS/streaming software
- **Interactive data collection** for training
- **Configurable settings** for different games/applications

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install opencv-python numpy mediapipe pyvirtualcam mss tensorflow scikit-learn matplotlib seaborn
   ```

2. **Collect training data:**
   ```bash
   python main.py collect
   ```
   - Select the region where character faces appear
   - Press 'f' when a character face is visible
   - Press 'n' when no character face is visible
   - Collect at least 20-30 samples of each type

3. **Train the model:**
   ```bash
   python main.py train
   ```

4. **Run the face overlay:**
   ```bash
   python main.py run
   ```

5. **Use in OBS:**
   - Add a "Video Capture Device" source
   - Select the virtual camera created by the system

## Detailed Usage

### Data Collection (`python main.py collect`)

The data collection tool helps you create a training dataset:

- **Face samples**: Screenshots when a character face is visible
- **No-face samples**: Screenshots when no character face is visible
- **Interactive selection**: Choose the screen region to monitor
- **Quality matters**: Collect diverse samples (different lighting, angles, characters)

**Tips:**
- Collect 50-100 samples of each type for best results
- Include different characters, lighting conditions, and camera angles
- Make sure the selected region consistently contains the character's face area

### Model Training (`python main.py train`)

Trains a neural network to detect character faces:

- **Input**: Grayscale images (184x184 pixels)
- **Output**: Binary classification (face/no-face)
- **Architecture**: Dense neural network with dropout
- **Validation**: Automatic train/test split with performance metrics

### Face Overlay (`python main.py run`)

The main application that performs real-time face overlay:

1. **Screen capture**: Captures the configured game region
2. **Face detection**: Uses the trained model to detect character faces
3. **Webcam processing**: Detects and crops your face from webcam
4. **Overlay**: Positions your face over the detected character face
5. **Virtual camera**: Outputs the result for streaming software

### Testing (`python main.py test`)

Validates system functionality:

- **Model performance**: Accuracy metrics and confusion matrix
- **Webcam detection**: Test face detection on your webcam
- **Screen capture**: Test face detection on screen content
- **Configuration**: Verify settings are loaded correctly

### Configuration (`python main.py config`)

Adjust system settings:

- **Game region**: Screen area to capture
- **Detection confidence**: Sensitivity of face detection
- **Face threshold**: Model prediction threshold
- **Overlay size**: Size of your face overlay
- **Mirror webcam**: Flip webcam horizontally
- **Debug info**: Show performance and detection info

## File Structure

```
Face Capture/
├── main.py                    # Main application interface
├── facecollector.py          # Data collection tool
├── train_face_model.py       # Model training script
├── face_detector.py          # Real-time face overlay
├── test_system.py           # System validation tests
├── template_match_test.py   # Basic face detection test
├── face_overlay_config.json # Configuration file
├── dataset/                 # Training data
│   ├── face/               # Character face samples
│   └── no_face/           # No character face samples
├── face_detector_model.h5  # Trained model (created after training)
└── README.md              # This file
```

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Run `python main.py train` first to create the model

2. **"Failed to access webcam"**
   - Check if another application is using the webcam
   - Try different camera indices (0, 1, 2, etc.)

3. **Poor face detection accuracy**
   - Collect more training data with diverse samples
   - Adjust the face threshold in configuration
   - Ensure consistent lighting in training data

4. **Virtual camera not appearing in OBS**
   - Install OBS Virtual Camera plugin
   - Restart OBS after running the face overlay
   - Check Windows camera permissions

5. **Low performance/lag**
   - Reduce game region size
   - Lower webcam resolution
   - Adjust FPS in configuration

### Performance Tips

- **Optimize game region**: Capture only the area where faces appear
- **Good lighting**: Ensure consistent lighting for webcam
- **Stable setup**: Minimize camera movement during streaming
- **Hardware**: Use a dedicated GPU for better performance

## Advanced Configuration

Edit `face_overlay_config.json` for fine-tuning:

```json
{
    "game_region": {
        "top": 100,
        "left": 100, 
        "width": 1000,
        "height": 1000
    },
    "input_size": 184,
    "webcam_face_size": 200,
    "fps": 30,
    "face_detection_confidence": 0.7,
    "face_threshold": 0.5,
    "overlay_alpha": 1.0,
    "mirror_webcam": true,
    "show_debug": true
}
```

## Requirements

- **Python 3.7+**
- **Webcam** for face capture
- **Windows** (for virtual camera support)
- **4GB+ RAM** for model training
- **Dedicated GPU** recommended for real-time performance

## License

This project is for educational and personal use. Please respect game terms of service when streaming.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run `python main.py test` to validate your setup
3. Review the configuration settings
4. Ensure all dependencies are installed correctly
