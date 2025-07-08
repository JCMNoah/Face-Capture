# Training Guide for Better Character Detection

## Why Train More?

Your current model might not be detecting characters well because:
1. **Limited training data** - Need more diverse samples
2. **Inconsistent lighting** - Characters look different in various scenes
3. **Different character types** - Various face shapes, angles, styles
4. **Game-specific features** - Each game has unique character rendering

## How to Collect Better Training Data

### 1. Run Data Collection
```bash
python main.py collect
```

### 2. Collection Strategy

**For FACE samples (press 'f'):**
- ✅ Character face clearly visible
- ✅ Different lighting conditions (bright, dark, indoor, outdoor)
- ✅ Different character angles (front, side, slight turn)
- ✅ Different character types (male, female, different races)
- ✅ Different distances (close-up, medium, far)
- ✅ Different expressions (happy, sad, angry, neutral)

**For NO_FACE samples (press 'n'):**
- ✅ Menus and UI screens
- ✅ Landscapes without characters
- ✅ Character backs/rear views
- ✅ Hands, weapons, objects close-up
- ✅ Text/dialogue screens
- ✅ Loading screens
- ✅ Cutscenes without faces visible

### 3. Quality Tips

**Good FACE samples:**
- Character face takes up 20-60% of the capture region
- Face is clearly recognizable as a face
- Good contrast between face and background
- Eyes, nose, mouth are visible

**Good NO_FACE samples:**
- Clearly no human/character faces visible
- Variety of game content (UI, environment, objects)
- Different color schemes and lighting

### 4. Quantity Recommendations

**Minimum for decent results:**
- 50 FACE samples
- 50 NO_FACE samples

**Recommended for good results:**
- 100-200 FACE samples
- 100-200 NO_FACE samples

**For excellent results:**
- 300+ FACE samples
- 300+ NO_FACE samples

## Training Tips

### 1. Balance Your Dataset
Make sure you have roughly equal numbers of FACE and NO_FACE samples.

### 2. Check Your Data
Before training, look through your collected images:
- Delete blurry or unclear samples
- Remove duplicates
- Ensure correct labeling

### 3. Train Multiple Times
If results aren't good enough:
1. Collect more data
2. Retrain the model
3. Test and repeat

### 4. Adjust Detection Threshold
After training, you can fine-tune:
```bash
python main.py config
# Option 3: Face threshold (try 0.3-0.7)
```

## Testing Your Model

### 1. Check Training Results
Look for:
- Training accuracy > 85%
- Validation accuracy > 80%
- Low difference between training and validation accuracy

### 2. Real-world Testing
```bash
python main.py test
```

### 3. Live Testing
```bash
python main.py run
```
Watch the debug output for prediction values and detection confidence.

## Troubleshooting

**Model detects faces when there aren't any:**
- Collect more NO_FACE samples
- Lower the face threshold

**Model misses obvious faces:**
- Collect more FACE samples with similar lighting/angles
- Raise the face threshold slightly
- Check if MediaPipe is detecting the faces (hybrid detection helps)

**Inconsistent detection:**
- Increase detection smoothing frames
- Collect more diverse training data
- Ensure good lighting in training samples

## Advanced Tips

1. **Game-specific training**: Collect data from the specific game you're streaming
2. **Session-based collection**: Collect data during different game sessions for variety
3. **Multiple characters**: Include different character types in your training data
4. **Environmental variety**: Collect samples from different game areas/levels

Remember: More diverse, high-quality training data = better detection results!
