# Emotion Analysis Backend Service

A high-performance backend service for analyzing emotions from audio and video files. This service provides RESTful API endpoints for processing media files and extracting emotional insights.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This service provides emotion analysis capabilities for different media types:

- **Audio Analysis**: Extract emotions from speech using machine learning models
- **Video Analysis**: Process video frames for facial emotion detection
- **Real-time Processing**: Efficient processing with parallel execution
- **Scalable Architecture**: Designed for high-performance and reliability

## Features

### Core Functionality
- Audio feature extraction using OpenSMILE
- Facial emotion detection using DeepFace
- Parallel processing of media files
- Comprehensive logging and error handling

### Performance Optimizations
- Efficient memory management
- Parallel frame processing
- Caching of location and weather data
- Resource cleanup

### Quality Controls
- Face detection and quality assessment
- Blur detection
- Brightness analysis
- Minimum face size requirements

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd flutterflow-video-processing
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the configuration as needed

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# API Keys
WEATHER_API_KEY=your_weather_api_key

# Paths (optional)
RESULTS_DIR=./results
MODEL_DIR=./models
```

## Usage

### Command Line

```bash
# Analyze a media file
python flutterapp.py path/to/your/media/file.mp4

# Analyze with custom settings
python flutterapp.py --max-frames 30 --workers 4 path/to/your/media/file.mp4
```

### API Server

Start the FastAPI server:

```bash
uvicorn api.server:app --reload
```

Then access the API documentation at `http://localhost:8000/docs`

## API Documentation

### POST /api/analyze

Analyze a media file for emotions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file`: The media file to analyze
  - `analyze_audio`: boolean (optional, default: true)
  - `analyze_video`: boolean (optional, default: true)
  - `max_frames`: integer (optional, default: 21)

**Response:**
```json
{
  "status": "success",
  "analysis_id": "unique-id",
  "results": {
    "audio_emotion": {
      "emotion": "happy",
      "confidence": 0.85
    },
    "video_emotion": {
      "dominant_emotion": "happy",
      "emotion_scores": {
        "angry": 0.1,
        "happy": 0.7,
        "sad": 0.1,
        "surprise": 0.1
      },
      "frame_count": 21,
      "processed_frames": 18
    },
    "context": {
      "timestamp": "2023-06-15T14:30:00Z",
      "location": "New York, NY, USA",
      "weather": "Sunny, 25°C"
    }
  },
  "processing_time": 12.5
}
```

## Development

### Project Structure

```
.
├── config/               # Configuration files
│   └── __init__.py
├── models/               # Trained models
├── results/              # Analysis results
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── location.py      # Location services
│   ├── weather.py       # Weather services
│   └── image_processor.py # Image processing utilities
├── api/
│   └── server.py       # API server implementation
├── flutterapp.py        # Main application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses `black` for code formatting and `flake8` for linting.

```bash
black .
flake8
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and in your system PATH
   - Or place the FFmpeg binary in the `ffmpeg_binary` directory

2. **Model loading errors**
   - Ensure model files are in the `models` directory
   - Check file permissions

3. **API key issues**
   - Verify your API keys in the `.env` file
   - Ensure the file is named exactly `.env`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Emotion Classification Model

The application uses an improved SVM-based model trained on the RAVDESS dataset for audio emotion classification. The model can identify 8 emotions:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

### Model Improvements
- Uses SMOTE for handling class imbalance
- Optimized SVM parameters (C=10, balanced class weights)
- Achieves high accuracy across all emotion categories
- Supports continuous learning and retraining

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg for audio/video processing

### Dependencies
```bash
pip install streamlit pandas numpy moviepy opencv-python pydub noisereduce deepface opensmile scikit-learn matplotlib seaborn imbalanced-learn joblib
```

### RAVDESS Dataset
To train the emotion classification model, you need the RAVDESS dataset:

1. Download using the provided script:
```bash
python download_ravdess.py
```

2. The script will download and extract the dataset to the RAVDESS directory.

## Usage

### Running the Application
```bash
streamlit run first.py
```

### Analyzing Media
1. Select the file type (Video, Audio, or Image)
2. Upload your file through the interface
3. Adjust settings in the sidebar if needed
4. View the analysis results
5. Export the results if desired

### Adjusting Settings
- **Performance Settings**: Control the number of video frames, audio duration, and parallel workers
- **Model Settings**: Adjust feature selection methods and number of features
- **Quality Settings**: Set minimum face size, blur threshold, and brightness threshold
- **Export Settings**: Configure automatic export and filename prefixes

## File Structure

```
.
├── first.py                         # Main application file
├── improved_ravdess_model.py        # RAVDESS model training script
├── improved_ravdess_integration.py  # Integration of model with main app
├── download_ravdess.py              # Script to download RAVDESS dataset
├── test_integration.py              # Test script for model integration
├── test_model_loading.py            # Test script for model loading
├── model_integration_summary.md     # Documentation of model integration
├── improved_audio_emotion_classifier.pkl  # Trained SVM model
├── improved_audio_feature_scaler.pkl     # Feature scaler for model
├── ravdess_features.csv             # Extracted RAVDESS features
├── ravdess_emotions.csv             # RAVDESS emotion labels
├── RAVDESS/                         # RAVDESS dataset directory
└── learning_data/                   # Directory for continuous learning data
```

## Technical Details

### Audio Emotion Classification
The audio emotion classification uses OpenSMILE to extract acoustic features from audio files. These features are then fed into an SVM model trained on the RAVDESS dataset.

#### Feature Extraction
- Uses OpenSMILE's ComParE_2016 feature set
- Extracts over 6,000 acoustic features
- Feature standardization using StandardScaler

#### SVM Model
- RBF kernel (determined by grid search)
- C=10 for better accuracy
- Balanced class weights to handle imbalanced data
- SMOTE for oversampling minority classes

### Video Processing
- Face detection using DeepFace
- Facial emotion classification
- Quality filtering (blur detection, brightness check, face size)
- Parallel frame processing for efficiency

### Continuous Learning
The application includes a continuous learning system that:
1. Collects high-confidence predictions
2. Allows manual correction of predictions
3. Accumulates learning data
4. Automatically retrains the model when sufficient data is collected

## Model Training and Evaluation

### Training Process
The improved RAVDESS model is trained using the following process:
1. Extract features from RAVDESS audio files
2. Split into training and test sets
3. Apply SMOTE to balance classes
4. Perform grid search to find optimal parameters
5. Train the final model with the best parameters
6. Evaluate on test set and generate confusion matrix

### Model Performance
The model achieves high accuracy across all emotion categories, with detailed evaluation metrics available in the confusion matrix and classification report.

### Testing
The model is tested with:
- Sample audio files from RAVDESS
- Integration tests to ensure proper functionality
- Performance evaluation on real-world data

## Integration

The improved RAVDESS model is integrated with the main application through the `improved_ravdess_integration.py` file, which provides:

1. Model loading functions
2. Prediction functions that return emotion with confidence scores
3. Retraining functions for continuous learning

The integration maintains the same API as the original emotion classifier, making it a seamless replacement.

## Advanced Features

### Retraining with User Feedback
The application can collect user feedback to improve the model over time:
```python
# When user corrects a prediction
save_learning_data(audio_features, corrected_emotion, filename)

# Automatic retraining when enough data is collected
auto_retrain_model_if_needed()
```

### Custom Model Parameters
Advanced users can modify model parameters in `improved_ravdess_integration.py`:
```python
svm = SVC(
    C=10,                # Higher C value for better accuracy
    gamma='scale',       # Use scale for gamma calculation
    kernel='rbf',        # RBF kernel for better performance
    probability=True,    # Enable probability estimates
    class_weight='balanced',  # Balance class weights
    random_state=42
)
```

## Troubleshooting

### Common Issues
- **Missing FFmpeg**: Install FFmpeg and ensure it's in your system PATH
- **OpenSMILE errors**: Ensure OpenSMILE is properly installed and configured
- **Memory errors**: Reduce parallel workers or max frames in performance settings
- **Model loading errors**: Ensure all model files are in the correct location

### Model Testing
If you encounter issues with the emotion model, you can run the test scripts:
```bash
python test_model_loading.py
python test_integration.py
```

### Reporting Issues
For any issues, please run the diagnostic tests and include the output in your report. 