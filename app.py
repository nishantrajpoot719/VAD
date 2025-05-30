import os
import sys
import tempfile
import logging
import traceback
import shutil
import time
import datetime
import json
import csv
import threading
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import cv2
from pydub import AudioSegment
import noisereduce as nr
from deepface import DeepFace
import opensmile
from concurrent.futures import ThreadPoolExecutor, as_completed
from improved_ravdess_integration import integrate_with_audio_pipeline, retrain_with_new_data  # Import the improved emotion classifier

# Auto-check for ffmpeg_binary/ffmpeg.exe
from download_ffmpeg import download_and_extract_ffmpeg

FFMPEG_PATH = os.path.join("ffmpeg_binary", "ffmpeg.exe")
if not os.path.exists(FFMPEG_PATH):
    download_and_extract_ffmpeg()


# Explicitly import audio preprocessing functions and make them available globally
try:
    from audio_preprocessing import preprocess_audio_for_opensmile, fix_nan_values_in_features, ensure_ffmpeg_available
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported audio preprocessing functions")
    
    # Ensure FFmpeg is available at startup
    ffmpeg_path = ensure_ffmpeg_available()
    if ffmpeg_path:
        # Set FFmpeg path for pydub
        AudioSegment.converter = ffmpeg_path
        logger.info(f"Set pydub to use FFmpeg at: {ffmpeg_path}")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Error importing audio preprocessing functions: {str(e)}")
    
    # Define fallback functions in case import fails
    def preprocess_audio_for_opensmile(audio_path):
        """Fallback preprocessing function if import fails"""
        logger.warning("Using fallback audio preprocessing function")
        return audio_path
        
    def fix_nan_values_in_features(features):
        """Fallback NaN fixing function if import fails"""
        logger.warning("Using fallback NaN fixing function")
        if isinstance(features, pd.DataFrame) and not features.empty:
            return features.fillna(0)
        return features
        
    def ensure_ffmpeg_available():
        """Fallback FFmpeg check function if import fails"""
        logger.warning("Using fallback FFmpeg check function")
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            ffmpeg_path = os.path.join(app_dir, 'ffmpeg_binary', 'ffmpeg.exe')
            if os.path.exists(ffmpeg_path):
                return ffmpeg_path
        return ffmpeg_path

# Import additional modules
from Contextual import get_device_info
from food_recommendations import get_food_recommendations

# Configure logging
LOG_FILE = "app_log.log"

# Global logger instance
logger = None

# Setup logging - append to log file instead of recreating it on each restart
def setup_logging():
    """Configure logging to properly append to the log file"""
    global logger
    
    # If logger is already initialized, return it
    if logger is not None:
        return logger
        
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # First reset the root logger to avoid duplicate handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Configure basic logging with 'a' (append) mode file handler
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),  # 'a' for append mode
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get our module logger
    logger = logging.getLogger(__name__)
    
    # Add a timestamp and separator line to indicate new session
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"{'='*25} NEW SESSION STARTED AT {current_time} {'='*25}")
    
    return logger

# Initialize logger - only do this once at the module level
logger = setup_logging()

# Add a function to clear log file if needed (called manually by user)
def clear_log_file():
    """Clear the log file but preserve the file"""
    global logger
    try:
        # First remove all handlers from logger to release file handles
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
        
        # Clear the log file
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")  # Write empty string
        
        # Re-initialize logging after clearing
        logger = setup_logging()
        logger.info("Log file cleared successfully")
        return True
    except Exception as e:
        # Try to re-initialize logger even if clearing failed
        try:
            logger = setup_logging()
            logger.error(f"Error clearing log file: {str(e)}")
        except:
            pass
        return False

# Performance configurations
MAX_AUDIO_DURATION = 15  # seconds, limit processing for very long videos (reduced from 60)
MAX_VIDEO_FRAMES = 21    # maximum frames to process (increased from 15)
CHUNK_SIZE = 1000        # ms, larger chunks for faster silence removal (increased from 500)
MIN_FACE_SIZE = 50       # minimum face size to detect (for better quality faces)
BLUR_THRESHOLD = 100     # Laplacian variance threshold for blur detection
MIN_BRIGHTNESS = 30      # Minimum average brightness for frame selection
TIMEOUT_SECONDS = 120    # maximum processing time before timeout (reduced from 180)
PARALLEL_WORKERS = 6     # number of parallel workers for frame processing
RESULTS_DIR = "analysis_results"  # directory to save analysis results
LEARNING_DATA_DIR = "learning_data"  # directory to save learning data

# Dynamic learning settings (hidden from UI)
AUTO_LEARN_ENABLED = False  # Enable silent learning in the background
AUTO_LEARN_THRESHOLD = 5   # Number of samples before auto-retraining
CONFIDENCE_THRESHOLD = 0.6  # Confidence below which we'll silently collect data

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LEARNING_DATA_DIR, exist_ok=True)

# Initialize OpenSMILE with eGeMAPS feature set (88 features)
def get_smile():
    try:
        # Use standard eGeMAPS feature set which provides 88 features
        smile_instance = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPS,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        logger.info(f"Initialized OpenSMILE with eGeMAPS feature set: {len(smile_instance.feature_names)} features")
        return smile_instance
    except Exception as e:
        error_msg = f"Failed to initialize OpenSMILE: {str(e)}"
        logger.error(error_msg)


# Initialize the OpenSMILE instance
smile = get_smile()

def detect_blur(image):
    """Detect if an image is blurry using Laplacian variance"""
    if image is None or image.size == 0:
        return True  # Consider empty images as blurry
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def calculate_brightness(image):
    """Calculate average brightness of an image"""
    if image is None or image.size == 0:
        return 0
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Return average of the V channel (brightness)
    return hsv[:, :, 2].mean()

def detect_face_quality(image, min_face_size=MIN_FACE_SIZE):
    """Detect faces and check if they meet quality criteria"""
    if image is None or image.size == 0:
        return False, 0
        
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Haar cascade for fast face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Check if any detected face meets size criteria
    max_face_size = 0
    for (x, y, w, h) in faces:
        face_size = min(w, h)
        max_face_size = max(max_face_size, face_size)
        
    return len(faces) > 0 and max_face_size >= min_face_size, max_face_size

def process_frame(frame_data):
    """Process a single frame for face detection with quality checks"""
    frame_count, frame, temp_dir = frame_data
    
    # Skip processing for empty frames
    if frame is None or frame.size == 0:
        logger.debug(f"Frame {frame_count} is empty, skipping")
        return (frame_count, 'skipped', None, 0)
    
    # Quality checks
    is_blurry = detect_blur(frame)
    brightness = calculate_brightness(frame)
    has_good_face, face_size = detect_face_quality(frame)
    
    # Calculate initial quality score
    quality_score = face_size/2 + (0 if is_blurry else 20) + brightness/5
    
    # Save frame even if lower quality
    frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)
    
    # Emotion analysis - try on all frames that have a face
    if has_good_face or face_size > 0:
        try:
            # Try fastest backend first, then fall back to others if needed
            backends = ['opencv', 'ssd', 'retinaface', 'mtcnn']
            success = False
            
            for backend in backends:
                try:
                    logger.debug(f"Trying face detection with backend: {backend}")
                    analysis = DeepFace.analyze(
                        img_path=frame_path,
                        actions=['emotion'],
                        enforce_detection=True, 
                        detector_backend=backend,
                        silent=True
                    )
                    
                    # If we get here, the analysis was successful
                    success = True
                    
                    # Boost score significantly for frames with emotions
                    emotion = analysis[0]['dominant_emotion']
                    emotion_score = 100  # High base score for any emotion detection
                    
                    logger.debug(f"Frame {frame_count}: Detected emotion '{emotion}' with backend {backend}")
                    return (frame_count, emotion, frame_path, quality_score + emotion_score)
                
                except Exception as e:
                    # Just try the next backend
                    logger.debug(f"Backend {backend} failed for frame {frame_count}: {str(e)}")
                    continue
            
            # If all backends failed but we had a face
            if not success:
                logger.debug(f"All face detection backends failed for frame {frame_count}")
            return (frame_count, 'no_emotion', frame_path, quality_score)
                
        except Exception as e:
            logger.debug(f"Face detection failed for frame {frame_count}: {str(e)}")
            return (frame_count, 'no_emotion', frame_path, quality_score)
    else:
        # No face detected
        logger.debug(f"No face detected in frame {frame_count}")
        return (frame_count, 'no_face', frame_path, quality_score/4)

def process_audio_pipeline(file_path):
    """Optimized audio processing pipeline with support for audio-only files"""
    start_time = time.time()
    temp_dir = None
    audio_path = None
    
    # Initialize variables with default values to prevent reference errors
    features = pd.DataFrame()
    audio_emotion = "neutral"
    audio_emotion_confidence = 0.5
    audio_emotion_top3 = [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2)]
    
    try:                
        logger.info(f"Starting audio processing for {os.path.basename(file_path)}")
        # Check cache first - use a more robust cache key
        cache_key = f"audio_{os.path.basename(file_path)}_{os.path.getsize(file_path)}"
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temp audio directory: {temp_dir}")

        # Validate file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            # Check if this is an audio-only file
            is_audio_file = file_path.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))
            
            audio_path = os.path.join(temp_dir, "original_audio.wav")
            
            if is_audio_file:
                # For audio-only files, use AudioFileClip instead of VideoFileClip
                audio = AudioFileClip(file_path)
                if audio is None:
                    raise ValueError("Failed to load audio file")
                
                # Further reduce max audio duration for processing
                max_duration = min(MAX_AUDIO_DURATION, 30)  # Increased to 30 seconds max
                safe_end = min(max_duration, audio.duration)
                if audio.duration > safe_end:
                    logger.debug(f"Limiting audio to first {safe_end} seconds (safe)")
                    audio = audio.subclip(0, safe_end)
                
                try:
                    audio.write_audiofile(audio_path, logger=None, fps=22050)  # Original sample rate
                except Exception as e:
                    logger.error(f"moviepy audio.write_audiofile failed: {str(e)}. Attempting fallback extraction with pydub.")
                    try:
                        raw_audio = AudioSegment.from_file(file_path)
                        raw_audio = raw_audio[:int(safe_end * 1000)]  # milliseconds
                        raw_audio.export(audio_path, format="wav")
                        logger.info(f"Fallback audio extraction with pydub succeeded: {audio_path}")
                    except Exception as e2:
                        logger.error(f"Fallback audio extraction also failed: {str(e2)}")
                        raise
                audio.close()
                del audio  # Force garbage collection
            else:
                # For video files, use VideoFileClip as before but with optimizations
                # Increase the audio_buffersize to prevent indexing errors
                video = VideoFileClip(file_path, audio_buffersize=500000, target_resolution=(360, None))  # Significantly increased buffer size
                if video.audio is None:
                    raise ValueError("Uploaded video contains no audio track")
                    
                # Limit audio duration for very long videos
                max_duration = min(MAX_AUDIO_DURATION, 30)  # Increased to 30 seconds max
                safe_end = min(max_duration, video.duration)
                if video.duration > safe_end:
                    logger.debug(f"Limiting audio to first {safe_end} seconds (safe)")
                    video = video.subclip(0, safe_end)
                
                try:
                    # Ensure the temp directory exists and is writable
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir, exist_ok=True)
                    
                    # Make sure the audio path is absolute and the directory exists
                    audio_path = os.path.abspath(audio_path)
                    audio_dir = os.path.dirname(audio_path)
                    if not os.path.exists(audio_dir):
                        os.makedirs(audio_dir, exist_ok=True)
                        
                    # Skip moviepy audio extraction entirely and go straight to ffmpeg
                    # This is more reliable and avoids the index out of bounds errors
                    logger.info(f"Skipping moviepy audio extraction and using ffmpeg directly")
                    
                    # Get FFmpeg path using our utility function
                    ffmpeg_path = ensure_ffmpeg_available()
                    
                    if not ffmpeg_path:
                        # Try to find FFmpeg in common locations as a last resort
                        app_dir = os.path.dirname(os.path.abspath(__file__))
                        ffmpeg_binary_path = os.path.join(app_dir, 'ffmpeg_binary', 'ffmpeg.exe')
                        if os.path.exists(ffmpeg_binary_path):
                            ffmpeg_path = ffmpeg_binary_path
                            logger.info(f"Found ffmpeg in ffmpeg_binary directory: {ffmpeg_path}")
                    
                    if ffmpeg_path:
                        import subprocess
                        logger.info(f"Using ffmpeg for audio extraction: {ffmpeg_path}")
                        cmd = [ffmpeg_path, '-i', file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', '-y', audio_path]
                        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                        
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        logger.info(f"ffmpeg extraction succeeded: {audio_path}")
                    else:
                        # Fall back to moviepy as a last resort with even larger buffer
                        logger.warning("FFmpeg not found, falling back to moviepy with extremely large buffer")
                        video.audio.write_audiofile(audio_path, logger=None, buffersize=10000000, fps=22050, nbytes=4)
                except Exception as e:
                    logger.error(f"moviepy video.audio.write_audiofile failed: {str(e)}. Attempting fallback extraction with pydub.")
                    try:
                        # Create a temporary file for pydub to use
                        temp_audio_path = os.path.join(temp_dir, "temp_audio_file.wav")
                        
                        # Get FFmpeg path using our utility function
                        ffmpeg_path = ensure_ffmpeg_available()
                        if ffmpeg_path:
                            # Set ffmpeg path for pydub
                            AudioSegment.converter = ffmpeg_path
                            logger.info(f"Set pydub to use FFmpeg at: {ffmpeg_path}")
                        else:
                            logger.warning("FFmpeg not found, pydub extraction may fail")
                            
                            # Try to find FFmpeg in common locations as a last resort
                            app_dir = os.path.dirname(os.path.abspath(__file__))
                            ffmpeg_binary_path = os.path.join(app_dir, 'ffmpeg_binary', 'ffmpeg.exe')
                            if os.path.exists(ffmpeg_binary_path):
                                AudioSegment.converter = ffmpeg_binary_path
                                logger.info(f"Last resort: Set pydub to use FFmpeg at: {ffmpeg_binary_path}")
                        
                        # Try to extract audio using pydub directly from the video file
                        logger.info(f"Attempting pydub extraction from: {file_path}")
                        raw_audio = AudioSegment.from_file(file_path)
                        raw_audio = raw_audio[:int(safe_end * 1000)]  # milliseconds
                        raw_audio.export(audio_path, format="wav")
                        logger.info(f"Fallback audio extraction with pydub succeeded: {audio_path}")
                    except Exception as e2:
                        logger.error(f"Fallback audio extraction also failed: {str(e2)}")
                        # Try one more fallback method using ffmpeg directly
                        try:
                            import subprocess
                            
                            # First check ffmpeg_binary directory (most reliable location)
                            app_dir = os.path.dirname(os.path.abspath(__file__))
                            ffmpeg_binary_path = os.path.join(app_dir, 'ffmpeg_binary', 'ffmpeg.exe')
                            logger.info(f"Looking for ffmpeg at: {ffmpeg_binary_path}")
                            
                            # List contents of the directory for debugging
                            try:
                                ffmpeg_binary_dir = os.path.join(app_dir, 'ffmpeg_binary')
                                if os.path.exists(ffmpeg_binary_dir):
                                    dir_contents = os.listdir(ffmpeg_binary_dir)
                                    logger.info(f"Contents of ffmpeg_binary directory: {dir_contents}")
                            except Exception as e:
                                logger.warning(f"Could not list contents of ffmpeg_binary directory: {str(e)}")
                                
                            if os.path.exists(ffmpeg_binary_path):
                                ffmpeg_path = ffmpeg_binary_path
                                logger.info(f"Found ffmpeg in ffmpeg_binary directory: {ffmpeg_path}")
                            else:
                                # Try system path
                                ffmpeg_path = shutil.which('ffmpeg')
                                
                                # If not found, try to find in ffmpeg directory
                                if not ffmpeg_path:
                                    ffmpeg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
                                    if os.path.exists(ffmpeg_dir):
                                        logger.info(f"Searching for ffmpeg in directory: {ffmpeg_dir}")
                                        for root, dirs, files in os.walk(ffmpeg_dir):
                                            for file in files:
                                                if file.lower() == 'ffmpeg.exe' or file.lower() == 'ffmpeg':
                                                    ffmpeg_path = os.path.join(root, file)
                                                    logger.info(f"Found ffmpeg in ffmpeg directory: {ffmpeg_path}")
                                                    break
                                            if ffmpeg_path:
                                                break
                            
                            if ffmpeg_path:
                                logger.info(f"Attempting final fallback with direct ffmpeg command: {ffmpeg_path}")
                                # Use more reliable command parameters
                                # Log the command for debugging
                                cmd = [ffmpeg_path, '-i', file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', '-y', audio_path]
                                logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                                
                                # Run with more detailed error handling
                                try:
                                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                                    logger.info(f"ffmpeg command stdout: {result.stdout[:200]}...")
                                except subprocess.CalledProcessError as cmd_error:
                                    logger.error(f"ffmpeg command failed with code {cmd_error.returncode}: {cmd_error.stderr}")
                                    raise
                                logger.info(f"Direct ffmpeg extraction succeeded: {audio_path}")
                            else:
                                # Last resort - create a silent audio file without using ffmpeg
                                logger.warning("Could not find ffmpeg executable, creating silent audio file")
                                try:
                                    # Try to create silent audio without relying on ffmpeg
                                    import wave
                                    import array
                                    
                                    # Create a simple silent WAV file manually
                                    with wave.open(audio_path, 'w') as wav_file:
                                        wav_file.setnchannels(1)  # Mono
                                        wav_file.setsampwidth(2)  # 2 bytes (16 bits)
                                        wav_file.setframerate(22050)  # 22.05kHz
                                        # Create 1 second of silence (22050 frames)
                                        silence_data = array.array('h', [0] * 22050)
                                        wav_file.writeframes(silence_data.tobytes())
                                    
                                    logger.info(f"Created silent audio file manually: {audio_path}")
                                except Exception as wav_error:
                                    logger.error(f"Failed to create manual WAV file: {str(wav_error)}")
                                    # Last attempt with pydub
                                    silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
                                    silent_audio.export(audio_path, format="wav")
                                    logger.info(f"Created silent audio file as fallback: {audio_path}")
                        except Exception as e3:
                            logger.error(f"All audio extraction methods failed: {str(e3)}")
                            # Create silent audio as final fallback instead of raising
                            try:
                                # Try to create silent audio without relying on ffmpeg
                                import wave
                                import array
                                
                                # Create a simple silent WAV file manually
                                with wave.open(audio_path, 'w') as wav_file:
                                    wav_file.setnchannels(1)  # Mono
                                    wav_file.setsampwidth(2)  # 2 bytes (16 bits)
                                    wav_file.setframerate(22050)  # 22.05kHz
                                    # Create 1 second of silence (22050 frames)
                                    silence_data = array.array('h', [0] * 22050)
                                    wav_file.writeframes(silence_data.tobytes())
                                
                                logger.info(f"Created silent audio file manually after all methods failed: {audio_path}")
                            except Exception as e4:
                                logger.error(f"Even manual silent audio creation failed: {str(e4)}")
                                # We'll continue and let the next error handler deal with it
                video.close()
                del video  # Force garbage collection
                
            logger.info(f"Successfully extracted audio to {audio_path}")
                
        except Exception as e:
            logger.error(f"Audio loading failed: {str(e)}")
            # Instead of raising, try to create a silent audio file
            try:
                if not audio_path:
                    audio_path = os.path.join(temp_dir, "emergency_silent.wav")
                silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
                silent_audio.export(audio_path, format="wav")
                logger.info(f"Created emergency silent audio file: {audio_path}")
                # Continue processing with the silent audio
            except Exception as e2:
                logger.error(f"Failed to create emergency silent audio: {str(e2)}")
                # Continue with default values if we reach here
        
        try:
            # Check if the audio file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                logger.error(f"Audio file does not exist or is empty: {audio_path}")
                # Try to create a silent audio file as a last resort
                silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
                silent_audio.export(audio_path, format="wav")
                logger.info(f"Created silent audio file as fallback: {audio_path}")
                
            audio = AudioSegment.from_wav(audio_path)
        except IndexError as e:
            logger.error(f"Index error when loading audio: {str(e)}. Attempting with a fallback method.")
            # Fallback to loading with default parameters
            try:
                # Try to load with pydub's from_file instead which is more flexible
                audio = AudioSegment.from_file(audio_path)
            except Exception as e2:
                logger.error(f"Fallback audio loading also failed: {str(e2)}")
                # Try to create a silent audio file as a last resort
                try:
                    silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
                    silent_audio.export(audio_path, format="wav")
                    audio = silent_audio
                    logger.info("Using silent audio as fallback")
                except Exception as e3:
                    logger.error(f"Silent audio fallback also failed: {str(e3)}")
                    # Return default values if we can't load the audio at all
                    return None, audio_path, temp_dir, "neutral", 0.5, [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2)]
        except Exception as e:
            logger.error(f"Unexpected error loading audio: {str(e)}")
            return None, audio_path, temp_dir, "neutral", 0.5, [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2)]

        # Only perform silence removal if audio is longer than 5 seconds
        if len(audio) > 5000:
            logger.debug(f"Original audio duration: {len(audio)/1000:.2f} seconds")
        
            # Use larger chunks for faster processing
            non_silent_audio = AudioSegment.silent(duration=0)
            
            # Increase chunk size for better performance
            chunk_size = CHUNK_SIZE * 2  # Doubled chunk size
            
            # Process in larger chunks with more permissive silence threshold
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if chunk.dBFS > -40:  # More permissive silence threshold (-45 → -40)
                    non_silent_audio += chunk
                
            if len(non_silent_audio) > 0:
                logger.debug(f"Non-silent audio duration: {len(non_silent_audio)/1000:.2f} seconds")
                
                # Save processed audio for analysis
                processed_audio_path = os.path.join(temp_dir, "processed_audio.wav")
                non_silent_audio.export(processed_audio_path, format="wav")
                audio_path = processed_audio_path
            else:
                logger.warning("No non-silent audio found, using original audio")
        else:
            logger.debug("Audio too short for silence removal, using as is")
            
        # Free memory
        del audio
        
        # Initialize features and emotion variables with default values to avoid reference errors
        features = pd.DataFrame()
        audio_emotion = "neutral"
        audio_emotion_confidence = 0.5
        audio_emotion_top3 = [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2)]
        
        # Extract audio features using OpenSMILE
        try:
            # Check if the audio file exists and is valid
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                logger.error(f"Audio file does not exist or is empty: {audio_path}")
                raise FileNotFoundError(f"Audio file not found or empty: {audio_path}")
                
            # Use the smile instance to extract features
            # Preprocess audio to avoid NaN values
            try:
                preprocessed_audio_path = preprocess_audio_for_opensmile(audio_path)
                logger.info(f"Audio preprocessed successfully: {preprocessed_audio_path}")
            except Exception as preprocess_error:
                logger.error(f"Error preprocessing audio: {str(preprocess_error)}")
                preprocessed_audio_path = audio_path  # Fall back to original audio path
                
            # Extract features with better error handling
            try:
                # Process the file to extract eGeMAPS features
                features = smile.process_file(preprocessed_audio_path)
                
                # Transpose features to get them in the right format (samples x features)
                features = pd.DataFrame(features).T
                
                feature_count = features.shape[1]
                logger.info(f"Successfully extracted {feature_count} audio features")
                
                # Verify we have the expected number of features (88 for eGeMAPS)
                expected_feature_count = 88
                if feature_count != expected_feature_count:
                    logger.warning(f"Unexpected feature count: got {feature_count}, expected {expected_feature_count}")
                
                # Check for NaN values and handle them
                nan_count = features.isna().sum().sum()
                if nan_count > 0:
                    nan_percentage = (nan_count / (features.shape[0] * features.shape[1])) * 100
                    logger.warning(f"NaN values detected in features: {nan_percentage:.2f}%")
                    
                    # Use our fix_nan_values_in_features function to handle NaNs
                    features = fix_nan_values_in_features(features)
                    logger.info(f"Fixed {nan_count} NaN values in features")
                
                # Log first few feature values for debugging
                if not features.empty:
                    sample_features = features.iloc[0].head(5)
                    logger.debug(f"Sample features: {sample_features.to_dict()}")
                
                # If we have features, predict emotion using the RAVDESS model
                if not features.empty:
                    # Integrate with improved RAVDESS model
                    emotion_prediction = integrate_with_audio_pipeline(features)
                
                    # Extract prediction results
                    audio_emotion = emotion_prediction['predicted_emotion']
                    audio_emotion_confidence = emotion_prediction['confidence']
                    audio_emotion_top3 = emotion_prediction['top_emotions']
                    
                    logger.info(f"Audio emotion detected: {audio_emotion} with confidence {audio_emotion_confidence:.2f}")
                    logger.info(f"Top 3 emotions: {audio_emotion_top3}")
                else:
                    logger.warning("No audio features extracted, using default emotion values")
            except Exception as feature_error:
                logger.error(f"Error extracting features: {str(feature_error)}")
                # Continue with default values
        except Exception as e:
            logger.error(f"Error in audio emotion detection: {str(e)}\n{traceback.format_exc()}")
            # Continue with default values
        
        # Return results
        return audio_path, temp_dir, audio_emotion, audio_emotion_confidence

    except TimeoutError as e:
        logger.error(f"Audio pipeline timed out: {str(e)}")
        try:
            logger.error(f"Audio Processing Timeout: Processing took too long and was aborted")
        except Exception:
            logger.debug("Could not display error in UI (safe to ignore)")
        return None, None, temp_dir, "neutral", 0.5, [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2)]
        
    except Exception as e:
        logger.error(f"Audio pipeline failed: {str(e)}\n{traceback.format_exc()}")
        try:
            logger.error(f"Audio Processing Error: {str(e)}")
        except Exception:
            logger.debug("Could not display error in UI (safe to ignore)")
        
        # Create a silent audio file as a last resort if we have a temp directory
        if temp_dir and not audio_path:
            try:
                audio_path = os.path.join(temp_dir, "silent_fallback.wav")
                silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
                silent_audio.export(audio_path, format="wav")
                logger.info(f"Created silent audio file as final fallback: {audio_path}")
            except Exception as silent_error:
                logger.error(f"Failed to create silent fallback: {str(silent_error)}")
                
        return None, audio_path, temp_dir, "neutral", 0.5, [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2)]

def process_video_pipeline(video_path, max_frames=MAX_VIDEO_FRAMES):
    """Optimized video processing pipeline with parallel frame processing"""
    start_time = time.time()
    temp_dir = None
    cap = None
    
    try:
        # Safely update progress
        logger.debug("Starting video processing")
        
        # Check cache
        cache_key = f"video_{os.path.basename(video_path)}_{os.path.getsize(video_path)}"
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temp video directory: {temp_dir}")

        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            logger.warning("Video metadata indicates 0 frames or 0 fps, trying to process anyway")
            total_frames = 1000  # Assume a reasonable default
            fps = 30  # Assume a reasonable default
        
        # Reduce sample frames for faster processing
        sample_count = min(total_frames, max_frames * 2)  # Reduced multiplier
        
        # For very long videos, ensure we sample from a reduced time window
        max_time_seconds = min(30, total_frames / fps if fps > 0 else 30)  # Reduced to 30 seconds
        max_time_frames = int(max_time_seconds * fps) if fps > 0 else 600
        
        # Take fewer frames at regular intervals throughout the video
        frame_interval = max(1, min(max_time_frames, total_frames) // sample_count)
        max_frames_to_sample = min(max_frames, min(max_time_frames, total_frames) // frame_interval)
        
        logger.debug(f"Sampling {max_frames_to_sample} frames with interval {frame_interval}")

        # Prepare frames for parallel processing
        frames_to_process = []
        
        
        # Extract fewer frames at regular intervals
        for i in range(max_frames_to_sample):
            if time.time() - start_time > TIMEOUT_SECONDS:
                raise TimeoutError("Video processing timeout during frame extraction")
                
            frame_idx = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize frame for faster processing
                if frame is not None and frame.size > 0 and frame.shape[1] > 640:  # If width > 640px
                    scale = 640 / frame.shape[1]
                    frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
                frames_to_process.append((frame_idx, frame, temp_dir))
        
        # Close video capture to free resources
        if cap is not None:
            cap.release()
            cap = None
        
        # Check if we have frames to process
        if not frames_to_process:
            logger.warning("No frames could be extracted from the video")
            return ["neutral"], [], temp_dir        
            
        frame_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(PARALLEL_WORKERS, len(frames_to_process), 6)  # Limit to 6 workers max
        logger.debug(f"Processing {len(frames_to_process)} frames in parallel with {num_workers} workers")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_frame, frame_data): frame_data[0] 
                      for frame_data in frames_to_process}
            
            for i, future in enumerate(as_completed(futures)):
                logger.debug(f"Processed frame {i+1}/{len(frames_to_process)}")
                try:
                    frame_results.append(future.result())
                except Exception as e:
                    logger.error(f"Frame processing error: {str(e)}")
                
                if time.time() - start_time > TIMEOUT_SECONDS:
                    executor.shutdown(wait=False)
                    raise TimeoutError("Video processing timeout")

        # Free up memory from frames_to_process
        del frames_to_process
        import gc
        gc.collect()

        # Check if we have any results
        if not frame_results:
            logger.warning("No frames could be processed from the video")
            default_emotion = "neutral"
            return [default_emotion], [], temp_dir
        
        # First prioritize frames with actual emotions
        emotion_results = [r for r in frame_results if r[1] not in ['no_emotion', 'no_face', 'low_quality', 'poor_quality', 'skipped']]
        
        # If we don't have any emotion results, use any frame with a face
        combined_results = emotion_results if emotion_results else frame_results
        
        # Sort and limit
        combined_results.sort(key=lambda x: x[3], reverse=True)  # Sort by quality score
        combined_results = combined_results[:max_frames]  # Take only needed frames
        
        # Extract results
        emotions = [result[1] for result in combined_results]
        frame_paths = [result[2] for result in combined_results]
        
        # If no emotions found, default to neutral
        if not emotions or all(e in ['no_emotion', 'no_face', 'low_quality', 'poor_quality', 'skipped'] for e in emotions):
            logger.warning("No valid emotions detected in video frames, defaulting to neutral")
            emotions = ["neutral"]
        
        # Log detected emotions for debugging
        emotion_counts = {}
        for emotion in emotions:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        logger.info(f"Video emotion detection results: {emotion_counts}")
        logger.info(f"Detected emotions: {emotions[:10]}" + ("..." if len(emotions) > 10 else ""))
    
        # Log success
        logger.info(f"Video processing completed successfully with {len(emotions)} emotions detected")

        return emotions, frame_paths, temp_dir



    except TimeoutError as e:
        logger.error(f"Video Processing Timeout: Processing took too long and was aborted")
        return ["neutral"], [], temp_dir

    except Exception as e:
        logger.error(f"Video pipeline failed: {str(e)}\n{traceback.format_exc()}")
        return ["neutral"], [], temp_dir

    finally:
        # Ensure resources are released
        if cap is not None:
            try:
                cap.release()
            except:
                pass

# VAD conversion function
EMOTION_VAD_MAP = {
    'neutral': [0.0, 0.0, 0.0],
    'calm': [0.0, -0.5, 0.5],
    'happy': [0.8, 0.6, 0.4],
    'sad': [-0.6, -0.3, -0.2],
    'angry': [-0.5, 0.7, 0.3],
    'fearful': [-0.7, 0.8, -0.5],
    'disgust': [-0.6, 0.4, -0.3],
    'surprised': [0.4, 0.8, 0.2]
}

def emotion_to_vad(emotion):
    """Convert an emotion label to VAD (Valence-Arousal-Dominance) values
    Args:
        emotion: String representing the emotion
    Returns:
        List of three values: [valence, arousal, dominance]
    """
    # Normalize emotion name (lowercase and strip whitespace)
    if emotion is None:
        logger.warning("None emotion passed to emotion_to_vad, using neutral")
        emotion = "neutral"
    emotion = emotion.lower().strip()
    # Get VAD values from mapping, with fallback to neutral
    vad_values = EMOTION_VAD_MAP.get(emotion)
    if vad_values is None:
        # Handle similar emotion names
        if 'happy' in emotion or 'joy' in emotion:
            vad_values = EMOTION_VAD_MAP.get('happy')
        elif 'sad' in emotion or 'unhappy' in emotion:
            vad_values = EMOTION_VAD_MAP.get('sad')
        elif 'angry' in emotion or 'anger' in emotion:
            vad_values = EMOTION_VAD_MAP.get('angry')
        elif 'fear' in emotion or 'scared' in emotion:
            vad_values = EMOTION_VAD_MAP.get('fearful')
        elif 'disgust' in emotion:
            vad_values = EMOTION_VAD_MAP.get('disgust')
        elif 'surprise' in emotion:
            vad_values = EMOTION_VAD_MAP.get('surprised')
        elif 'calm' in emotion or 'relaxed' in emotion:
            vad_values = EMOTION_VAD_MAP.get('calm')
        else:
            logger.warning(f"Unknown emotion '{emotion}', using neutral VAD values")
            vad_values = EMOTION_VAD_MAP.get('neutral')
    logger.debug(f"Converted emotion '{emotion}' to VAD values: {vad_values}")
    return vad_values



# Configure ffmpeg path for better compatibility

def setup_ffmpeg():
    """Setup ffmpeg by finding the best available FFmpeg installation"""
    try:
        logger.info("Starting FFmpeg setup")
        ffmpeg_path = None
        ffmpeg_found = False
        # Get the absolute path to the application directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Application directory: {app_dir}")
        # First check for ffmpeg in the ffmpeg_binary directory (prioritize this)
        binary_dir = os.path.join(app_dir, "ffmpeg_binary")
        logger.info(f"Looking for ffmpeg in: {binary_dir}")
        if os.path.exists(binary_dir):
            logger.info(f"Found ffmpeg_binary directory at: {binary_dir}")
            # List contents of the directory for debugging
            try:
                dir_contents = os.listdir(binary_dir)
                logger.info(f"Contents of ffmpeg_binary directory: {dir_contents}")
            except Exception as e:
                logger.warning(f"Could not list contents of ffmpeg_binary directory: {str(e)}")
            # Look for ffmpeg executable in the directory
            if os.name == 'nt':  # Windows
                exe_name = "ffmpeg.exe"
            else:  # Unix/Linux/Mac
                exe_name = "ffmpeg"
            ffmpeg_path = os.path.join(binary_dir, exe_name)
            logger.info(f"Checking for ffmpeg at: {ffmpeg_path}")
            if os.path.exists(ffmpeg_path):
                logger.info(f"Found ffmpeg executable at: {ffmpeg_path}")
                # Verify file size and permissions
                try:
                    file_stats = os.stat(ffmpeg_path)
                    file_size = file_stats.st_size
                    is_executable = os.access(ffmpeg_path, os.X_OK)
                    logger.info(f"ffmpeg file size: {file_size} bytes, executable: {is_executable}")
                except Exception as e:
                    logger.warning(f"Could not verify ffmpeg stats: {str(e)}")
                ffmpeg_found = True
                # Set environment variables for better compatibility
                os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
            else:
                logger.warning(f"ffmpeg executable not found at expected path: {ffmpeg_path}")
                # Configure libraries to use this ffmpeg
                try:
                    # Configure moviepy
                    import moviepy.config as moviepy_config
                    moviepy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
                    # Configure pydub
                    from pydub import AudioSegment
                    AudioSegment.converter = ffmpeg_path
                    # Set additional environment variables
                    os.environ["FFMPEG_BINARY"] = ffmpeg_path
                    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
                    logger.info("Libraries configured to use local ffmpeg")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to configure libraries with ffmpeg binary: {str(e)}")
        # Next check if ffmpeg is available in local ffmpeg directory
        local_ffmpeg_dir = os.path.abspath("ffmpeg")
        if os.path.exists(local_ffmpeg_dir) and not ffmpeg_found:
            logger.info(f"Found local ffmpeg directory at: {local_ffmpeg_dir}")
            # Look for ffmpeg executable in the directory
            if os.name == 'nt':  # Windows
                exe_name = "ffmpeg.exe"
            else:
                exe_name = "ffmpeg"
            # Search in all subdirectories for the executable
            for root, dirs, files in os.walk(local_ffmpeg_dir):
                for file in files:
                    if file.lower() == exe_name.lower():
                        exe_path = os.path.join(root, file)
                        logger.info(f"Using local ffmpeg at: {exe_path}")
                        ffmpeg_path = exe_path
                        ffmpeg_found = True
                    
                        # Configure libraries to use local ffmpeg
                        try:
                            # For moviepy
                            try:
                                import moviepy.config as moviepy_config
                                # Check if change_settings is available in this version of moviepy
                                if hasattr(moviepy_config, 'change_settings'):
                                    moviepy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
                                else:
                                    # Direct attribute assignment for newer versions of moviepy
                                    moviepy_config.FFMPEG_BINARY = ffmpeg_path
                                # Set environment variable for subprocess calls
                                os.environ["FFMPEG_BINARY"] = ffmpeg_path
                            except Exception as e:

                                logger.warning(f"Failed to configure moviepy: {str(e)}")
                            # For pydub
                            from pydub import AudioSegment
                            AudioSegment.converter = ffmpeg_path
                            logger.info("Libraries configured to use local ffmpeg")
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to configure libraries with local ffmpeg: {str(e)}")
                        break
                if ffmpeg_found:
                    break
        # Next try to use imageio-ffmpeg (reliable)
        if not ffmpeg_found:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                if os.path.exists(ffmpeg_exe):
                    logger.info(f"Using imageio-ffmpeg at: {ffmpeg_exe}")
                    ffmpeg_path = ffmpeg_exe
                    ffmpeg_found = True
                    # Configure libraries to use imageio ffmpeg
                    try:
                        # For moviepy
                        import moviepy.config as moviepy_config
                        if hasattr(moviepy_config, 'change_settings'):
                            moviepy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
                        else:
                          # Direct attribute assignment for newer versions
                            moviepy_config.FFMPEG_BINARY = ffmpeg_path
                        # Set environment variable for subprocess calls
                        os.environ["FFMPEG_BINARY"] = ffmpeg_path
                        # For pydub
                        from pydub import AudioSegment
                        AudioSegment.converter = ffmpeg_path
                        logger.info("Libraries configured to use imageio-ffmpeg")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to configure libraries with imageio-ffmpeg: {str(e)}")
            except ImportError:
                logger.info("imageio-ffmpeg not installed, trying other methods")
            except Exception as e:
                from pydub import AudioSegment
                AudioSegment.converter = ffmpeg_path
                logger.info("Libraries configured to use imageio-ffmpeg")
                return True
            except Exception as e:
                logger.warning(f"Failed to configure libraries with imageio-ffmpeg: {str(e)}")
            except ImportError:
                logger.info("imageio-ffmpeg not installed, trying other methods")
            except Exception as e:
                logger.warning(f"Error using imageio-ffmpeg: {str(e)}")
            # Next check if ffmpeg is available in PATH
        if not ffmpeg_found:
            import subprocess
            try:
                # Different command for windows vs other OS
                if os.name == 'nt':  # Windows
                    result = subprocess.run(['where', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                else:
                    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    ffmpeg_path = result.stdout.strip()
                    ffmpeg_found = True
                    logger.info(f"System ffmpeg found at: {ffmpeg_path}")
                    # Configure libraries to use system ffmpeg
                    try:
                        import moviepy.config as moviepy_config
                        if hasattr(moviepy_config, 'change_settings'):
                            moviepy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
                        else:
                            moviepy_config.FFMPEG_BINARY = ffmpeg_path
                        # Set environment variable for subprocess calls
                        os.environ["FFMPEG_BINARY"] = ffmpeg_path
                        # For pydub
                        from pydub import AudioSegment
                        AudioSegment.converter = ffmpeg_path
                        logger.info("Libraries configured to use system ffmpeg")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to configure libraries with system ffmpeg: {str(e)}")
            except Exception as e:
                logger.warning(f"System ffmpeg not found: {str(e)}")
        # If we still haven't found ffmpeg, report failure
        if not ffmpeg_found:
            logger.error("Could not find any FFmpeg installation. Please ensure ffmpeg.exe is in the ffmpeg_binary directory.")
            return False
        return ffmpeg_found
    except Exception as e:
        logger.error(f"Error in setup_ffmpeg: {str(e)}")
        return False


# Test ffmpeg integration with libraries
def test_ffmpeg_integration():
    """Test if ffmpeg is properly configured with pydub and moviepy"""
    integration_ok = True
    temp_dir = None
    try:
        logger.info("Testing FFmpeg integration")
        # Test pydub
        try:
            from pydub import AudioSegment
            logger.info("Testing pydub integration with ffmpeg...")
            # Create a short silent AudioSegment (doesn't need a file)
            silent = AudioSegment.silent(duration=500)  # 500ms of silence
            # Test exporting to ensure ffmpeg works
            temp_dir = tempfile.mkdtemp()
            test_path = os.path.join(temp_dir, "test_audio.wav")
            silent.export(test_path, format="wav")
            # Verify the file was created
            if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                logger.info(f"Pydub integration passed - created test file at {test_path}")
                # Clean up
                try:
                    os.remove(test_path)
                    os.rmdir(temp_dir)
                    temp_dir = None  # Mark as cleaned up
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up test files: {cleanup_error}")
            else:
                logger.error("Pydub integration failed - could not create test file")
                integration_ok = False
        except Exception as e:
            logger.error(f"Pydub integration failed: {str(e)}")
            integration_ok = False
        # Test moviepy (only if pydub worked)
        if integration_ok:
            try:
                # Get ffmpeg path - first check environment, then check ffmpeg_binary directory
                ffmpeg_path = os.environ.get("FFMPEG_BINARY", None)
                if not ffmpeg_path or not os.path.exists(ffmpeg_path):
                    binary_dir = os.path.abspath("ffmpeg_binary")
                    exe_name = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
                    ffmpeg_path = os.path.join(binary_dir, exe_name)
                logger.info(f"Testing MoviePy with ffmpeg at: {ffmpeg_path}")
                # Set ffmpeg path for moviepy
                import moviepy.config as moviepy_config
                if hasattr(moviepy_config, 'change_settings'):
                    moviepy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
                else:
                    moviepy_config.FFMPEG_BINARY = ffmpeg_path
                # Import with the configured path
                from moviepy.editor import VideoFileClip, AudioFileClip
                # Simple test case - create a very short audio clip
                moviepy_temp_dir = tempfile.mkdtemp()
                test_wav_path = os.path.join(moviepy_temp_dir, "test.wav")
                try:
                    # Create a simple silent audio clip
                    dummy_audio = AudioSegment.silent(duration=500)
                    dummy_audio.export(test_wav_path, format="wav")
                    # Try to load it with AudioFileClip
                    if os.path.exists(test_wav_path):
                        logger.info(f"Testing AudioFileClip with {test_wav_path}")
                        audio_clip = AudioFileClip(test_wav_path)
                        audio_clip.close()
                        logger.info("MoviePy integration passed")
                    else:
                        raise FileNotFoundError(f"Failed to create test audio file at: {test_wav_path}")
                finally:
                    # Clean up temp files
                    try:
                        if os.path.exists(test_wav_path):
                            os.unlink(test_wav_path)
                        if os.path.exists(moviepy_temp_dir):
                            os.rmdir(moviepy_temp_dir)
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up test files: {cleanup_error}")
            except Exception as e:
                logger.error(f"MoviePy integration failed: {str(e)}")
                integration_ok = False
        # Final result
        if integration_ok:
            logger.info("All FFmpeg integrations successful")
        else:
            logger.warning("Some FFmpeg integrations failed")
        return integration_ok
    except Exception as e:
        logger.error(f"Error testing FFmpeg integration: {str(e)}")
        return False
    finally:
        # Final cleanup in case of early exit
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Already logged above

def calculate_final_vad(video_path, audio_path):
    """Calculate final VAD score using confidence-weighted average of audio and video emotion"""
    
    # Process video and extract emotions
    video_emotions, _, _ = process_video_pipeline(video_path)
    
    # Count valid video emotions
    emotion_counts = {}
    for emo in video_emotions:
        if emo not in ['neutral', 'no_emotion', 'no_face', 'skipped']:
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    if emotion_counts:
        primary_video_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        video_confidence = emotion_counts[primary_video_emotion] / len(video_emotions)
        video_confidence = max(0.6, video_confidence)  # Min confidence
    else:
        primary_video_emotion = "neutral"
        video_confidence = 0.5

    logger.info(f"Primary video emotion: {primary_video_emotion} (confidence: {video_confidence})")

    # Process audio
    _, _, audio_emotion, audio_confidence = process_audio_pipeline(audio_path)
    audio_confidence = max(0.5, audio_confidence)

    logger.info(f"Audio emotion: {audio_emotion} (confidence: {audio_confidence})")

    # Convert both to VAD
    video_vad = emotion_to_vad(primary_video_emotion)
    audio_vad = emotion_to_vad(audio_emotion)

    # Confidence-weighted average
    total_conf = video_confidence + audio_confidence
    final_vad = [(video_vad[i]*video_confidence + audio_vad[i]*audio_confidence) / total_conf for i in range(3)]
    final_vad = [round(v, 2) for v in final_vad]

    return final_vad, primary_video_emotion, audio_emotion

def main():
    logger.info("=== Starting Final VAD + Contextual Pipeline ===")

    import argparse

    parser = argparse.ArgumentParser(description="Emotion-aware VAD pipeline")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    args = parser.parse_args()

    video_path = args.video
    audio_path = video_path  # Same file for audio extraction

    # Step 1: Calculate final VAD
    final_vad, vid_emotion, aud_emotion = calculate_final_vad(video_path, audio_path)

    # Step 2: Get device/contextual data
    try:
        context_data = get_device_info()
    except Exception as e:
        logger.error(f"Failed to retrieve contextual info: {str(e)}")
        context_data = ["N/A", "N/A", "N/A", "N/A"]

    # Step 3: Print final results
    print(f"Dominant Video Emotion : {vid_emotion}")
    print(f"Dominant Audio Emotion : {aud_emotion}")
    print(f"Final VAD Score        : {final_vad}")
    print(f"Contextual Information : {context_data}")


    logger.info("Pipeline completed successfully.")

        # Step 4: Call food recommendation engine
    try:
        intent_selections = ["Hot", "Light", "Tangy"]  # Placeholder — replace with user input logic
        recommendations = get_food_recommendations(final_vad, intent_selections, context_data)
        
        print("\n🧠 Final Food Recommendations:")
        print(json.dumps(recommendations, indent=4))
    except Exception as e:
        logger.error(f"Error getting food recommendations: {str(e)}")
        print("❌ Failed to get recommendations. Check logs for details.")



if __name__ == "__main__":
    python_version = (sys.version_info.major, sys.version_info.minor)
    if python_version < (3, 8):
        logger.error(f"Requires Python 3.8+, found {python_version}")
        sys.exit(1)

    if not setup_ffmpeg():
        logger.warning("FFmpeg setup failed. Attempting fallback.")
    
    if not test_ffmpeg_integration():
        logger.warning("FFmpeg integration test failed. Continuing with caution.")

    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
