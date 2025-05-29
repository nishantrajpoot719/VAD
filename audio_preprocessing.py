import os
import sys
import logging
import shutil
import subprocess
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def ensure_ffmpeg_available():
    """Ensure FFmpeg is available for audio processing"""
    # Check if ffmpeg is in the system path
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        logger.info(f"Found FFmpeg in system path: {system_ffmpeg}")
        return system_ffmpeg
    
    # Check for FFmpeg in the app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First check ffmpeg_binary directory
    ffmpeg_binary_path = os.path.join(app_dir, 'ffmpeg_binary', 'ffmpeg.exe')
    if os.path.exists(ffmpeg_binary_path):
        logger.info(f"Found FFmpeg in ffmpeg_binary directory: {ffmpeg_binary_path}")
        return ffmpeg_binary_path
    
    # Then check ffmpeg directory
    ffmpeg_dir = os.path.join(app_dir, 'ffmpeg')
    if os.path.exists(ffmpeg_dir):
        logger.info(f"Searching for FFmpeg in directory: {ffmpeg_dir}")
        for root, dirs, files in os.walk(ffmpeg_dir):
            for file in files:
                if file.lower() == 'ffmpeg.exe' or file.lower() == 'ffmpeg':
                    ffmpeg_path = os.path.join(root, file)
                    logger.info(f"Found FFmpeg in ffmpeg directory: {ffmpeg_path}")
                    return ffmpeg_path
    
    logger.warning("FFmpeg not found in any known location")
    return None

def preprocess_audio_for_opensmile(audio_path):
    """Preprocess audio file to make it compatible with OpenSMILE feature extraction"""
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            return audio_path
            
        # Create output path for preprocessed audio
        temp_dir = os.path.dirname(audio_path)
        filename = os.path.basename(audio_path)
        preprocessed_path = os.path.join(temp_dir, f"preprocessed_{filename}")
        
        # Get ffmpeg path
        ffmpeg_path = ensure_ffmpeg_available()
        if not ffmpeg_path:
            logger.warning(f"FFmpeg not found, returning original audio path: {audio_path}")
            return audio_path
        
        # Use ffmpeg to preprocess the audio file with specific parameters
        # -ar 22050: Set sample rate to 22050 Hz
        # -ac 1: Convert to mono
        # -acodec pcm_s16le: Use 16-bit PCM encoding
        cmd = [
            ffmpeg_path,
            "-i", audio_path,
            "-ar", "22050",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",  # Overwrite output file if it exists
            preprocessed_path
        ]
        
        logger.debug(f"Running ffmpeg preprocessing command: {' '.join(cmd)}")
        
        # Run with detailed error handling
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"FFmpeg preprocessing completed successfully")
        except subprocess.CalledProcessError as cmd_error:
            logger.error(f"FFmpeg preprocessing failed with code {cmd_error.returncode}: {cmd_error.stderr}")
            return audio_path
        
        if os.path.exists(preprocessed_path) and os.path.getsize(preprocessed_path) > 0:
            logger.debug(f"Successfully preprocessed audio file: {preprocessed_path}")
            return preprocessed_path
        else:
            logger.warning(f"Preprocessed file does not exist or is empty: {preprocessed_path}")
            return audio_path
    except Exception as e:
        logger.error(f"Error in audio preprocessing: {str(e)}")
        return audio_path

def fix_nan_values_in_features(features):
    """Fix NaN values in audio features"""
    try:
        if not isinstance(features, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame")
            return pd.DataFrame()
            
        if features.empty:
            logger.warning("Empty features DataFrame provided")
            return features
            
        # Check for NaN values
        nan_count = features.isna().sum().sum()
        if nan_count == 0:
            logger.debug("No NaN values found in features")
            return features
            
        # Calculate percentage of NaN values
        nan_percentage = (nan_count / (features.shape[0] * features.shape[1])) * 100
        logger.warning(f"High percentage of NaN values in features: {nan_percentage:.2f}%. Audio quality may be poor.")
        
        # Strategy depends on percentage of NaNs
        if nan_percentage > 90:
            logger.warning("Extremely high NaN percentage, using zeros instead of means")
            features_fixed = features.fillna(0)
        else:
            # Fill with column means first
            column_means = features.mean()
            column_means.fillna(0, inplace=True)  # Handle columns that are all NaN
            features_fixed = features.fillna(column_means)
            
            # Check if we still have NaNs
            remaining_nans = features_fixed.isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"Still have {remaining_nans} NaN values after filling with means, filling with zeros")
                features_fixed = features_fixed.fillna(0)
        
        logger.info(f"Successfully fixed {nan_count} NaN values in features")
        return features_fixed
    except Exception as e:
        logger.error(f"Error fixing NaN values: {str(e)}")
        return features.fillna(0)  # Last resort
