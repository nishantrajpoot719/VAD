import os
import sys
import joblib
import pandas as pd
import numpy as np
import opensmile
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

# Get logger
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "audio_emotion_classifier_egemaps.pkl"
SCALER_PATH = "audio_feature_scaler_egemaps.pkl"
LEARNING_DATA_DIR = "learning_data"

# Feature set information
FEATURE_SET = "eGeMAPS"
EXPECTED_FEATURE_COUNT = 88  # eGeMAPS feature set has 88 features

# Standard emotion set to ensure consistent output
STANDARD_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Create a fallback model for when no trained model is available
def create_fallback_model():
    """Create a simple fallback model when no trained model is available"""
    logger.warning("Creating fallback emotion model - this will use simplified heuristics")
    # Simple heuristic model using basic rules
    model = SVC(probability=True)
    # Basic feature scaler
    scaler = StandardScaler()
    return model, scaler, "Fallback"

def verify_feature_count(features):
    """Verify that features match the expected count for our model"""
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        logger.error("Invalid or empty features provided")
        return False
    
    current_count = features.shape[1]
    if current_count != EXPECTED_FEATURE_COUNT:
        logger.warning(f"Feature count mismatch: Expected {EXPECTED_FEATURE_COUNT}, got {current_count}")
        return False
    
    return True

def load_model():
    """Load the emotion prediction model"""
    try:
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading emotion model from {MODEL_PATH}...")
            model = joblib.load(MODEL_PATH)
            model_type = "SVM"
            
            # Check model classes
            if hasattr(model, 'classes_'):
                logger.info(f"Model loaded with classes: {model.classes_}")
                
                # Warn if model has very limited classes
                if len(model.classes_) < 4:
                    logger.warning(f"Model has limited emotion classes: {model.classes_}")
            
            # Load the scaler
            if os.path.exists(SCALER_PATH):
                scaler = joblib.load(SCALER_PATH)
                logger.info(f"Feature scaler loaded from {SCALER_PATH}")
            else:
                logger.warning(f"Scaler {SCALER_PATH} not found, using default scaler")
                scaler = StandardScaler()
            
            return model, scaler, model_type
        else:
            logger.warning(f"Model file {MODEL_PATH} not found, using fallback model")
            return create_fallback_model()
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return create_fallback_model()

def predict_emotion_heuristic(audio_features):
    """Predict emotion using simple heuristics when models aren't available"""
    logger.warning("Using heuristic emotion prediction - less accurate than trained models")
    
    try:
        # Default emotion and confidence
        emotion = "neutral"
        confidence = 0.5
        
        # Get a few key features if available
        if not audio_features.empty:
            # Handle NaN values before calculating statistics
            if audio_features.isna().any().any():
                logger.warning("NaN values detected in heuristic prediction, filling with zeros")
                audio_features = audio_features.fillna(0)
                
            # Calculate feature statistics
            feature_mean = audio_features.mean().mean()
            feature_std = audio_features.std().mean()
            
            # Simple heuristic rules based on common patterns
            # Higher energy/amplitude features often correlate with excited emotions
            # Higher variance often correlates with stronger emotions
            
            if feature_std > 2.0:  # High variance
                if feature_mean > 1.0:  # High mean values
                    emotion = "happy"
                    confidence = 0.6
                elif feature_mean < -0.5:  # Low mean values
                    emotion = "sad"
                    confidence = 0.55
                else:
                    emotion = "surprised"
                    confidence = 0.5
            elif feature_std > 1.0:  # Medium variance
                if feature_mean > 0.5:
                    emotion = "calm"
                    confidence = 0.7
                else:
                    emotion = "neutral"
                    confidence = 0.65
            else:  # Low variance
                emotion = "neutral"
                confidence = 0.7
                
            logger.info(f"Heuristic detected {emotion} with confidence {confidence:.2f}")
        
        # Create top emotions with some spread of probabilities
        if emotion == "neutral":
            top_emotions = [
                ("neutral", confidence),
                ("calm", confidence * 0.8),
                ("happy", confidence * 0.3),
                ("sad", confidence * 0.2)
            ]
        elif emotion == "happy":
            top_emotions = [
                ("happy", confidence),
                ("neutral", confidence * 0.5),
                ("surprised", confidence * 0.4),
                ("calm", confidence * 0.3)
            ]
        elif emotion == "sad":
            top_emotions = [
                ("sad", confidence),
                ("neutral", confidence * 0.6),
                ("calm", confidence * 0.3),
                ("fearful", confidence * 0.2)
            ]
        elif emotion == "calm":
            top_emotions = [
                ("calm", confidence),
                ("neutral", confidence * 0.8),
                ("happy", confidence * 0.3),
                ("sad", confidence * 0.1)
            ]
        else:
            top_emotions = [
                (emotion, confidence),
                ("neutral", confidence * 0.7),
                ("calm", confidence * 0.4),
                ("happy", confidence * 0.2)
            ]
            
        return {
            'predicted_emotion': emotion,
            'confidence': confidence,
            'top_emotions': top_emotions,
            'model_type': "Heuristic"
        }
    except Exception as e:
        logger.error(f"Error in heuristic prediction: {str(e)}")
        # Absolute fallback values
        return {
            'predicted_emotion': "neutral",
            'confidence': 0.5,
            'top_emotions': [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2), ("sad", 0.1)],
            'model_type': "Fallback"
        }

def predict_emotion(audio_features):
    """Predict emotion from audio features using the best available model"""
    # Make sure audio_features is a DataFrame
    try:
        if audio_features is None or not isinstance(audio_features, pd.DataFrame) or audio_features.empty:
            logger.error("Invalid or empty audio features provided")
            return "unknown", {}, None
            
        # Log feature statistics for debugging
        try:
            nan_count = audio_features.isna().sum().sum()
            feature_stats = {
                'shape': audio_features.shape,
                'nan_count': nan_count,
                'nan_percentage': f"{(nan_count / (audio_features.shape[0] * audio_features.shape[1]) * 100):.2f}%" if audio_features.size > 0 else "0.00%",
                'mean': audio_features.mean().mean(),
                'std': audio_features.std().mean(),
                'min': audio_features.min().min(),
                'max': audio_features.max().max()
            }
            logger.debug(f"Audio feature statistics: {feature_stats}")
            
            # Handle NaN values if present
            if nan_count > 0:
                nan_percentage = (nan_count / (audio_features.shape[0] * audio_features.shape[1])) * 100
                logger.warning(f"NaN values detected in features: {nan_percentage:.2f}%")
                
                # If we have a high percentage of NaN values, use zeros instead of means
                if nan_percentage > 50:
                    logger.warning("Extremely high NaN percentage, using zeros instead of means")
                    audio_features = audio_features.fillna(0)
                else:
                    # Replace NaN values with column means
                    column_means = audio_features.mean()
                    # If a column is all NaN, use 0 instead
                    column_means = column_means.fillna(0)
                    audio_features = audio_features.fillna(column_means)
                
                logger.info(f"Fixed {nan_count} NaN values in features")
        except Exception as stats_error:
            logger.error(f"Error calculating feature statistics: {str(stats_error)}")
            # Continue with prediction despite statistics error
        
        # Load the model
        model, scaler, model_type = load_model()
        
        # Check if we have a valid model
        if model is None:
            logger.error("No valid model available for prediction")
            return "unknown", {}, None
        
        # Log model information
        if hasattr(model, 'classes_'):
            logger.debug(f"Using {model_type} for emotion prediction")
            logger.debug(f"Audio features shape: {audio_features.shape}")
            
            # Warn if model has limited classes
            if len(model.classes_) < 4:
                logger.warning(f"Model has limited emotion classes: {model.classes_}")
        
        # Scale features
        try:
            # Check for NaN values before scaling
            if audio_features.isna().any().any():
                logger.warning("NaN values detected before scaling, filling with zeros")
                audio_features = audio_features.fillna(0)
                
            # Scale features
            scaled_features = scaler.transform(audio_features)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Get prediction probabilities if available
            probabilities = {}
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(scaled_features)[0]
                probabilities = {emotion: float(prob) for emotion, prob in zip(model.classes_, proba)}
                
                # Log top 3 emotions with probabilities
                top_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"Top emotions: {top_emotions}")
            
            # Check if the model has limited classes (less than 4 emotions)
            has_limited_classes = len(model.classes_) < 4
            
            # If the model has limited classes
            # supplement with additional emotions from heuristics
            if has_limited_classes:
                # Get heuristic prediction to supplement
                heuristic_result = predict_emotion_heuristic(audio_features)
                
                # Create a list of additional emotions not in the model
                additional_emotions = []
                for emotion_name in STANDARD_EMOTIONS:
                    if emotion_name not in model.classes_:
                        for heur_emotion, heur_prob in heuristic_result['top_emotions']:
                            if heur_emotion == emotion_name:
                                # Add emotion from heuristic but with lower confidence
                                additional_emotions.append((emotion_name, heur_prob * 0.6))
                                break
                
                # Combine model emotions with heuristic emotions
                all_emotions = [(e, probabilities[e]) for e in model.classes_]
                all_emotions.extend(additional_emotions)
                
                # Sort by probability
                top_emotions = sorted(all_emotions, key=lambda x: x[1], reverse=True)[:4]
                
                # If 'neutral' is not in the top emotions, add it
                has_neutral = any(e[0] == 'neutral' for e in top_emotions)
                if not has_neutral:
                    # Get the highest probability as confidence
                    max_confidence = max(probabilities.values()) if probabilities else 0.5
                    neutral_prob = 0.3 if max_confidence > 0.7 else 0.4
                    # Replace the lowest probability emotion
                    top_emotions = top_emotions[:3] + [('neutral', neutral_prob)]
                    # Re-sort
                    top_emotions = sorted(top_emotions, key=lambda x: x[1], reverse=True)
            else:
                # Normal case with sufficient emotion classes
                top_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:4]
            
            # Calculate confidence as the highest probability
            confidence = max(probabilities.values()) if probabilities else 0.7
            
            logger.info(f"Model predicted {prediction} with confidence {confidence:.2f}")
            
            return {
                'predicted_emotion': prediction,
                'confidence': confidence,
                'top_emotions': top_emotions,
                'model_type': model_type
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return predict_emotion_heuristic(audio_features)
            
    except Exception as e:
        logger.error(f"General error in emotion prediction: {str(e)}")
        return predict_emotion_heuristic(audio_features)

def integrate_with_audio_pipeline(audio_features):
    """Integration function for audio pipeline
    
    This is a drop-in replacement for the original integration function
    that uses the improved emotion classification model.
    
    Args:
        audio_features: The OpenSMILE features extracted using eGeMAPS feature set
        
    Returns:
        Dictionary with emotion prediction and confidence
    """
    try:
        # Ensure audio_features is a proper DataFrame
        if not isinstance(audio_features, pd.DataFrame):
            logger.error("Audio features must be a pandas DataFrame")
            return {
                'predicted_emotion': "neutral",
                'confidence': 0.5,
                'top_emotions': [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2), ("sad", 0.1)],
                'model_type': "Error Fallback"
            }
            
        # Check if we need to transpose the features
        # We expect features to be in the format (samples x features), typically (1 x 88)
        if audio_features.shape[0] > audio_features.shape[1] and audio_features.shape[0] == EXPECTED_FEATURE_COUNT:
            logger.info(f"Transposing features from shape {audio_features.shape} to match expected format")
            audio_features = audio_features.T
            
        # Log feature characteristics for debugging
        if not audio_features.empty:
            # Check for NaN values first
            nan_count = audio_features.isna().sum().sum()
            nan_percentage = (nan_count / (audio_features.shape[0] * audio_features.shape[1])) * 100 if audio_features.size > 0 else 0
            
            feature_stats = {
                "shape": audio_features.shape,
                "nan_count": nan_count,
                "nan_percentage": f"{nan_percentage:.2f}%",
                "mean": audio_features.mean().mean(),
                "std": audio_features.std().mean(),
                "min": audio_features.min().min(),
                "max": audio_features.max().max()
            }
            logger.debug(f"Audio feature statistics: {feature_stats}")
            
            # If we have too many NaN values, log a warning
            if nan_percentage > 50:
                logger.warning(f"High percentage of NaN values in features: {nan_percentage:.2f}%. Audio quality may be poor.")
        
        # Predict emotion
        result = predict_emotion(audio_features)
        logger.info(f"Final prediction: {result['predicted_emotion']} ({result['confidence']:.2f}) using {result['model_type']} model")
        
        # Always ensure 'neutral' is available in the top emotions
        has_neutral = any(e[0] == 'neutral' for e in result['top_emotions'])
        if not has_neutral:
            # Add neutral with a moderate probability
            neutral_prob = 0.3 if result['confidence'] > 0.7 else 0.45
            new_top = result['top_emotions'][:3] + [('neutral', neutral_prob)]
            # Sort by probability
            result['top_emotions'] = sorted(new_top, key=lambda x: x[1], reverse=True)
        
        return result
    except Exception as e:
        logger.error(f"Integration error: {str(e)}")
        # Return default values as fallback
        return {
            'predicted_emotion': "neutral",
            'confidence': 0.5,
            'top_emotions': [("neutral", 0.5), ("calm", 0.3), ("happy", 0.2), ("sad", 0.1)],
            'model_type': "Error Fallback"
        }

def retrain_with_new_data(features_df, emotions_df):
    """Retrain the emotion model with new data
    
    Args:
        features_df: DataFrame with audio features
        emotions_df: DataFrame with emotion labels
    
    Returns:
        True if retraining was successful, False otherwise
    """
    try:
        logger.info(f"Retraining emotion model with {len(emotions_df)} new samples using {FEATURE_SET} feature set")
        
        # Convert emotions to a Series if it's a DataFrame
        if isinstance(emotions_df, pd.DataFrame):
            emotions_series = emotions_df.iloc[:, 0]
        else:
            emotions_series = emotions_df
            
        # Check if we have enough data
        if len(emotions_series) < 5:
            logger.warning(f"Not enough data for retraining: {len(emotions_series)} samples. Need at least 5.")
            return False
            
        # Check if we have at least 2 different emotions
        unique_emotions = emotions_series.unique()
        if len(unique_emotions) < 2:
            logger.warning(f"Need at least 2 different emotions for training. Got only: {unique_emotions}")
            return False
            
        # Check for NaN values in features
        nan_count = features_df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in features. Filling with column means.")
            # Fill NaN values with column means
            column_means = features_df.mean()
            column_means.fillna(0, inplace=True)  # For columns that are all NaN
            features_df = features_df.fillna(column_means)
            
            # Check if we still have NaNs
            remaining_nans = features_df.isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"Still have {remaining_nans} NaN values after filling with means. Filling with zeros.")
                features_df = features_df.fillna(0)
        
        # Verify feature count
        if features_df.shape[1] != EXPECTED_FEATURE_COUNT:
            logger.error(f"Feature count mismatch: got {features_df.shape[1]}, expected {EXPECTED_FEATURE_COUNT}")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, emotions_series, test_size=0.2, random_state=42, stratify=emotions_series
        )
        
        # Scale features with new scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Train a new SVM model
        logger.info("Training new SVM model")
        # Use a more robust classifier that can handle potential NaN values better
        from sklearn.ensemble import RandomForestClassifier
        
        # First try with SVM which is good for this type of data
        try:
            model = SVC(probability=True, kernel='rbf', C=10.0, gamma='scale')
            model.fit(X_train_resampled, y_train_resampled)
            logger.info("Successfully trained SVM model")
        except Exception as e:
            logger.warning(f"SVM training failed: {str(e)}. Trying RandomForest instead.")
            # Fall back to RandomForest which is more robust
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train_resampled, y_train_resampled)
            logger.info("Successfully trained RandomForest model as fallback")
        
        # Save new model and scaler
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        logger.info(f"Scaler saved to {SCALER_PATH}")
        
        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")
        
        return True
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing improved RAVDESS integration module...")
    
    # Create a simple test if we have the feature file
    if os.path.exists('ravdess_features.csv'):
        # Load a single sample
        features = pd.read_csv('ravdess_features.csv').iloc[[0]]
        
        # Get prediction
        prediction = integrate_with_audio_pipeline(features)
        
        # Print results
        print(f"Predicted emotion: {prediction['predicted_emotion']}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        print(f"Model type: {prediction['model_type']}")
        print("\nTop emotions:")
        for emotion, prob in prediction['top_emotions']:
            print(f"{emotion}: {prob:.2f}")
    else:
        print("No feature file found for testing. Please run improved_ravdess_model.py first.") 