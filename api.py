import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from Contextual import get_device_info
from food_recommendations import get_food_recommendations
from app import calculate_final_vad

# Load environment variables from .env
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '🎯 VAD-Based Food Recommendation API is running!'

@app.route('/get-recommendation', methods=['POST', 'GET'])
def get_recommendation():
    try:
        # Expecting JSON input with video path + intent options
        data = request.get_json()
        video_path = data.get('video_path')
        intent_selections = data.get('intent_selections', ["Hot", "Tangy", "Light"])

        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Invalid or missing video_path'}), 400

        # Step 1: Calculate VAD
        final_vad, video_emotion, audio_emotion = calculate_final_vad(video_path, video_path)

        # Step 2: Get contextual data
        context_data = get_device_info()

        # Step 3: Get food recommendations
        result = get_food_recommendations(final_vad, intent_selections, context_data)

        # Step 4: Respond with everything
        return jsonify({
            "video_emotion": video_emotion,
            "audio_emotion": audio_emotion,
            "final_vad": final_vad,
            "context": context_data,
            "recommendations": result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
