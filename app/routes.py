from flask import Blueprint, jsonify, request, send_from_directory, current_app, send_file
from .model import predict_safety_hybrid, generate_safety_map, feature_importance, standardized_data
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the API key
GENAI_API_KEY = os.getenv("API_KEY")

# Configure the Gemini API client
genai.configure(api_key=GENAI_API_KEY)

def clean_response(raw_text):
    """
    Cleans and formats the AI-generated response to extract meaningful content and format it properly.
    """
    import re

    if not isinstance(raw_text, str):
        raw_text = str(raw_text)  # Convert to string if necessary

    # Step 1: Remove API artifacts (e.g., "parts {", "text", "role")
    cleaned_text = re.sub(r"parts\s*\{.*?text\s*\"", "", raw_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\"\s*\}\s*role\s*\"model\"", "", cleaned_text, flags=re.DOTALL)

    # Step 2: Normalize text and remove unnecessary characters
    cleaned_text = cleaned_text.replace("\\n", "\n").replace("\\'", "'")
    cleaned_text = re.sub(r"\\", "", cleaned_text)  # Remove escape characters
    cleaned_text = re.sub(r"\s+\n", "\n", cleaned_text)  # Remove extra spaces before newlines

    # Step 3: Format subheadings (more robust regex)
    def format_subheadings(match):
        return f"\n**{match.group(1).strip()}**\n"

    # Match subheadings that end with a colon or are followed by a newline
    cleaned_text = re.sub(r"^([A-Za-z0-9\s\-]+):", format_subheadings, cleaned_text, flags=re.MULTILINE)

    # Step 4: Format bullet points (handle numbers, dashes, and asterisks)
    cleaned_text = re.sub(r"^(\d+\.|\-|\*)\s", "- ", cleaned_text, flags=re.MULTILINE)

    # Step 5: Remove excess newlines (more robust handling)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # Limit consecutive newlines to 2

    # Step 6: Trim leading/trailing whitespace and return
    return cleaned_text.strip()

def generate_response(user_input):
    """Generate a response using the Gemini API."""
    prompt = f"{user_input}"
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 64,
        "max_output_tokens": 2000,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config=generation_config,
    )
    
    try:
        response = model.generate_content(prompt)
        
        # Simple extraction - just get the text
        if hasattr(response, 'text') and response.text:
            raw_content = response.text
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    raw_content = candidate.content.parts[0].text
                else:
                    raw_content = str(candidate.content)
            else:
                raw_content = "Unable to generate detailed recommendations at this time."
        else:
            raw_content = "Unable to generate recommendations. Please try again."
        
        return clean_response(raw_content)
        
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise e



main_bp = Blueprint('main', __name__)
CORS(main_bp, resources={r"/*": {"origins": "*"}}) 

# Route to display the safety prediction
@main_bp.route('/predict_safety', methods=['POST'])
def predict_safety():
    data = request.get_json()
    country_name = data.get('country_name', '')
    if not country_name:
        return jsonify({'error': 'No country name provided'}), 400

    # Predict safety and trend
    predicted_status, trend = predict_safety_hybrid(country_name)

    return jsonify({
        'country': country_name,
        'safety_status': predicted_status,
        'trend': trend
    })

# Route to get recommendations
@main_bp.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        country = data.get('country')
        intent = data.get('intent')

        if not country or not intent:
            return jsonify({"error": "Missing 'country' or 'intent' in request"}), 400

        # Fetch safety status and trend
        result = predict_safety_hybrid(country)
        if not result or len(result) != 2:
            return jsonify({"error": "Unexpected result from predict_safety_hybrid"}), 500

        safety_status, trend = result

        # Prepare the prompt for Gemini AI
        if intent.lower() == "travel":
            prompt = (
                f"As a travel advisor, provide detailed travel recommendations for {country}. "
                f"Given that {country} currently has a {safety_status} status with a {trend} trend, "
                f"explain what this means for travelers and provide specific advice on:\n\n"
                f"1. What travelers should expect in {country} given the {safety_status} status\n"
                f"2. Recommended precautions and preparation based on this risk level\n"
                f"3. Best areas to visit and areas to be more cautious about\n"
                f"4. Transportation and accommodation recommendations\n"
                f"5. Cultural attractions and experiences worth visiting\n"
                f"6. Practical travel tips including best times to visit, currency, and communication\n\n"
                f"Make your recommendations practical and specific to the {safety_status} classification."
            )
        elif intent.lower() == "settlement":
            prompt = (
                f"As a relocation advisor, provide detailed settlement recommendations for {country}. "
                f"Given that {country} currently has a {safety_status} status with a {trend} trend, "
                f"explain what this means for people considering long-term residence and provide specific advice on:\n\n"
                f"1. What living in {country} is like given the {safety_status} status\n"
                f"2. Quality of life considerations and infrastructure reliability\n"
                f"3. Employment opportunities and economic stability\n"
                f"4. Cost of living, housing market, and financial planning\n"
                f"5. Legal requirements, visa processes, and residency procedures\n"
                f"6. Social integration, community life, and cultural considerations\n\n"
                f"Make your recommendations practical and specific to the {safety_status} classification and {trend} trend."
            )
        else:
            return jsonify({"error": "Invalid intent value. Supported intents are 'Travel' and 'Settlement'"}), 400

        # Generate response using Gemini API
        try:
            detailed_recommendation = generate_response(prompt)
        except Exception as api_error:
            print(f"Error in generate_response: {api_error}")
            return jsonify({"error": f"Failed to generate response: {str(api_error)}"}), 500

        return jsonify({"recommendation": detailed_recommendation}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Failed to fetch recommendation: {str(e)}"}), 500


@main_bp.route('/visualize_map')
def visualize_map():
    map_path = os.path.join(current_app.static_folder, 'safety_map.html')
    if os.path.exists(map_path):
        return send_from_directory(current_app.static_folder, 'safety_map.html')
    else:
        return "Map not generated yet. Please try again later.", 404


# Route to display the heatmap
@main_bp.route('/heatmap')
def heatmap():
    plt.figure(figsize=(18, 8))
    sns.heatmap(
        standardized_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5
    )
    plt.title("Feature Heatmap by Safety Classification", fontsize=18)
    plt.xlabel("Features")
    plt.ylabel("Safety Classification")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


# Route to display feature importance
@main_bp.route('/feature_importance')
def feature_importance_chart():
    plt.figure(figsize=(18, 8))
    sns.barplot(
        x='Importance', y='Feature', data=feature_importance, palette="viridis"
    )
    plt.title("Feature Importance in Determining Safety Status", fontsize=18)
    plt.xlabel("Importance")
    plt.ylabel("Features")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')