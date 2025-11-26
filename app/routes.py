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
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2000,
        "response_mime_type": "text/plain",
    }
    
    # Configure safety settings to be less restrictive
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH", 
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
    
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    
    try:
        response = model.generate_content(prompt)
        
        # Check if response was blocked by safety filters
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                # Safety block - return a generic informative response
                return "I understand you're looking for travel information. For the most current and detailed guidance about this destination, I recommend consulting official government travel advisories, local tourism boards, and recent traveler reviews. Each destination has unique considerations for visitors."
            
            # Try to extract text content
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    raw_content = candidate.content.parts[0].text
                else:
                    raw_content = str(candidate.content)
            else:
                raw_content = "Unable to generate detailed recommendations at this time."
        else:
            # Fallback if no candidates
            if hasattr(response, 'text') and response.text:
                raw_content = response.text
            else:
                raw_content = "Unable to generate recommendations. Please try again."
        
        return clean_response(raw_content)
        
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "I apologize, but I'm unable to generate recommendations at this time. Please try again later or consult official travel resources."



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
                f"Please provide comprehensive travel guidance for {country}. "
                f"This destination has a current stability classification of '{safety_status}' with a '{trend}' outlook. "
                "Please structure your response with clear sections covering:\n"
                "**Planning Your Visit**: Optimal travel times and preparation tips\n"
                "**Cultural Experiences**: Major attractions, local customs, and unique experiences\n"
                "**Practical Information**: Transportation, accommodation, and local guidelines\n"
                "**Health and Wellness**: Medical facilities and health considerations\n"
                "**Local Insights**: Currency, communication, and helpful travel tips\n"
                "Format your response with clear headings and bullet points for easy reading."
            )
        elif intent.lower() == "settlement":
            prompt = (
                f"Please provide comprehensive relocation guidance for {country}. "
                f"This destination has a current stability classification of '{safety_status}' with a '{trend}' outlook. "
                "Please structure your response with clear sections covering:\n"
                "**Quality of Life**: Living standards, infrastructure, and lifestyle\n"
                "**Economic Landscape**: Employment opportunities and business environment\n"
                "**Financial Considerations**: Cost of living, housing, and expenses\n"
                "**Social Environment**: Community, culture, and integration aspects\n"
                "**Practical Requirements**: Legal processes, documentation, and logistics\n"
                "Format your response with clear headings and bullet points for easy reading."
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