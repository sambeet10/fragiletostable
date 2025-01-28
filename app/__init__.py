from flask import Flask
from .routes import main_bp  # Import the routes blueprint

def create_app():
    app = Flask(__name__)  # Create the Flask app instance
    
    # Register the blueprint for routes
    app.register_blueprint(main_bp)
    
    return app
