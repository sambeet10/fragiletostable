from app import create_app  # Import the create_app function
from app.model import generate_safety_map, merged_data  # Import map generation logic and data

app = create_app()  # Call the create_app function to get the app instance

# Generate the map when the app starts
generate_safety_map(merged_data)  # This ensures safety_map.html is created

if __name__ == "__main__":
    app.run(debug=True)
