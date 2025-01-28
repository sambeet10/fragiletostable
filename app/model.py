import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import folium

from flask import render_template
from branca.element import Template, MacroElement
from jinja2 import Template
import os

# Load data
fsi_data = pd.read_csv('./data/FSI_Data_2006_2023.csv')
lat_lon_data = pd.read_csv('./data/Country_Latitude_Longitude.csv')
merged_data = pd.merge(fsi_data, lat_lon_data, on='Country', how='left').dropna(subset=['latitude', 'longitude'])

# Features for clustering
clustering_features = [
    'Security Apparatus', 'Factionalized Elites', 'Group Grievance', 'Economy', 
    'Economic Inequality', 'Human Flight and Brain Drain', 'State Legitimacy', 
    'Public Services', 'Human Rights', 'Demographic Pressures', 
    'Refugees and IDPs', 'External Intervention', 'Total'
]

filtered_data = merged_data.dropna(subset=clustering_features)
kmeans = KMeans(n_clusters=3, random_state=42)
filtered_data['Cluster'] = kmeans.fit_predict(filtered_data[clustering_features])

# Map clusters to safety statuses
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=clustering_features)
sorted_indices = cluster_centers.mean(axis=1).argsort()
cluster_mapping = {
    sorted_indices[0]: 'Low Risk: Minimal risk to travelers and residents.',
    sorted_indices[1]: 'Medium Risk: requiring caution and awareness.',
    sorted_indices[2]: 'High Risk: potentially requiring avoidance or careful planning.'
}
filtered_data['Safety_Status'] = filtered_data['Cluster'].map(cluster_mapping)

def predict_safety_hybrid(country_name):
    country_data = filtered_data[filtered_data['Country'] == country_name]
    if country_data.empty:
        return f"Country '{country_name}' not found in the dataset."
    
    country_data = country_data.sort_values(by='Year', ascending=False)
    recent_years = country_data['Year'].unique()
    weights = [10, 8, 6, 4, 2] + [1] * max(0, len(recent_years) - 5)
    weighted_status = Counter()

    for i, year in enumerate(recent_years):
        year_data = country_data[country_data['Year'] == year]
        status_counts = year_data['Safety_Status'].value_counts()
        for status, count in status_counts.items():
            weighted_status[status] += count * weights[min(i, len(weights) - 1)]
    
    predicted_status = weighted_status.most_common(1)[0][0]
    recent_safety_statuses = country_data.groupby('Year')['Safety_Status'].agg(lambda x: x.mode()[0]).reset_index()
    
    if len(recent_safety_statuses) > 1:
        def map_risk(status):
            if "Low Risk" in status:
                return 1
            elif "Medium Risk" in status:
                return 0
            elif "High Risk" in status:
                return -1
            return None
        
        recent_safety_statuses['Trend_Score'] = recent_safety_statuses['Safety_Status'].apply(map_risk)
        trend_diff = recent_safety_statuses['Trend_Score'].diff()
        trend_score = trend_diff.sum()

        trend = "Improving" if trend_score > 0 else "Deteriorating" if trend_score < 0 else "Stable"
    else:
        trend = "No Significant Change"
    
    return predicted_status, trend

def generate_safety_map(output_path):
    map_data = filtered_data[['Country', 'Safety_Status', 'latitude', 'longitude']].drop_duplicates()
    world_map = folium.Map(location=[10, 20], zoom_start=2)

    # Add legend to the map
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: 180px; background-color: white; z-index:9999; font-size:14px; border:2px solid grey; border-radius:10px; padding: 10px;">
    <b>Safety Classification</b><br>
    <span style="color: green;">● Low Risk</span>: Minimal risk to travelers and residents.<br>
    <span style="color: orange;">● Medium Risk</span>: requiring caution and awareness.<br>
    <span style="color: red;">● High Risk</span>: potentially requiring avoidance or careful planning.<br>
    </div>
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    world_map.get_root().add_child(legend)

    # Add circle markers for each country
    for _, row in map_data.iterrows():
        predicted_status, trend = predict_safety_hybrid(row['Country'])
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"""
            <b>Country:</b> {row['Country']}<br>
            <b>Safety Status:</b> {predicted_status}<br>
            <b>Trend:</b> {trend}
            """,
            color='green' if 'Low Risk' in predicted_status else 
                   'orange' if 'Medium Risk' in predicted_status else 'red',
            fill=True,
            fill_opacity=0.7
        ).add_to(world_map)

    # Save the map to the static folder
    static_folder = os.path.join('app', 'static')
    os.makedirs(static_folder, exist_ok=True)  # Ensure the static folder exists
    map_path = os.path.join(static_folder, 'safety_map.html')
    world_map.save(map_path)
    print(f"Map saved to {map_path}")

from sklearn.preprocessing import StandardScaler

# Generate Heatmap Visualization Data
grouped_data = filtered_data.groupby('Safety_Status')[clustering_features].mean()

# Standardize the data
scaler = StandardScaler()
standardized_data = pd.DataFrame(
    scaler.fit_transform(grouped_data),
    index=grouped_data.index,
    columns=grouped_data.columns
)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to compute feature importance
def analyze_factors_with_random_forest(data, features, target_col='Cluster'):
    rf_data = data.dropna(subset=features + [target_col])  # Filter relevant data
    X = rf_data[features]
    y = rf_data[target_col]
    
    # Split data for model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model performance (accuracy is not really needed here, but you can log it)
    y_pred = rf_model.predict(X_test)
    print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Extract feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance

# Ensure to call this function and assign the result to `feature_importance` when loading the model
feature_importance = analyze_factors_with_random_forest(filtered_data, clustering_features)
