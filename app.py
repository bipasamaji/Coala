# ui_map_app.py
import gradio as gr
import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear

# Load model and data
model = joblib.load("advanced_model.pkl")
df = pd.read_csv("advanced_coal_data.csv")

# Dummy coordinates for mines (to be replaced with real ones)
coordinates = [
    (23.6102, 85.2799), (23.6842, 86.9869), (22.7896, 88.4314),
    (24.2073, 84.8702), (22.8046, 86.2029), (23.8315, 85.3825),
    (23.6693, 85.3391), (23.5302, 87.3076), (23.0703, 85.2743),
    (22.7749, 85.1781)
] * 100  

# Helper to determine zone
def get_zone(lifetime):
    if lifetime >= 20:
        return "Safe", "green"
    elif lifetime >= 10:
        return "Warning", "orange"
    else:
        return "Danger", "red"

# Prediction function
def predict_lifetime(depth, gas_methane, gas_co2, temperature, humidity,
                     pressure, accidents, soil_type, water_nearby, mining_intensity):
    input_data = pd.DataFrame([{ 
        "depth": depth,
        "gas_methane": gas_methane,
        "gas_co2": gas_co2,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "accidents": accidents,
        "soil_type": soil_type,
        "water_nearby": water_nearby,
        "mining_intensity": mining_intensity,
    }])
    lifetime = model.predict(input_data)[0]
    zone, color = get_zone(lifetime)
    return f"Predicted Lifetime: {lifetime:.2f} years (Zone: {zone})"

# Folium map renderer
def render_map():
    m = folium.Map(location=[23.6, 85.3], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in df.iterrows():
        lat, lon = coordinates[idx]
        zone, color = get_zone(row["lifetime"])
        folium.CircleMarker(
            location=(lat, lon),
            radius=7,
            popup=f"Lifetime: {row['lifetime']} yrs\nZone: {zone}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(marker_cluster)
    return m

map_component = gr.components.HTML(render_map()._repr_html_(), label="Mine Safety Zones Map")

# Gradio UI
def ui():
    with gr.Blocks(css="body { background-color: #f9f9f9; font-family: 'Segoe UI'; }") as demo:
        gr.Markdown("## 🏭 Advanced Coal Mine Lifetime & Safety Zone Predictor")
        with gr.Row():
            with gr.Column():
                depth = gr.Slider(800, 2000, label="Depth (m)")
                gas_methane = gr.Slider(80, 300, label="Methane Level (ppm)")
                gas_co2 = gr.Slider(50, 200, label="CO₂ Level (ppm)")
                temperature = gr.Slider(30, 55, step=0.1, label="Temperature (°C)")
                humidity = gr.Slider(60, 95, step=0.1, label="Humidity (%)")
                pressure = gr.Slider(95, 105, step=0.1, label="Pressure (kPa)")
                accidents = gr.Slider(0, 10, label="Accident Count")
                soil_type = gr.Dropdown(["rocky", "sandy", "clay"], label="Soil Type")
                water_nearby = gr.Radio(["yes", "no"], label="Water Nearby")
                mining_intensity = gr.Dropdown(["low", "medium", "high"], label="Mining Intensity")
                predict_btn = gr.Button("Predict Lifetime")
            with gr.Column():
                output = gr.Textbox(label="Prediction Output")
                map_html = gr.HTML(render_map()._repr_html_())
        predict_btn.click(
            predict_lifetime,
            inputs=[depth, gas_methane, gas_co2, temperature, humidity, pressure,
                    accidents, soil_type, water_nearby, mining_intensity],
            outputs=output
        )
    return demo

# Launch
app = ui()

if __name__ == "__main__":
    app.launch()
