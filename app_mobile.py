import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow_hub as hub  # Import TensorFlow Hub

# Custom objects dictionary to load models with custom layers
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the pre-trained model with custom objects
model = load_model('C:/Users/vatsa/Downloads/house_plant_species/model/20240921-2014-full-image-set-mobilenetv2-Adam.h5', custom_objects=custom_objects)

# Initialize the Flask app
app = Flask(__name__)

# Define the class names for the 47 plant species
class_names = [
    'African Violet (Saintpaulia ionantha)', 'Aloe Vera',
    'Anthurium (Anthurium andraeanum)', 'Areca Palm (Dypsis lutescens)',
    'Asparagus Fern (Asparagus setaceus)', 'Begonia (Begonia spp.)',
    'Bird of Paradise (Strelitzia reginae)', 'Birds Nest Fern (Asplenium nidus)',
    'Boston Fern (Nephrolepis exaltata)', 'Calathea',
    'Cast Iron Plant (Aspidistra elatior)', 'Chinese Money Plant (Pilea peperomioides)',
    'Chinese evergreen (Aglaonema)', 'Christmas Cactus (Schlumbergera bridgesii)',
    'Chrysanthemum', 'Ctenanthe', 'Daffodils (Narcissus spp.)', 'Dracaena',
    'Dumb Cane (Dieffenbachia spp.)', 'Elephant Ear (Alocasia spp.)',
    'English Ivy (Hedera helix)', 'Hyacinth (Hyacinthus orientalis)',
    'Iron Cross begonia (Begonia masoniana)', 'Jade plant (Crassula ovata)',
    'Kalanchoe', 'Lilium (Hemerocallis)', 'Lily of the valley (Convallaria majalis)',
    'Money Tree (Pachira aquatica)', 'Monstera Deliciosa (Monstera deliciosa)', 'Orchid',
    'Parlor Palm (Chamaedorea elegans)', 'Peace lily', 'Poinsettia (Euphorbia pulcherrima)',
    'Polka Dot Plant (Hypoestes phyllostachya)', 'Ponytail Palm (Beaucarnea recurvata)',
    'Pothos (Ivy arum)', 'Prayer Plant (Maranta leuconeura)',
    'Rattlesnake Plant (Calathea lancifolia)', 'Rubber Plant (Ficus elastica)',
    'Sago Palm (Cycas revoluta)', 'Schefflera', 'Snake plant (Sanseviera)',
    'Tradescantia', 'Tulip', 'Venus Flytrap', 'Yucca', 'ZZ Plant (Zamioculcas zamiifolia'
]  # Replace with your actual species names

# Homepage route
@app.route('/')
def index():
    return render_template('index_mobile.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file:
        # Open the uploaded image and preprocess it
        image = Image.open(file)
        
        # Convert image to RGBA (to ensure no alpha channel)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = image.resize((224, 224))  # Adjust size as per your model
        image = np.array(image) / 255.0  # Normalize if necessary
        image = np.expand_dims(image, axis=0)

        # Make the prediction
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]

        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
