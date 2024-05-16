import os
from flask import Flask, render_template, request
from keras.preprocessing import image
import numpy as np
from googlesearch import search
import pandas as pd
import platform
import tensorflow as tf


if platform.system() == "Darwin" and platform.processor() == "arm":
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)
else:
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Load the disease solutions from Excel file
solutions_df = pd.read_excel('leaf_disease_dataset.xlsx')
solutions_df['Class'] = solutions_df['Class'].str.replace('-', '_')
solutions = dict(zip(solutions_df['Class'], solutions_df['Solution']))


# Load the pre-trained model
model = tf.keras.models.load_model('improved_leaf_disease_model.h5')


# List of class labels
labels = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy', 'Blueberry__healthy',
          'Cherry(including_sour)__healthy', 'Cherry(including_sour)___Powdery_mildew',
          'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)_Common_rust', 'Corn(maize)___healthy',
          'Corn_(maize)__Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape___healthy',
          'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing(Citrus_greening)',
          'Peach___Bacterial_spot', 'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy',
          'Potato_Early_blight', 'Potato__healthy', 'Potato__Late_blight', 'Raspberry_healthy', 'Soybean_healthy',
          'Squash_Powdery_mildew', 'Strawberry__healthy', 'Strawberry__Leaf_scorch', 'Tomato_Bacterial_spot',
          'Tomato_Early_blight', 'Tomato_healthy', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
          'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot',
          'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']

# Function to perform Google search
def google_search(query, num_results=5):
    try:
        return list(search(query, num_results=num_results))
    except Exception as e:
        print(f"Google search failed: {str(e)}")
        return []

# Flask route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the form
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Create the 'static/uploads' folder if it doesn't exist
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')

            # Save the uploaded file to 'static/uploads'
            image_filename = uploaded_file.filename
            image_path = os.path.join('static/uploads', image_filename)
            uploaded_file.save(image_path)

            # Preprocess the image
            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)
            predicted_class_label = labels[predicted_class_index[0]]

            # Get the solution for the predicted class
            solution = solutions.get(predicted_class_label, 'No specific solution available.')

            # Perform Google search for solutions
            query = f"{predicted_class_label} disease solutions"
            search_results = google_search(query)
            
            context = {
                "predicted_class": predicted_class_label,
                "solution": solution,
                "image_path": image_path, 
                "image_filename": image_filename,
                "search_results_json": search_results 
            }

            # Render the result template with the predicted class, image path, solution, and search results
            return render_template('result.html', **context)

    # Render the main page template
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
