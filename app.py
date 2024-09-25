from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('FINAL_POSIBILITY.csv')

# Extract unique states and weather conditions for dropdown options
states = df['State'].unique()
weather_conditions = df['Type Of Weather'].unique()

# Load the pre-trained pipeline
with open('f_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Get feature names (important for ensuring the correct shape after encoding)
model_columns = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Route to serve the home page
@app.route('/')
def home():
    return render_template('index.html', states=states, weather_conditions=weather_conditions)

# Route to get cities for a selected state (AJAX request)
@app.route('/get_cities/<state>', methods=['GET'])
def get_cities(state):
    # Filter cities based on the selected state
    cities_data = df[df['State'] == state][['City Name', 'Lat', 'Long']].drop_duplicates()
    cities = [{'city': row['City Name'], 'lat': row['Lat'], 'long': row['Long']} for _, row in cities_data.iterrows()]

    return jsonify({'cities': cities})

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        city_name = request.form['city']
        state = request.form['state']
        weather = request.form['weather_condition']
        lat = float(request.form['lat'])
        long = float(request.form['long'])

        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'City Name': [city_name],
            'Lat': [lat],
            'Long': [long],
            'State': [state],
            'Type Of Weather': [weather]
        })

        #___________________________________

        # Apply the same preprocessing as the training pipeline
        #input_data_encoded = model_pipeline.named_steps['preprocessor'].transform(input_data)

        # Ensure the encoded data has the correct structure (align columns)
        #input_data_encoded = pd.DataFrame(input_data_encoded, columns=model_columns)

        # Check the shape of the transformed input data
        #print(f"Transformed input shape: {input_data_encoded.shape}")  # Debugging print

        # Make prediction using the trained model
        #prediction = model_pipeline.named_steps['model'].predict(input_data_encoded)
        #____________________________
        # Apply the same preprocessing as the training pipeline
        input_data_encoded = model_pipeline.named_steps['preprocessor'].transform(input_data)

        # Make prediction using the trained model
        prediction = model_pipeline.named_steps['model'].predict(input_data_encoded)


        # Map the prediction back to the label (e.g., 0 -> 'Low', 1 -> 'Medium', 2 -> 'High')
        possibility_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        prediction_label = possibility_mapping[prediction[0]]

        # Render the result back to the page
        return render_template('index.html', prediction=f"The possibility of accident in {city_name} during {weather} is {prediction_label}.", states=states, weather_conditions=weather_conditions)

    except Exception as e:
        return render_template('index.html', error=str(e), states=states, weather_conditions=weather_conditions)

if __name__ == '__main__':
    app.run(debug=True)

