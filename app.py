from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__, template_folder="templates")

# Load the trained model
with open('House.pkl', 'rb') as f:
    model = pickle.load(f)

# Home Route (renders the HTML form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        Rooms = int(request.form['bedrooms'])
        Bathrooms = int(request.form['bathrooms'])
        Place = int(request.form['location'])
        Area = int(request.form['area'])
        Status = int(request.form['status'])
        Facing = int(request.form['facing'])
        P_Type = int(request.form['type'])

        # Convert input data into NumPy array
        input_data = np.array([[Rooms, Bathrooms, Place, Area, Status, Facing, P_Type]])

        # Predict house price
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
