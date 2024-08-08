from flask import Flask, request, render_template
import pickle
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model (change the path to where your model is stored)
model = joblib.load('model.joblib')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        input1 = float(request.form['input1'])
        input2 = float(request.form['input2'])
        input3 = float(request.form['input3'])
        
        # Create an input array
        input_data = np.array([[input1, input2, input3]])
        
        # Check if the model has been loaded correctly
        if hasattr(model, 'predict'):
            # Predict the class using the loaded model
            prediction = model.predict(input_data)[0]
            
            # Convert numerical prediction to label (assuming 0 is 'Running' and 1 is 'Falling')
            if prediction == 0:
                prediction_label = 'Running'
            else:
                prediction_label = 'Falling'
            
            # Render the prediction result
            return render_template('index.html', prediction=prediction_label)
        else:
            return "Model does not have a 'predict' method. Ensure the correct model is loaded."

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
