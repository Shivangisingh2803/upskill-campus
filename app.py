import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model from the pickle file
with open("result.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template('predict.html') 

@app.route('/predict', methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

        # Make prediction
    input_data = np.array([[N,P,K,temp,humidity,ph,rainfall]])
    prediction = model.predict(input_data)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Interpret prediction
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('predict.html',prediction_text = result)
if __name__ == "__main__":
    app.run(debug=True)
