import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model once when the app starts
loaded_model = pickle.load(open("decision_tree_model.pkl", "rb"))

# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 7)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Retrieve form data and convert to a list of floats
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        # Get prediction
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
           prediction = "Given Transaction is fraudulent. This means the transaction has characteristics typical of fraudulent behavior and may require further investigation or action."
        else:
           prediction = "Given Transaction is NOT fraudulent. This indicates that the transaction does not match common patterns of fraud and is likely legitimate."


        
        # Pass prediction to the result template
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
