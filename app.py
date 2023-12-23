from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import pickle
from sklearn import preprocessing

app = Flask(__name__)
CORS(app)  # Enable CORS

model = pickle.load(open("linear_regression_model.pkl", "rb"))
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    if request.method == "POST":
        rm = float(request.form["RM"])
        lstat = float(request.form["LSTAT"])
        ptratio = float(request.form["PTRATIO"])
        df_prediction = pd.DataFrame({
            "RM": [rm],
            "LSTAT": [lstat],
            "PTRATIO": [ptratio],
        })
        df_prediction[['RM', 'LSTAT', 'PTRATIO']] = data_scaler.fit_transform(
            df_prediction[['RM', 'LSTAT', 'PTRATIO']])
        result_medv = model.predict(df_prediction)[0]
        return render_template("prediction.html", medv=result_medv, rm=rm, lstat=lstat, ptratio=ptratio)

@app.route("/api/prediction", methods=["POST"])
def api_prediction():
    try:
        input_data = request.get_json()  # Get data as JSON
        rm = float(input_data["RM"])
        lstat = float(input_data["LSTAT"])
        ptratio = float(input_data["PTRATIO"])
        df_prediction = pd.DataFrame({
            "RM": [rm],
            "LSTAT": [lstat],
            "PTRATIO": [ptratio],
        })
        df_prediction[['RM', 'LSTAT', 'PTRATIO']] = data_scaler.fit_transform(
            df_prediction[['RM', 'LSTAT', 'PTRATIO']])
        result = model.predict(df_prediction)
        result = result.tolist()  # Convert to list
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting",
            },
            "data": {
                "medv_result": result[0]  # Send the first result
            }
        })
    except Exception as e:
        return jsonify({
            "status": {
                "code": 500,
                "message": "Error in prediction: " + str(e)
            }
        })

if __name__ == "__main__":
    app.run()
