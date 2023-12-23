from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn import preprocessing


app = Flask(__name__)

model = pickle.load(open("linear_regression_model.pkl", "rb"))
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    if request.method == "POST":
        # Terima input features dari user dalam bentuk float
        # Pastikan request form menerima key yang sama dari atribut name
        # Yang ada di input index.html
        rm = float(request.form["RM"])
        lstat = float(request.form["LSTAT"])
        ptratio = float(request.form["PTRATIO"])
        
        # Karena modelnya ditrain dalam bentuk Pandas DF,
        # jadi input user harus diolah juga dalam bentuk Pandas DF
        df_prediction = pd.DataFrame({
            "RM": [rm],
            "LSTAT": [lstat],
            "PTRATIO": [ptratio]
        })
        
        # Sedikit pre-processing, karena di data training datanya ditransform,
        # Maka input datanya juga harus ditransform
        df_prediction[['RM', 'LSTAT', 'PTRATIO']] = data_scaler.fit_transform(
            df_prediction[['RM', 'LSTAT', 'PTRATIO']])
        
        # Lakukan prediksi ke target
        result_medv = model.predict(df_prediction)[0]
        return render_template("prediction.html", medv=result_medv, rm=rm, lstat=lstat, ptratio=ptratio)

@app.route("/api/prediction", methods=["POST"])
def api_prediction():
    try:
        # Proses data input
        rm = float(request.form["RM"])
        lstat = float(request.form["LSTAT"])
        ptratio = float(request.form["PTRATIO"])

        df_prediction = pd.DataFrame({
            "RM": [rm],
            "LSTAT": [lstat],
            "PTRATIO": [ptratio]
        })

        df_prediction[['RM', 'LSTAT', 'PTRATIO']] = data_scaler.fit_transform(df_prediction[['RM', 'LSTAT', 'PTRATIO']])

        # Lakukan prediksi
        result_medv = model.predict(df_prediction)[0]

        # Kembalikan hasil dalam format JSON
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting"
            },
            "data": {
                "medv_result": result_medv
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