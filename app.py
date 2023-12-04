from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn import preprocessing


app = Flask(...)

model = pickle.load(open("...", "rb"))
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["..."])
def prediction():
    if request.method == "...":
        # Terima input features dari user dalam bentuk float
        # Pastikan request form menerima key yang sama dari atribut name
        # Yang ada di input index.html
        rm = float(request.form["RM"])
        lstat = float(request.form["LSTAT"])
        ptratio = float(request.form["PTRATIO"])
        
        # Karena modelnya ditrain dalam bentuk Pandas DF,
        # jadi input user harus diolah juga dalam bentuk Pandas DF
        df_prediction = pd.DataFrame({
            ...
        })
        
        # Sedikit pre-processing, karena di data training datanya ditransform,
        # Maka input datanya juga harus ditransform
        df_prediction[['RM', 'LSTAT', 'PTRATIO']] = data_scaler.fit_transform(
            df_prediction[['RM', 'LSTAT', 'PTRATIO']])
        
        # Lakukan prediksi ke target
        result_medv = model....(df_prediction)
        return render_template("prediction.html", medv=result_medv, rm=rm, lstat=lstat, ptratio=ptratio)


if __name__ == "__main__":
    app.run()
