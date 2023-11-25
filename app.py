from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from flask_bootstrap import Bootstrap


app = Flask(__name__)
Bootstrap(app)

# Load the dataset
path_dataset = "data/harga_rumah.csv"
data = pd.read_csv(path_dataset)

# Features (independent variables)
features = data.columns[:-1]

# Home route
@app.route('/')
def home():
    return render_template('index.html', features=features)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    user_input = [float(request.form[feature]) for feature in ["luas", "kasur", "km"]]

    # Load the trained model
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(data[["luas", "kasur", "km"]], data["harga"])

    # Make a prediction
    prediction = model.predict([user_input])[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
