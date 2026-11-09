from flask import Flask, request, jsonify, render_template
import pickle
from pathlib import Path

app = Flask(__name__)

# Load ML model
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model.pkl", "rb") as file:
    model = pickle.load(file)

flower_names = ["Setosa", "Versicolor", "Virginica"]


# Homepage
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = data["features"]

        if len(features) != 4:
            return jsonify({"error": "Exactly four features required"}), 400

        features = [float(value) for value in features]

        prediction = model.predict([features])

        result = flower_names[int(prediction[0])]

        return jsonify({"prediction": result})

    except (TypeError, ValueError, KeyError):
        return jsonify({"error": "Invalid input"}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)