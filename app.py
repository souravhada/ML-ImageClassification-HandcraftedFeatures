from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
import pickle

app = Flask(__name__)

# Load the model from the specified folder using pickle
with open("model/svm_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load the PCA model
with open("model/pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

# Define or load label mapping
label_mapping = {'Building': 0, 'Forest': 1, 'Glacier': 2, 'Mountains': 3, 'Sea': 4, 'Streets': 5}
# Inverse mapping to get labels from prediction indices
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

def preprocess_and_extract_features(input_data, fixed_length=45000):
    if not isinstance(input_data, np.ndarray):
        raise TypeError("Input data must be an image array")

    gray = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
    hog_features, _ = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True, feature_vector=True)

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_features = lbp.flatten()

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    sift_features = descriptors.flatten() if descriptors is not None else np.array([])

    combined_features = np.concatenate((hog_features, lbp_features, sift_features))
    if combined_features.size > fixed_length:
        combined_features = combined_features[:fixed_length]
    else:
        combined_features = np.pad(combined_features, (0, max(0, fixed_length - combined_features.size)), 'constant')

    return combined_features

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_data = preprocess_and_extract_features(img)

    processed_data_pca = pca.transform([processed_data])
    prediction = model.predict(processed_data_pca)

    # Get the predicted class label from the mapping
    predicted_label = inverse_label_mapping[prediction[0]]

    result = {
        'prediction': predicted_label,
        'encoded_label': int(prediction[0])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
