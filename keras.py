import tensorflow as tf
from PIL import Image
import numpy as np
import flask 
import io

app = flask.Flask(__name__)
model = None

def load_model():
	
	global model

	model = tf.keras.applications.ResNet50(weights="imagenet")

def prepare_dataset(image, target):

	if image.mode != "RGB":
		image= image.convert("RGB")

	image = image.resize(target)
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = tf.keras.applications.imagenet_utils.preprocess_input(image)

	return image

@app.route("/predict", methods=["POST"])

def predict():

	data = {"success":False}

	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			image = prepare_dataset(image, target=(224,224))

			preds = model.predict(image)
			results = tf.keras.applications.imagenet_utils.decode_predictions(preds)

			data["predictions"] = []

			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probablity": float(prob)}
				data["predictions"].append(r)

			data["success"] = True

	return flask.jsonify(data)

if __name__ == "__main__":
	print("Loading Keras")
	load_model()
	app.run()


