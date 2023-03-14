import io
import numpy as np
import onnxruntime
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

inference = onnxruntime.InferenceSession('models/mobilenetv2-12-int8.onnx')
with open('static/mobilenet_labels.txt', 'r') as f:
    class_dict = [line.strip() for line in f.readlines()]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes))

    # Image preprocessing
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    input_name = inference.get_inputs()[0].name
    output_name = inference.get_outputs()[0].name

    output_data = inference.run([output_name], {input_name: img})[0]
    class_index = np.argmax(output_data)
    predicted_class = class_dict[class_index]

    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run()
