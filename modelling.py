import numpy as np
import onnxruntime
from PIL import Image

inference = onnxruntime.InferenceSession('models/mobilenetv2-12-int8.onnx')

img = Image.open('img/drone.jpg')
img = img.resize((224, 224))
img = np.array(img).astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

input_name = inference.get_inputs()[0].name
output_name = inference.get_outputs()[0].name

output_data = inference.run([output_name], {input_name: img})[0]

class_index = np.argmax(output_data)

with open('static/mobilenet_labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

predicted_class = class_names[class_index]

print('Predicted class:', predicted_class)
print('Probability:', output_data[0, class_index])
