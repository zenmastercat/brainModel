from flask import Flask, request, jsonify
import torch, cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import get_boxes, load_model

app = Flask(__name__)
model = load_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    image = Image.open(file).convert("L").resize((256, 256))
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    predicted = torch.argmax(output, dim=1).squeeze().numpy()
    boxes = get_boxes(predicted)
    return jsonify({
        "classification": "Possible lesion detected",
        "boxes": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in boxes]
    })

if __name__ == "__main__":
    app.run()
