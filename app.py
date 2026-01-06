
from flask import Flask, render_template, request
import torch
from PIL import Image
import model
import numpy as np
import os
import io
import base64
import cv2
from flask import jsonify, send_file
import zipfile
import tarfile
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid

app = Flask(__name__)
net = model.CNN()
# Load pretrained weights if available
MODEL_PATH = "model.pth"
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        # support checkpoints that wrap weights in a 'state_dict' key
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        net.load_state_dict(state, strict=False)
        print(f"Loaded model weights (partial) from {MODEL_PATH}")
    except Exception as e:
        print("Failed to load model.pth:", e)
else:
    print("model.pth not found — running with random weights")
net.eval()


@app.route("/", methods=["GET","POST"]) 
def index():
    result=None
    if request.method=="POST":
        img = Image.open(request.files["image"]).convert("RGB").resize((224,224))
        arr = np.array(img)/255.0
        x = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()
        out = net(x)
        pred = out.softmax(dim=1).detach().numpy()[0]
        label = int(out.argmax().item())
        result = "DEFECT" if label==0 else "NORMAL"
    return render_template("index.html", result=result)


def tensor_from_pil(img_pil):
    arr = np.array(img_pil)/255.0
    return torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()


def grad_cam(net, input_tensor, target_class=None):
    # simple Grad-CAM implementation for this small CNN
    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    # target the last conv layer
    target_layer = net.conv[3]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    net.zero_grad()
    out = net(input_tensor)
    if target_class is None:
        target_class = int(out.argmax().item())
    loss = out[0, target_class]
    loss.backward()

    fh.remove(); bh.remove()

    weights = gradients.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.squeeze().cpu().numpy()
    cam_resized = cv2.resize(cam_np, (224,224))
    return cam_resized


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # accepts an uploaded image (file) or raw blob
        if 'image' in request.files:
            img = Image.open(request.files['image']).convert('RGB').resize((224,224))
        else:
            return jsonify({'error':'no image provided'}), 400

        input_t = tensor_from_pil(img)
        out = net(input_t)
        probs = out.softmax(dim=1).detach().cpu().numpy()[0].tolist()
        label = int(out.argmax().item())
        result = 'NORMAL' if label==0 else 'DEFECT'

        # generate grad-cam heatmap
        cam = grad_cam(net, input_t, target_class=label)
        # overlay on image
        img_np = np.array(img)
        heatmap = (255 * cam).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np[...,::-1], 0.6, heatmap_color, 0.4, 0)
        _, buf = cv2.imencode('.png', overlay[...,::-1])
        heat_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        # produce a simple parts table based on heatmap peaks (heuristic)
        thresh = (cam > 0.6).astype(np.uint8)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts = []
        for i, cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            parts.append({'part_id': f'P{i+1}', 'bbox':[int(x),int(y),int(w),int(h)], 'severity': float(cam[y:y+h,x:x+w].mean())})

        # decide fixability heuristically
        for p in parts:
            p['fixable'] = 'Yes' if p['severity'] < 0.85 else 'No'

        # Generate unique chip ID and timestamp
        chip_id = f"CHIP-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Format confidence as percentage
        confidence_pct = max(probs) * 100
        
        report_html = render_template('report_fragment.html', chip_id=chip_id, image_b64=heat_b64, result=result, probs=probs, confidence_pct=confidence_pct, timestamp=timestamp, parts=parts)

        return jsonify({'result': result, 'probs': probs, 'heatmap_b64': heat_b64, 'report_html': report_html, 'parts': parts, 'chip_id': chip_id, 'timestamp': timestamp})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print('Analyze failed:', e)
        print(tb)
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/report/download', methods=['POST'])
def report_download():
    data = request.get_json()
    if not data or 'report_html' not in data:
        return 'Missing report data', 400
    html = data['report_html']
    return send_file(io.BytesIO(html.encode('utf-8')), mimetype='text/html', as_attachment=True, download_name='report.html')



@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    # Accepts a single file upload (zip/tar or other). Saves to ./uploaded_datasets/
    if 'dataset' not in request.files:
        return jsonify({'error':'no dataset file provided'}), 400
    f = request.files['dataset']
    filename = secure_filename(f.filename)
    os.makedirs('uploaded_datasets', exist_ok=True)
    path = os.path.join('uploaded_datasets', filename)
    f.save(path)

    # try to extract archives
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall('dataset')
            info = 'extracted zip to dataset/'
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, 'r:*') as t:
                t.extractall('dataset')
            info = 'extracted tar to dataset/'
        else:
            info = f'saved to {path} (not extracted)'
    except Exception as e:
        return jsonify({'error':'failed to extract', 'exc': str(e)}), 500

    return jsonify({'status':'ok', 'message': info, 'saved_path': path})


if __name__ == "__main__":
    app.run(debug=True)
