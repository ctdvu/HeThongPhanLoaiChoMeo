import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# ── Cấu hình ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(os.path.dirname(BASE_DIR), 'saved_models')
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
IMG_SIZE    = (128, 128)
CLASS_NAMES = ['cat', 'dog']
CLASS_VI    = {'cat': 'Mèo 🐱', 'dog': 'Chó 🐶'}
MAX_CONTENT = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT

# ── Load mô hình ─────────────────────────────────────────────
model = None
model_info = {}

def load_model():
    global model, model_info
    keras_path = os.path.join(MODEL_DIR, 'best_model_final.keras')
    if os.path.exists(keras_path):
        print(f'Loading model: {keras_path}')
        model = keras.models.load_model(keras_path)
        print('Model loaded successfully!')
    else:
        print(f'WARNING: Model not found at {keras_path}')
        print('Please run the training notebook first.')

    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)

load_model()

# ── Helper functions ─────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image(image_path):
    """Tiền xử lý ảnh: resize + normalize."""
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(image_path):
    """Dự đoán và trả về kết quả."""
    if model is None:
        return None, 0, 'Model chưa được load. Hãy chạy notebook training trước!'

    img_array = preprocess_image(image_path)
    prob      = float(model.predict(img_array, verbose=0)[0][0])
    label     = CLASS_NAMES[int(prob > 0.5)]
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, None

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    """Trang chủ."""
    stats = {}
    if model_info:
        stats['model_name'] = model_info.get('best_model', 'N/A')
        results = model_info.get('results', [])
        best = next((r for r in results if r['name'] == stats['model_name']), {})
        stats['accuracy'] = f"{float(best.get('accuracy', 0)):.1%}" if best else 'N/A'
        stats['auc']      = f"{float(best.get('auc', 0)):.3f}" if best else 'N/A'
    return render_template('index.html', stats=stats, model_ready=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận ảnh và trả về kết quả phân loại."""
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file trong request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Định dạng không hỗ trợ. Chấp nhận: {", ".join(ALLOWED_EXT)}'}), 400

    # Lưu file tạm
    import uuid
    ext       = file.filename.rsplit('.', 1)[1].lower()
    filename  = f'{uuid.uuid4().hex}.{ext}'
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    # Dự đoán
    label, confidence, error = predict_image(save_path)
    if error:
        return jsonify({'error': error}), 500

    label_vi = CLASS_VI[label]
    color    = '#4FC3F7' if label == 'cat' else '#EF9A9A'

    return jsonify({
        'label'      : label,
        'label_vi'   : label_vi,
        'confidence' : f'{confidence:.2%}',
        'confidence_raw': round(confidence * 100, 1),
        'cat_prob'   : round((1 - confidence) * 100 if label == 'dog' else confidence * 100, 1),
        'dog_prob'   : round(confidence * 100 if label == 'dog' else (1 - confidence) * 100, 1),
        'color'      : color,
        'image_url'  : url_for('serve_upload', filename=filename)
    })

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Phục vụ file ảnh đã upload."""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/health')
def health():
    """Health check endpoint cho Render."""
    return jsonify({
        'status'     : 'ok',
        'model_ready': model is not None,
        'model_name' : model_info.get('best_model', 'N/A')
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
