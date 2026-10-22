import os
import gc
import json
import uuid
import traceback
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from PIL import Image

app = Flask(__name__)

# ── Cấu hình ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, 'saved_models')
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
IMG_SIZE    = (128, 128)
CLASS_NAMES = ['cat', 'dog']
CLASS_VI    = {'cat': 'Mèo 🐱', 'dog': 'Chó 🐶'}
MAX_CONTENT = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT

# ── Load mô hình (lazy + nhẹ RAM) ───────────────────────────
model      = None
model_info = {}

def load_model():
    """Load model, cấu hình TF để dùng ít RAM nhất có thể."""
    global model, model_info

    # Giảm memory footprint cho Render free tier (512 MB RAM)
    try:
        import tensorflow as tf
        # Chỉ dùng CPU, không cấp phát GPU memory
        tf.config.set_visible_devices([], 'GPU')
        # Không pre-allocate toàn bộ RAM
        for dev in tf.config.list_physical_devices('CPU'):
            try:
                tf.config.experimental.set_memory_growth(dev, True)
            except Exception:
                pass
        # Tắt log verbose TF
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except Exception as e:
        print(f'TF config warning: {e}')

    keras_path = os.path.join(MODEL_DIR, 'best_model_final.keras')
    if os.path.exists(keras_path):
        try:
            print(f'Loading model: {keras_path}')
            from tensorflow import keras
            model = keras.models.load_model(keras_path, compile=False)
            # Warm-up: chạy predict 1 lần để JIT compile xong trước khi có request
            dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
            model.predict(dummy, verbose=0)
            gc.collect()
            print('Model loaded & warmed up OK!')
        except Exception as e:
            print(f'ERROR loading model: {e}')
            traceback.print_exc()
            model = None
    else:
        print(f'WARNING: Model not found at {keras_path}')

    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                model_info = json.load(f)
        except Exception:
            pass

load_model()

# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image(image_path):
    """Mở ảnh, resize về IMG_SIZE, chuẩn hóa [0,1]."""
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(image_path):
    """Chạy inference, trả về (label, confidence, error_msg)."""
    if model is None:
        return None, 0.0, 'Model chưa sẵn sàng. Kiểm tra logs server.'

    img_array  = preprocess_image(image_path)
    prob       = float(model.predict(img_array, verbose=0)[0][0])
    label      = CLASS_NAMES[int(prob > 0.5)]
    confidence = prob if prob > 0.5 else 1 - prob
    gc.collect()   # Giải phóng RAM ngay sau predict
    return label, confidence, None

# ── Global error handlers – luôn trả JSON ────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    """Bắt mọi exception không được xử lý, trả JSON thay vì HTML."""
    print(f'Unhandled exception: {e}')
    traceback.print_exc()
    return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File quá lớn. Tối đa 10 MB.'}), 413

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    stats = {}
    if model_info:
        stats['model_name'] = model_info.get('best_model', 'N/A')
        results = model_info.get('results', [])
        best = next((r for r in results if r['name'] == stats['model_name']), {})
        stats['accuracy'] = f"{float(best.get('accuracy', 0)):.1%}" if best else 'N/A'
        stats['auc']      = f"{float(best.get('auc', 0)):.3f}"      if best else 'N/A'
    return render_template('index.html', stats=stats, model_ready=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint nhận ảnh → JSON kết quả. Luôn trả JSON dù có lỗi."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy file trong request.'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'Chưa chọn file.'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': f'Định dạng không hỗ trợ. Chấp nhận: {", ".join(ALLOWED_EXT)}'}), 400

        # Lưu file
        ext       = file.filename.rsplit('.', 1)[1].lower()
        filename  = f'{uuid.uuid4().hex}.{ext}'
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # Dự đoán
        label, confidence, error = predict_image(save_path)
        if error:
            return jsonify({'error': error}), 503

        label_vi = CLASS_VI[label]
        color    = '#4FC3F7' if label == 'cat' else '#EF9A9A'
        cat_prob = round((1 - confidence) * 100 if label == 'dog' else confidence * 100, 1)
        dog_prob = round(100 - cat_prob, 1)

        return jsonify({
            'label'         : label,
            'label_vi'      : label_vi,
            'confidence'    : f'{confidence:.2%}',
            'confidence_raw': round(confidence * 100, 1),
            'cat_prob'      : cat_prob,
            'dog_prob'      : dog_prob,
            'color'         : color,
            'image_url'     : url_for('serve_upload', filename=filename)
        })

    except Exception as e:
        print(f'Error in /predict: {e}')
        traceback.print_exc()
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/health')
def health():
    return jsonify({
        'status'     : 'ok',
        'model_ready': model is not None,
        'model_name' : model_info.get('best_model', 'N/A')
    })

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
