# PawAI – Cat & Dog Classifier 🐱🐶

Web app phân loại chó mèo sử dụng CNN + Flask, deploy trên Render.

## Cấu Trúc Project

```
CNN_ChoMeo/
├── data/                      # Dataset
│   ├── train/
│   │   ├── cat/  (275 ảnh)
│   │   └── dog/  (180 ảnh)
│   └── val/
│       ├── cat/  (95 ảnh)
│       └── dog/  (46 ảnh)
├── saved_models/              # Mô hình đã train (tạo sau khi chạy notebook)
│   ├── best_model_final.keras
│   └── model_info.json
├── cnn_cat_dog_training.ipynb # Notebook huấn luyện
└── webapp/                    # Flask web app
    ├── app.py
    ├── requirements.txt
    ├── Procfile
    ├── render.yaml
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── css/style.css
    │   └── js/main.js
    └── uploads/               # Ảnh upload tạm thời
```

## Bước 1: Huấn Luyện Mô Hình

```bash
# Cài đặt thư viện
pip install tensorflow keras pillow scikit-learn matplotlib seaborn

# Chạy notebook
jupyter notebook cnn_cat_dog_training.ipynb
```

Sau khi chạy xong, folder `saved_models/` sẽ có:
- `best_model_final.keras`
- `model_info.json`

## Bước 2: Chạy Thử Local

```bash
cd webapp
pip install -r requirements.txt
python app.py
```

Mở trình duyệt: http://localhost:5000

## Bước 3: Deploy lên Render

### Chuẩn bị

1. **Upload model lên GitHub LFS hoặc cloud storage**  
   File `.keras` thường > 50MB. Dùng một trong các phương án:
   - [Git LFS](https://git-lfs.github.com/)
   - Google Drive + gdown script
   - Render Persistent Disk (trả phí)
   - Hugging Face Hub (miễn phí)

2. **Tạo file `startup.py`** để download model khi server start (xem bên dưới)

### Deploy thủ công

```bash
# 1. Tạo repo git
git init
git add .
git commit -m "Initial commit"

# 2. Push lên GitHub
git remote add origin https://github.com/username/pawai.git
git push -u origin main
```

3. Vào [render.com](https://render.com) → **New** → **Web Service**
4. Connect GitHub repo
5. Cấu hình:
   - **Root Directory**: `webapp`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Environment**: Python 3
   - **Plan**: Free (hoặc Starter)

### Download Model Từ Hugging Face Hub (khuyến nghị)

```python
# startup.py – thêm vào đầu app.py
from huggingface_hub import hf_hub_download
import os

def download_model_if_needed():
    model_path = '/tmp/best_model_final.keras'
    if not os.path.exists(model_path):
        hf_hub_download(
            repo_id='your-username/pawai-model',
            filename='best_model_final.keras',
            local_dir='/tmp'
        )
    return model_path
```

### Biến môi trường trên Render

| Key | Value |
|-----|-------|
| `FLASK_DEBUG` | `false` |
| `MODEL_PATH` | `/tmp/best_model_final.keras` |
| `PORT` | (tự động) |

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| GET | `/` | Trang chủ |
| POST | `/predict` | Upload ảnh → JSON kết quả |
| GET | `/health` | Health check |

### Response `/predict`

```json
{
  "label"        : "cat",
  "label_vi"     : "Mèo 🐱",
  "confidence"   : "92.34%",
  "cat_prob"     : 92.3,
  "dog_prob"     : 7.7,
  "color"        : "#4FC3F7",
  "image_url"    : "/uploads/abc123.jpg"
}
```

## Tính Năng Web App

- ✅ Drag & Drop ảnh
- ✅ Camera capture (mobile)
- ✅ Paste từ clipboard (Ctrl+V)
- ✅ Hiển thị probability bars có animation
- ✅ Hiệu ứng confetti khi độ tin cậy > 90%
- ✅ Responsive: Desktop + Tablet + Mobile
- ✅ Health check endpoint cho Render
