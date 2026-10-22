/* ============================================================
   PawAI – main.js
   Xử lý upload ảnh, drag-drop, camera, gọi API /predict
   ============================================================ */

(function () {
  'use strict';

  // ── DOM Elements ──────────────────────────────────────────
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const cameraInput = document.getElementById('camera-input');
  const resultArea  = document.getElementById('result-area');
  const previewImg  = document.getElementById('preview-img');
  const resetBtn    = document.getElementById('reset-btn');
  const loadingEl   = document.getElementById('loading');
  const predictionEl= document.getElementById('prediction');
  const resultEmoji = document.getElementById('result-emoji');
  const resultLabel = document.getElementById('result-label');
  const resultConf  = document.getElementById('result-conf');
  const barCat      = document.getElementById('bar-cat');
  const barDog      = document.getElementById('bar-dog');
  const valCat      = document.getElementById('val-cat');
  const valDog      = document.getElementById('val-dog');
  const errorToast  = document.getElementById('error-toast');

  // ── Drag & Drop ───────────────────────────────────────────
  ['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault(); dropZone.classList.add('drag-over');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'))
  );
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  // ── File inputs ───────────────────────────────────────────
  fileInput.addEventListener('change',   e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
  cameraInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });
  resetBtn.addEventListener('click', resetUI);

  // ── Core: handle file ─────────────────────────────────────
  function handleFile(file) {
    const allowed = ['image/jpeg','image/png','image/gif',
                     'image/bmp','image/webp'];
    if (!allowed.includes(file.type)) {
      showError('Định dạng file không hỗ trợ. Vui lòng chọn ảnh JPG, PNG, WEBP...');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError('File quá lớn. Tối đa 10 MB.');
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = evt => {
      previewImg.src = evt.target.result;
      showResultArea();
      uploadAndPredict(file);
    };
    reader.readAsDataURL(file);
  }

  // ── Upload & Predict ──────────────────────────────────────
  async function uploadAndPredict(file) {
    showLoading();
    hideError();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Lỗi server');
      }

      showPrediction(data);
    } catch (err) {
      showError('Lỗi: ' + (err.message || 'Không thể kết nối server.'));
      hideLoading();
    }
  }

  // ── UI State Functions ────────────────────────────────────
  function showResultArea() {
    resultArea.hidden = false;
    dropZone.hidden   = true;
    setTimeout(() => resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
  }

  function showLoading() {
    loadingEl.hidden    = false;
    predictionEl.hidden = true;
  }

  function hideLoading() {
    loadingEl.hidden = true;
  }

  function showPrediction(data) {
    hideLoading();

    const isCat = data.label === 'cat';
    resultEmoji.textContent = isCat ? '🐱' : '🐶';
    resultLabel.textContent = data.label_vi;
    resultLabel.style.color = data.color;
    resultConf.textContent  = `Độ tin cậy: ${data.confidence}`;

    // Animate probability bars
    barCat.style.width = '0%';
    barDog.style.width = '0%';
    valCat.textContent = '';
    valDog.textContent = '';

    requestAnimationFrame(() => {
      setTimeout(() => {
        barCat.style.width = data.cat_prob + '%';
        barDog.style.width = data.dog_prob + '%';
        valCat.textContent = data.cat_prob + '%';
        valDog.textContent = data.dog_prob + '%';
      }, 50);
    });

    predictionEl.hidden = false;

    // Confetti effect for high confidence
    if (parseFloat(data.confidence) > 0.9) {
      triggerConfetti(data.color);
    }
  }

  function resetUI() {
    resultArea.hidden = true;
    dropZone.hidden   = false;
    fileInput.value   = '';
    cameraInput.value = '';
    previewImg.src    = '';
    loadingEl.hidden    = false;
    predictionEl.hidden = true;
    hideError();
    dropZone.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function showError(msg) {
    errorToast.textContent = msg;
    errorToast.hidden      = false;
  }

  function hideError() {
    errorToast.hidden = true;
  }

  // ── Confetti Effect (lightweight) ────────────────────────
  function triggerConfetti(color) {
    const count = 40;
    for (let i = 0; i < count; i++) {
      createParticle(color);
    }
  }

  function createParticle(color) {
    const el = document.createElement('div');
    el.style.cssText = `
      position: fixed;
      top: ${Math.random() * 60}vh;
      left: ${Math.random() * 100}vw;
      width: ${Math.random() * 10 + 6}px;
      height: ${Math.random() * 10 + 6}px;
      background: ${color};
      border-radius: ${Math.random() > 0.5 ? '50%' : '2px'};
      pointer-events: none;
      z-index: 9999;
      opacity: 0.9;
      transform: translateY(0) rotate(0deg);
      transition: transform ${Math.random() * 1.5 + 1}s ease,
                  opacity   ${Math.random() * 1.5 + 1}s ease;
    `;
    document.body.appendChild(el);
    requestAnimationFrame(() => {
      el.style.transform = `translateY(${Math.random() * 200 + 100}px) rotate(${Math.random() * 360}deg)`;
      el.style.opacity = '0';
    });
    setTimeout(() => el.remove(), 2500);
  }

  // ── Paste from clipboard ─────────────────────────────────
  document.addEventListener('paste', e => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        handleFile(item.getAsFile());
        break;
      }
    }
  });

})();
