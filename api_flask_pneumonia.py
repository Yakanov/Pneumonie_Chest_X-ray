
from flask import Flask, request, jsonify, render_template_string
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import joblib
import numpy as np
import json
from pathlib import Path
import tempfile
import base64
import io
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

app = Flask(__name__)

# Résolution robuste du dossier d'artefacts, peu importe le dossier de lancement.
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR_CANDIDATES = [
    SCRIPT_DIR / "saved_models",
    SCRIPT_DIR / "Projet Science des donnees" / "saved_models"
]
ARTIFACT_DIR = next((p for p in ARTIFACT_DIR_CANDIDATES if p.exists()), None)
if ARTIFACT_DIR is None:
    raise FileNotFoundError(
        "Impossible de trouver le dossier saved_models. "
        "Chemins testés: " + ", ".join(str(p) for p in ARTIFACT_DIR_CANDIDATES)
    )

# Chargement des artefacts
model_eff = keras.models.load_model(ARTIFACT_DIR / "efficientnet_best.keras")
model_res = keras.models.load_model(ARTIFACT_DIR / "resnet50_best.keras")
bundle = joblib.load(ARTIFACT_DIR / "gradient_boosting_bundle.joblib")

gb_model = bundle["gb_model"]
scaler = bundle["scaler"]
pca = bundle["pca"]
IMG_SIZE = (224, 224)
IMG_SIZE_ML = tuple(bundle["img_size_ml"])

with open(ARTIFACT_DIR / "ensemble_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

classes = manifest["classes"]
threshold = float(manifest["threshold"])


HOME_HTML = """
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Détecteur de Pneumonie - XAI avec GRAD-CAM</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 1rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        
        .card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8ebff;
        }
        
        .upload-area p { color: #666; margin-bottom: 1rem; }
        .upload-area input { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .btn-secondary {
            background: #eef2ff;
            color: #374151;
            border: 1px solid #c7d2fe;
            margin-top: 1rem;
        }
        .btn-secondary:hover {
            box-shadow: none;
            background: #e0e7ff;
        }
        
        .results {
            display: none;
            margin-top: 2rem;
        }
        .results.active { display: block; }

        .section-hidden { display: none; }
        
        .result-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        @media (max-width: 768px) {
            .result-row { grid-template-columns: 1fr; }
        }
        
        .result-box {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1rem;
            background: #fafafa;
        }
        .result-box h3 { margin-bottom: 0.5rem; color: #333; }
        .result-box img { width: 100%; height: auto; border-radius: 8px; }

        .batch-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1.25rem;
        }
        .batch-card {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            background: #ffffff;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }
        .batch-card-header {
            padding: 0.9rem 1rem;
            border-bottom: 1px solid #eef2f7;
            background: #f8fafc;
        }
        .batch-card-header strong {
            display: block;
            color: #111827;
            word-break: break-word;
        }
        .batch-card-header span {
            color: #4b5563;
            font-size: 0.9rem;
        }
        .batch-card-body {
            padding: 1rem;
        }
        .batch-card-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-top: 0.9rem;
        }
        .batch-card-grid img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
            background: #f3f4f6;
        }
        .mini-status {
            display: inline-block;
            margin-top: 0.6rem;
            padding: 0.3rem 0.6rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .mini-status.ok {
            color: #0f5132;
            background: #d1e7dd;
        }
        .mini-status.fallback {
            color: #664d03;
            background: #fff3cd;
        }
        
        .predictions {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
        }
        .predictions h3 { margin-bottom: 1rem; }
        
        .pred-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #eee;
        }
        .pred-item:last-child { border-bottom: none; }
        .pred-label { font-weight: 600; color: #333; }
        .pred-value {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        .prediction-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .prediction-result h2 { font-size: 2rem; margin-bottom: 0.5rem; }

        .xai-status {
            display: inline-block;
            margin-bottom: 1rem;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.2px;
            border: 1px solid transparent;
        }
        .xai-status.ok {
            color: #0f5132;
            background: #d1e7dd;
            border-color: #badbcc;
        }
        .xai-status.fallback {
            color: #664d03;
            background: #fff3cd;
            border-color: #ffecb5;
        }
        
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 2rem auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #c33;
            display: none;
            margin-bottom: 1rem;
        }
        .error.active { display: block; }
        .error.warning {
            background: #fff3cd;
            color: #856404;
            border-left-color: #f39c12;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Détecteur de Pneumonie</h1>
            <p>Upload une radiographie thoracique pour une prédiction avec explications (GRAD-CAM)</p>
        </div>
        
        <div class="card">
            <div class="error" id="errorMsg"></div>
            
            <div class="upload-area" id="uploadArea">
                <p>📁 Drag & drop une ou plusieurs radios ici ou cliquez pour sélectionner</p>
                <button class="btn">Choisir une image</button>
                <input type="file" id="fileInput" accept="image/*" multiple>
            </div>
            
            <div id="loading" class="loader" style="display: none;"></div>
            
            <div class="results" id="results">
                <div class="prediction-result" id="predictionResult">
                    <h2 id="predLabel">Détection...</h2>
                    <p id="predConfidence">Confiance: --</p>
                </div>
                <div id="xaiStatus" class="xai-status">XAI: en attente</div>
                <button id="resetBtn" class="btn btn-secondary" type="button">Recommencer</button>
                
                <div id="singleResultsContent">
                <div class="predictions">
                    <h3>📊 Probabilités par modèle</h3>
                    <div id="predsContainer"></div>
                </div>
                
                <div class="result-row">
                    <div class="result-box">
                        <h3>🖼️ Image originale</h3>
                        <img id="originalImg" src="" alt="Original">
                    </div>
                    <div class="result-box">
                        <h3>🔍 GRAD-CAM Moyenne</h3>
                        <img id="gradcamImg" src="" alt="GRAD-CAM">
                    </div>
                </div>
                </div>

                <div id="batchResults" class="section-hidden">
                    <div class="predictions">
                        <h3>🧪 Résultats du lot</h3>
                        <div id="batchSummary"></div>
                        <div id="batchGrid" class="batch-grid"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const errorMsg = document.getElementById('errorMsg');
        const xaiStatus = document.getElementById('xaiStatus');
        const resetBtn = document.getElementById('resetBtn');
        const singleResultsContent = document.getElementById('singleResultsContent');
        const batchResults = document.getElementById('batchResults');
        const batchSummary = document.getElementById('batchSummary');
        const batchGrid = document.getElementById('batchGrid');
        
        // Drag & drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
            uploadFile();
        });
        
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', uploadFile);
        resetBtn.addEventListener('click', resetResults);
        
        async function requestPrediction(file) {
            const formData = new FormData();
            formData.append('file', file);

            const resp = await fetch('/predict-upload-xai', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();

            if (!resp.ok) {
                // Retourner un objet erreur (ne pas lever d'exception) pour que le mode batch continue
                return { _isError: true, error: data.error || 'Erreur serveur', error_type: data.error_type || 'server_error' };
            }

            return data;
        }

        async function uploadFile() {
            const files = Array.from(fileInput.files || []);
            if (!files.length) return;
            
            errorMsg.classList.remove('active', 'warning');
            errorMsg.innerHTML = '';
            loading.style.display = 'block';
            results.classList.remove('active');
            batchGrid.innerHTML = '';
            batchSummary.innerHTML = '';
            
            try {
                if (files.length === 1) {
                    const data = await requestPrediction(files[0]);
                    if (data._isError) {
                        if (data.error_type === 'not_chest_xray') {
                            errorMsg.innerHTML = '⚠️&nbsp;' + data.error
                                + '<br><small style="font-weight:normal">Merci de sélectionner une véritable radiographie du thorax.</small>';
                            errorMsg.classList.add('active', 'warning');
                        } else {
                            throw new Error(data.error);
                        }
                    } else {
                        displayResults(data, files[0]);
                    }
                } else {
                    const batchItems = [];
                    for (const file of files) {
                        const data = await requestPrediction(file);
                        batchItems.push({ file, data });
                    }
                    displayBatchResults(batchItems);
                }
            } catch (err) {
                errorMsg.textContent = '❌ Erreur: ' + err.message;
                errorMsg.classList.add('active');
            } finally {
                loading.style.display = 'none';
            }
        }

        function resetResults() {
            fileInput.value = '';
            loading.style.display = 'none';
            results.classList.remove('active');
            errorMsg.classList.remove('active', 'warning');
            errorMsg.innerHTML = '';
            document.getElementById('originalImg').src = '';
            document.getElementById('gradcamImg').src = '';
            document.getElementById('predLabel').textContent = 'Détection...';
            document.getElementById('predConfidence').textContent = 'Confiance: --';
            document.getElementById('predsContainer').innerHTML = '';
            document.getElementById('predictionResult').style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            xaiStatus.className = 'xai-status';
            xaiStatus.textContent = 'XAI: en attente';
            singleResultsContent.classList.remove('section-hidden');
            batchResults.classList.add('section-hidden');
            batchGrid.innerHTML = '';
            batchSummary.innerHTML = '';
        }

        function displayBatchResults(items) {
            const validItems = items.filter(item => !item.data._isError);
            const invalidItems = items.filter(item => item.data._isError);
            const validGradcams = validItems.filter((item) => (item.data.gradcam_status || 'fallback') === 'ok').length;
            const pneumoniaCount = validItems.filter((item) => item.data.prediction_label === 'PNEUMONIA').length;
            const normalCount = validItems.length - pneumoniaCount;

            document.getElementById('predictionResult').style.background = 'linear-gradient(135deg, #334155 0%, #0f172a 100%)';
            document.getElementById('predLabel').textContent = `Lot analysé: ${items.length} image(s)`;
            document.getElementById('predConfidence').textContent =
                `NORMAL: ${normalCount} | PNEUMONIA: ${pneumoniaCount}`
                + (invalidItems.length > 0 ? ` | ⚠️ Invalides: ${invalidItems.length}` : '');

            xaiStatus.className = (validGradcams === validItems.length && invalidItems.length === 0) ? 'xai-status ok' : 'xai-status fallback';
            xaiStatus.textContent = `XAI lot: ${validGradcams}/${validItems.length} GRAD-CAM valides`;

            document.getElementById('predsContainer').innerHTML = '';
            document.getElementById('originalImg').src = '';
            document.getElementById('gradcamImg').src = '';

            singleResultsContent.classList.add('section-hidden');
            batchResults.classList.remove('section-hidden');

            const invalidBadge = invalidItems.length > 0
                ? `<div class="pred-item"><span class="pred-label">⚠️ Images invalides</span><span class="pred-value">${invalidItems.length}</span></div>`
                : '';
            batchSummary.innerHTML = `<div class="pred-item"><span class="pred-label">Images traitées</span><span class="pred-value">${items.length}</span></div><div class="pred-item"><span class="pred-label">PNEUMONIA</span><span class="pred-value">${pneumoniaCount}</span></div><div class="pred-item"><span class="pred-label">NORMAL</span><span class="pred-value">${normalCount}</span></div>${invalidBadge}`;
            batchGrid.innerHTML = '';

            for (const item of items) {
                const card = document.createElement('div');
                card.className = 'batch-card';
                if (item.data._isError) {
                    const originalUrl = URL.createObjectURL(item.file);
                    card.innerHTML = `<div class="batch-card-header" style="background:linear-gradient(135deg,#dc2626 0%,#991b1b 100%);"><strong>${item.file.name}</strong><span>⚠️ Image invalide</span></div><div class="batch-card-body"><div class="mini-status fallback">Non reconnue</div><div style="margin-top:0.75rem;"><img src="${originalUrl}" alt="Image" style="max-width:100%;border-radius:6px;opacity:0.6;"><p style="font-size:0.8rem;color:#555;margin-top:0.4rem;">${item.data.error}</p></div></div>`;
                } else {
                    const status = (item.data.gradcam_status || 'fallback').toLowerCase();
                    const confidence = (Math.max(...Object.values(item.data.probabilities)) * 100).toFixed(1);
                    const originalUrl = URL.createObjectURL(item.file);
                    const gradcamUrl = item.data.gradcam_base64 ? `data:image/png;base64,${item.data.gradcam_base64}` : '';
                    const statusLabel = status === 'ok' ? 'GRAD-CAM valide' : 'Fallback';
                    card.innerHTML = `<div class="batch-card-header"><strong>${item.file.name}</strong><span>${item.data.prediction_label} | ${confidence}%</span></div><div class="batch-card-body"><div class="mini-status ${status}">${statusLabel}</div><div class="batch-card-grid"><div><img src="${originalUrl}" alt="Image originale"><span>Originale</span></div><div><img src="${gradcamUrl}" alt="GRAD-CAM"><span>GRAD-CAM</span></div></div></div>`;
                }
                batchGrid.appendChild(card);
            }

            results.classList.add('active');
        }
        
        function displayResults(data, file) {
            singleResultsContent.classList.remove('section-hidden');
            batchResults.classList.add('section-hidden');

            // Afficher l'image originale
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('originalImg').src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            // Afficher l'image GRAD-CAM
            if (data.gradcam_base64) {
                document.getElementById('gradcamImg').src = 'data:image/png;base64,' + data.gradcam_base64;
            } else {
                document.getElementById('gradcamImg').src = '';
            }

            // Badge de statut XAI
            const status = (data.gradcam_status || 'fallback').toLowerCase();
            if (status === 'ok') {
                xaiStatus.className = 'xai-status ok';
                xaiStatus.textContent = 'XAI: GRAD-CAM valide';
            } else {
                xaiStatus.className = 'xai-status fallback';
                xaiStatus.textContent = 'XAI: mode fallback (carte indisponible)';
            }
            
            // Afficher le résultat de prédiction
            const label = data.prediction_label;
            const confidence = Math.max(...Object.values(data.probabilities)) * 100;
            const predColor = label === 'PNEUMONIA' ? '#e74c3c' : '#27ae60';
            
            document.getElementById('predictionResult').style.background = `linear-gradient(135deg, ${predColor} 0%, ${predColor}dd 100%)`;
            document.getElementById('predLabel').textContent = label === 'PNEUMONIA' ? '⚠️ PNEUMONIE' : '✅ NORMAL';
            document.getElementById('predConfidence').textContent = `Confiance: ${confidence.toFixed(1)}%`;
            
            // Afficher les probas
            const predsContainer = document.getElementById('predsContainer');
            predsContainer.innerHTML = '';
            for (const [model, prob] of Object.entries(data.probabilities)) {
                const pct = (prob * 100).toFixed(1);
                const bar = document.createElement('div');
                bar.className = 'pred-item';
                bar.innerHTML = `<span class="pred-label">${model}</span><span class="pred-value">${pct}%</span>`;
                predsContainer.appendChild(bar);
            }
            
            results.classList.add('active');
        }
    </script>
</body>
</html>
"""


def preprocess_cnn(path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_gb(path):
    img = load_img(path, target_size=IMG_SIZE_ML)
    arr = img_to_array(img).astype("float32") / 255.0
    arr = arr.reshape(1, -1)
    arr = scaler.transform(arr)
    return pca.transform(arr)


def is_chest_xray(pil_image):
    """
    Heuristique : vérifie si l'image ressemble à une radiographie thoracique.
    Les radios sont quasi-monochromes (R≈G≈B pour chaque pixel) et
    présentent une dynamique de contraste significative.
    Retourne (bool, message_raison).
    """
    img_rgb = pil_image.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)

    # --- Critère 1 : fraction de pixels colorés (saturation > 0.25) ---
    sat = np.where(max_ch > 10, (max_ch - min_ch) / (max_ch + 1e-6), 0.0)
    frac_colorful = float(np.mean(sat > 0.25))
    if frac_colorful > 0.10:          # seuil strict : 10% suffit pour rejeter
        return False, (
            f"L'image semble trop colorée ({frac_colorful * 100:.0f}% de pixels colorés). "
            "Une radiographie thoracique est quasi-monochrome (niveaux de gris)."
        )

    # --- Critère 2 : écart moyen entre canaux (R≈G≈B sur une vraie radio) ---
    mean_channel_diff = float(
        np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    ) / 255.0
    if mean_channel_diff > 0.05:
        return False, (
            f"Les canaux couleur sont trop différents (écart moyen : {mean_channel_diff * 100:.1f}%). "
            "Une radiographie thoracique est quasi-monochrome."
        )

    # --- Critère 3 : contraste minimal ---
    gray = np.mean(arr, axis=2)
    if float(np.std(gray)) < 10.0:
        return False, "L'image est trop uniforme pour être une radiographie thoracique."

    return True, "ok"


def generate_gradcam(model, input_array, layer_name=None):
    """Génère une heatmap GRAD-CAM pour les modèles Sequential (backbone + tête)."""
    try:
        img_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

        # Dans ce projet, le backbone CNN est la première couche du Sequential.
        backbone = model.layers[0]

        with tf.GradientTape() as tape:
            conv_outputs = backbone(img_tensor, training=False)
            tape.watch(conv_outputs)

            x = conv_outputs
            for layer in model.layers[1:]:
                x = layer(x, training=False)

            preds = x
            # Sortie sigmoid binaire -> indice 0
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        conv_map = conv_outputs[0]
        weights = pooled_grads[0]
        heatmap = tf.reduce_sum(conv_map * weights, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if float(max_val) == 0.0:
            return None

        heatmap = heatmap / max_val
        return heatmap.numpy().astype("float32")
    except Exception as e:
        print(f"Erreur GRAD-CAM: {e}")
        return None


def heatmap_to_png_base64(heatmap, original_img_path):
    """Convertit une heatmap GRAD-CAM en image PNG base64."""
    try:
        img = load_img(original_img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img).astype("float32") / 255.0

        def apply_strict_lung_mask(heatmap_in, image_in):
            """Masque pulmonaire strict + seuillage fort des activations."""
            h, w = heatmap_in.shape
            gray = np.mean(image_in, axis=2)

            # Zone corps (évite les bords externes très clairs/noirs).
            body_mask = gray < 0.95
            ys, xs = np.where(body_mask)
            if len(ys) < 20 or len(xs) < 20:
                # Fallback minimal si extraction corps impossible.
                body_mask = np.ones((h, w), dtype=bool)
                y0, y1, x0, x1 = 0, h - 1, 0, w - 1
            else:
                y0, y1 = int(np.min(ys)), int(np.max(ys))
                x0, x1 = int(np.min(xs)), int(np.max(xs))

            # Grille normalisée dans la boîte thoracique.
            yy, xx = np.mgrid[0:h, 0:w]
            bw = max(x1 - x0 + 1, 1)
            bh = max(y1 - y0 + 1, 1)
            x_norm = (xx - x0) / bw
            y_norm = (yy - y0) / bh

            # Deux ellipses anatomiques (poumon gauche et droit), plus serrées.
            left_lung = (((x_norm - 0.33) / 0.17) ** 2 + ((y_norm - 0.50) / 0.32) ** 2) <= 1.0
            right_lung = (((x_norm - 0.67) / 0.17) ** 2 + ((y_norm - 0.50) / 0.32) ** 2) <= 1.0

            # Coupes anatomiques pour exclure épaules, abdomen et bords latéraux.
            vertical_band = (y_norm >= 0.14) & (y_norm <= 0.86)
            lateral_band = (x_norm >= 0.18) & (x_norm <= 0.82)
            mediastinum_cut = (x_norm <= 0.45) | (x_norm >= 0.55)

            lung_roi = (left_lung | right_lung) & vertical_band & lateral_band & mediastinum_cut & body_mask
            if np.mean(lung_roi) < 0.08:
                # Fallback si masque trop restrictif sur image atypique.
                lung_roi = (left_lung | right_lung) & vertical_band & lateral_band

            out = heatmap_in * lung_roi.astype(np.float32)
            # Seuil clinique fort: ne garder que les activations élevées dans les poumons.
            roi_vals = out[lung_roi]
            if roi_vals.size > 20:
                thr = np.percentile(roi_vals, 85)
                out = np.where(out >= thr, out, 0.0)

            max_val = np.max(out)
            if max_val > 1e-10:
                out = out / max_val
            return out
        
        # Redimensionner la heatmap à la taille de l'image
        if heatmap.shape != (IMG_SIZE[0], IMG_SIZE[1]):
            heatmap = tf.image.resize(
                heatmap[..., np.newaxis],
                (IMG_SIZE[0], IMG_SIZE[1]),
                method="bilinear"
            ).numpy()[..., 0]

        # Masque pulmonaire strict : zéro en dehors des champs pulmonaires.
        heatmap = apply_strict_lung_mask(heatmap, img_array)
        
        # Normaliser
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        # Appliquer le colormap
        heatmap_colored = cm.jet(heatmap)
        
        # Superposer sur l'image originale
        overlay = 0.4 * heatmap_colored[:, :, :3] + 0.6 * img_array
        overlay = np.clip(overlay, 0, 1)
        
        # Convertir en PNG base64
        fig = plt.figure(figsize=(6, 6), dpi=60)
        ax = fig.add_subplot(111)
        ax.imshow(overlay)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Erreur heatmap_to_png_base64: {e}")
        return None


def predict_from_path(image_path):
    x_cnn = preprocess_cnn(image_path)
    p_eff = float(model_eff.predict(x_cnn, verbose=0).ravel()[0])
    p_res = float(model_res.predict(x_cnn, verbose=0).ravel()[0])

    x_ml = preprocess_gb(image_path)
    p_gb = float(gb_model.predict_proba(x_ml)[:, 1][0])

    votes = int(p_eff >= threshold) + int(p_res >= threshold) + int(p_gb >= threshold)
    pred_idx = int(votes >= 2)

    return {
        "prediction_index": pred_idx,
        "prediction_label": classes[pred_idx],
        "probabilities": {
            "efficientnet": p_eff,
            "resnet50": p_res,
            "gradient_boosting": p_gb,
            "ensemble_soft": (p_eff + p_res + p_gb) / 3.0
        },
        "votes": votes
    }


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "pneumonia-api"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image_path")
    if not image_path:
        return jsonify({"error": "Champ 'image_path' manquant."}), 400

    try:
        result = predict_from_path(image_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Prediction impossible: {exc}"}), 500


@app.route("/predict-upload", methods=["POST"])
def predict_upload():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé. Champ attendu: 'file'."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Nom de fichier vide."}), 400

    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return jsonify({"error": "Format non supporte. Utilise png/jpg/jpeg/bmp/webp."}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        result = predict_from_path(temp_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Prediction impossible: {exc}"}), 500
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.route("/predict-upload-xai", methods=["POST"])
def predict_upload_xai():
    """Endpoint avec GRAD-CAM pour l'explainability."""
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé. Champ attendu: 'file'."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Nom de fichier vide."}), 400

    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return jsonify({"error": "Format non supporte. Utilise png/jpg/jpeg/bmp/webp."}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Validation : vérifier que l'image ressemble à une radio thoracique
        with Image.open(temp_path) as pil_img:
            valid_xray, xray_reason = is_chest_xray(pil_img)
        if not valid_xray:
            return jsonify({
                "error": xray_reason,
                "error_type": "not_chest_xray"
            }), 422

        # Prédictions
        x_cnn = preprocess_cnn(temp_path)
        p_eff = float(model_eff.predict(x_cnn, verbose=0).ravel()[0])
        p_res = float(model_res.predict(x_cnn, verbose=0).ravel()[0])

        x_ml = preprocess_gb(temp_path)
        p_gb = float(gb_model.predict_proba(x_ml)[:, 1][0])

        votes = int(p_eff >= threshold) + int(p_res >= threshold) + int(p_gb >= threshold)
        pred_idx = int(votes >= 2)

        # Générer les GRAD-CAM
        heatmap_eff = generate_gradcam(model_eff, x_cnn)
        heatmap_res = generate_gradcam(model_res, x_cnn)
        
        # Créer une heatmap moyennée
        if heatmap_eff is not None and heatmap_res is not None:
            heatmap_avg = (heatmap_eff + heatmap_res) / 2
            gradcam_b64 = heatmap_to_png_base64(heatmap_avg, temp_path)
        elif heatmap_eff is not None:
            gradcam_b64 = heatmap_to_png_base64(heatmap_eff, temp_path)
        elif heatmap_res is not None:
            gradcam_b64 = heatmap_to_png_base64(heatmap_res, temp_path)
        else:
            gradcam_b64 = None
        
        # Fallback: si GRAD-CAM échoue, retourner l'image originale
        if not gradcam_b64:
            try:
                img = Image.open(temp_path)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                gradcam_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            except:
                gradcam_b64 = None

        result = {
            "prediction_index": pred_idx,
            "prediction_label": classes[pred_idx],
            "probabilities": {
                "efficientnet": p_eff,
                "resnet50": p_res,
                "gradient_boosting": p_gb,
                "ensemble_soft": (p_eff + p_res + p_gb) / 3.0
            },
            "votes": votes,
            "gradcam_base64": gradcam_b64,
            "gradcam_status": "fallback" if (heatmap_eff is None and heatmap_res is None) else "ok"
        }
        
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Prediction impossible: {exc}"}), 500
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
