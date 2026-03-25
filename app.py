from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import cv2

# ============================
# Flask app setup
# ============================
app = Flask(__name__)

# Path to your trained model
MODEL_PATH = 'apple_disease_model.h5'

# Load the trained model
print(" ** Loading Model **")
model = load_model(MODEL_PATH, compile=False)
print(" ** Model Loaded Successfully **")

# ============================
# CORRECTED CLASS LABELS
# Must match: {'apple_scab': 0, 'black_rot': 1, 'cedar_apple_rust': 2, 'healthy': 3}
# ============================
class_labels = [
    'apple_scab',
    'black_rot',
    'cedar_apple_rust',
    'healthy'
]

# ============================
# Disease Information (English)
# ============================
disease_info = {
    'black_rot': {
        'precautions': ['Prune out dead or infected branches.', 'Avoid overhead irrigation.', 'Remove fallen leaves.'],
        'fertilizers': ['Apply balanced NPK fertilizer (10-10-10).', 'Use organic manure.'],
        'pesticides': ['Use fungicides with Captan or Mancozeb.']
    },
    'cedar_apple_rust': {
        'precautions': ['Remove nearby juniper plants.', 'Prune affected twigs early.', 'Ensure proper airflow.'],
        'fertilizers': ['Apply nitrogen-rich fertilizers.', 'Use compost.'],
        'pesticides': ['Use Myclobutanil-based fungicides.']
    },
    'apple_scab': {
        'precautions': ['Remove infected leaves immediately.', 'Avoid overhead watering.', 'Use resistant varieties.'],
        'fertilizers': ['Use potassium-rich fertilizers.', 'Avoid excessive nitrogen.'],
        'pesticides': ['Apply Sulfur or Mancozeb fungicides.']
    },
    'healthy': {
        'precautions': ['Maintain regular pruning.', 'Ensure tree hygiene.'],
        'fertilizers': ['Apply NPK based on soil tests.'],
        'pesticides': ['No pesticide required.']
    }
}

# ============================
# Translation Data (Hindi & Telugu)
# ============================
translated_info = {
    'hi': {
        'match_label': 'मैच (समानता)',
        'analysis_badge': 'विश्लेषण पूरा हुआ',
        'prob_header': '📊 सभी वर्गों की संभावनाएं:',
        'gradcam_label': '🔍 एआई विश्लेषण मानचित्र (Grad-CAM)',
        'headers': {'prec': 'सावधानियां', 'fert': 'उर्वरक', 'pest': 'कीटनाशक'},
        'data': {
            'black_rot': {
                'disease': 'ब्लैक रॉट (काला सड़न)',
                'precautions': ['संक्रमित टहनियों को काटें।', 'हवा के संचार के लिए छंटाई करें।'],
                'fertilizers': ['संतुलित NPK का उपयोग करें।'],
                'pesticides': ['कैप्टन या मैंकोज़ेब का छिड़काव करें।']
            },
            'cedar_apple_rust': {
                'disease': 'रस्ट (जंग लगना)',
                'precautions': ['पास के जुनिपर पौधों को हटाएं।', 'संक्रमित पत्तियों को जलाएं।'],
                'fertilizers': ['मिट्टी के स्वास्थ्य के लिए खाद डालें।'],
                'pesticides': ['माइक्लोबुटानिल का प्रयोग करें।']
            },
            'apple_scab': {
                'disease': 'स्कैब (पपड़ी रोग)',
                'precautions': ['गिरे हुए पत्तों को साफ करें।', 'ऊपर से पानी न डालें।'],
                'fertilizers': ['पोटेशियम युक्त उर्वरक डालें।'],
                'pesticides': ['सल्फर कवकनाशी का प्रयोग करें।']
            },
            'healthy': {
                'disease': 'स्वस्थ पत्ता',
                'precautions': ['नियमित सफाई बनाए रखें।'],
                'fertilizers': ['नियमित रूप से खाद डालें।'],
                'pesticides': ['कीटनाशक की आवश्यकता नहीं है।']
            }
        }
    },
    'te': {
        'match_label': 'సరిపోలిక',
        'analysis_badge': 'విశ్లేషణ పూర్తయింది',
        'prob_header': '📊 అన్ని తరగతుల సంభావ్యతలు:',
        'gradcam_label': '🔍 AI విశ్లేషణ పటం (Grad-CAM)',
        'headers': {'prec': 'జాగ్రత్తలు', 'fert': 'ఎరువులు', 'pest': 'పురుగుమందులు'},
        'data': {
            'black_rot': {
                'disease': 'బ్లాక్ రాట్ (నలుపు కుళ్లు)',
                'precautions': ['వ్యాధి సోకిన కొమ్మలను తొలగించండి.', 'గాలి వెలుతురు వచ్చేలా చూడండి.'],
                'fertilizers': ['సమతుల్య NPK ఎరువులను వాడండి.'],
                'pesticides': ['క్యాప్టాన్ లేదా మ్యాంకోజెబ్ వాడండి.']
            },
            'cedar_apple_rust': {
                'disease': 'తుప్పు తెగులు (Rust)',
                'precautions': ['జునిపెర్ మొక్కలను తొలగించండి.', 'సోకిన ఆకులను ఏరి వేయండి.'],
                'fertilizers': ['సేంద్రియ ఎరువులను వాడండి.'],
                'pesticides': ['మైక్లోబుటానిల్ వాడండి.']
            },
            'apple_scab': {
                'disease': 'స్కాబ్ (పొలుసు తెగులు)',
                'precautions': ['రాలిన ఆకులను శుభ్రం చేయండి.', 'నీరు నేరుగా పడకుండా చూడండి.'],
                'fertilizers': ['పొటాషియం ఎరువులు వాడండి.'],
                'pesticides': ['గంధకం (Sulfur) వాడండి.']
            },
            'healthy': {
                'disease': 'ఆరోగ్యకరమైన ఆకు',
                'precautions': ['తోటను శుభ్రంగా ఉంచండి.'],
                'fertilizers': ['తగిన ఎరువులు వేయండి.'],
                'pesticides': ['మందులు అవసరం లేదు.']
            }
        }
    }
}

# ============================
# Grad-CAM Function
# ============================
def get_gradcam_image(img_path, model, upload_folder, filename):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img_keras = image.load_img(img_path, target_size=(300, 300))
        x = image.img_to_array(img_keras)
        x = np.expand_dims(x, axis=0) / 255.0

        last_conv_layer_name = None
        for layer in reversed(model.layers):
            try:
                shape = layer.output_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) == 4:
                    last_conv_layer_name = layer.name
                    break
            except Exception:
                continue
        
        if not last_conv_layer_name:
            last_conv_layer_name = 'top_conv'

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            outputs = grad_model(x)
            last_conv_layer_output = outputs[0]
            preds = outputs[1]
            
            if isinstance(preds, (list, tuple)): preds = preds[0]
            if isinstance(last_conv_layer_output, (list, tuple)): last_conv_layer_output = last_conv_layer_output[0]
            
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)

        gradcam_filename = "gradcam_" + filename
        gradcam_path = os.path.join(upload_folder, gradcam_filename)
        cv2.imwrite(gradcam_path, superimposed_img)
        return gradcam_filename
    except Exception as e:
        print("Grad-CAM Error:", str(e))
        return None

# ============================
# Prediction Function
# ============================
def model_predict(img_path, model):
    # UPDATED: target_size is now 300x300 for EfficientNet-B3
    img = image.load_img(img_path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    predicted_label = class_labels[pred_idx]
    confidence = round(float(preds[0][pred_idx]) * 100, 2)
    return predicted_label, confidence, preds[0]

# ============================
# Routes
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    try:
        lang = request.form.get('lang', 'en')
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # 1. Prediction
        predicted_label, confidence, raw_preds = model_predict(file_path, model)
        
        # 1.5 Grad-CAM Generation
        gradcam_filename = get_gradcam_image(file_path, model, upload_folder, secure_filename(f.filename))

        # 2. Localized Content Selection
        if lang in translated_info:
            t_data = translated_info[lang]
            info = t_data['data'].get(predicted_label, {})
            display_disease = info.get('disease', 'Unknown')
            precautions = info.get('precautions', [])
            fertilizers = info.get('fertilizers', [])
            pesticides = info.get('pesticides', [])
            match_word = t_data['match_label']
            analysis_word = t_data['analysis_badge']
            prob_header = t_data['prob_header']
            gradcam_label = t_data.get('gradcam_label', '🔍 AI Analysis Map (Grad-CAM)')
            h = t_data['headers']
        else:
            # English Defaults
            info = disease_info.get(predicted_label, {})
            display_disease = predicted_label.replace('_', ' ').title()
            precautions = info.get('precautions', [])
            fertilizers = info.get('fertilizers', [])
            pesticides = info.get('pesticides', [])
            match_word = "Match"
            analysis_word = "Analysis Complete"
            prob_header = "📊 All Class Probabilities:"
            gradcam_label = "🔍 AI Analysis Map (Grad-CAM)"
            h = {'prec': 'Precautions', 'fert': 'Fertilizers', 'pest': 'Pesticides'}

        gradcam_html = ""
        if gradcam_filename:
            gradcam_html = f'''
            <div class="text-center mb-4">
                <img src="/static/uploads/{gradcam_filename}" class="img-fluid rounded border border-2 shadow-sm" style="max-height: 250px;" alt="Grad-CAM Heatmap">
                <p class="text-muted small mt-2 fw-bold">{gradcam_label}</p>
            </div>
            '''

        # 3. Localized Probability List Mapping
        prob_list_html = ""
        for i, p in enumerate(raw_preds):
            label_key = class_labels[i]
            conf_val = round(float(p) * 100, 2)
            if lang in translated_info:
                name = translated_info[lang]['data'].get(label_key, {}).get('disease', label_key)
            else:
                name = label_key.replace('_', ' ').title()
            
            prob_list_html += f"""
                <li class='d-flex justify-content-between border-bottom py-1'>
                    <span>{name}</span> <strong>{conf_val}%</strong>
                </li>"""

        # 4. Final HTML Result Card
        return f"""
        <div class="card result-card shadow-lg mt-4 animate__animated animate__fadeInUp">
            <div class="result-header text-center">
                <span class="badge rounded-pill bg-success mb-2 px-3 py-2 text-uppercase">{analysis_word}</span>
                <h2 class="fw-bold text-success mb-0">🍎 {display_disease}</h2>
                <p class="text-center text-muted mb-0"><b>{confidence}%</b> {match_word}</p>
            </div>
            <div class="card-body p-4">
                {gradcam_html}
                <div class="progress mb-4" style="height: 12px; border-radius: 10px;">
                    <div class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: {confidence}%"></div>
                </div>
                <div class="row">
                    <div class="col-12 mb-4">
                        <div class="p-3 rounded-4" style="background-color: rgba(59, 130, 246, 0.1); border-left: 5px solid #3b82f6;">
                            <h5 class="fw-bold text-primary mb-3">📋 {h['prec']}</h5>
                            <ul class="info-list mb-0">{"".join(f"<li>{p}</li>" for p in precautions)}</ul>
                        </div>
                    </div>
                    <div class="col-md-6 mb-3">
                        <div class="p-3 h-100 rounded-4" style="background-color: rgba(34, 197, 94, 0.1); border-left: 5px solid #22c55e;">
                            <h5 class="fw-bold text-success mb-3">🌿 {h['fert']}</h5>
                            <ul class="info-list mb-0">{"".join(f"<li>{f}</li>" for f in fertilizers)}</ul>
                        </div>
                    </div>
                    <div class="col-md-6 mb-3">
                        <div class="p-3 h-100 rounded-4" style="background-color: rgba(239, 68, 68, 0.1); border-left: 5px solid #ef4444;">
                            <h5 class="fw-bold text-danger mb-3">🧪 {h['pest']}</h5>
                            <ul class="info-list mb-0">{"".join(f"<li>{p}</li>" for p in pesticides)}</ul>
                        </div>
                    </div>
                </div>
                <hr>
                <div class="p-3 rounded-4" style="background-color: rgba(128, 128, 128, 0.1);">
                    <h5 class="fw-bold mb-3">{prob_header}</h5>
                    <ul class="list-unstyled mb-0">{prob_list_html}</ul>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div class='alert alert-danger'>Error: {str(e)}</div>", 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
