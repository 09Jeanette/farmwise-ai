from transformers import pipeline
from PIL import Image

# Treatment tips mapped to disease labels
TREATMENT_TIPS = {
    "Apple___Apple_scab": "Remove fallen leaves. Apply fungicide early in the season.",
    "Apple___Black_rot": "Prune infected branches. Apply copper-based fungicide.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees if possible. Apply fungicide at pink bud stage.",
    "Apple___healthy": "Your plant looks healthy! Keep up good watering and fertilising practices.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops. Apply foliar fungicide.",
    "Corn_(maize)___Common_rust_": "Plant resistant varieties. Apply fungicide if severe.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids. Apply fungicide at tasseling.",
    "Corn_(maize)___healthy": "Your plant looks healthy! Ensure adequate nitrogen fertilisation.",
    "Grape___Black_rot": "Remove mummified berries. Apply fungicide from bud break.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood. No chemical cure — focus on prevention.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve air circulation. Apply copper fungicide.",
    "Grape___healthy": "Your plant looks healthy! Monitor regularly during wet seasons.",
    "Tomato___Bacterial_spot": "Avoid overhead irrigation. Apply copper-based bactericide.",
    "Tomato___Early_blight": "Remove lower infected leaves. Apply chlorothalonil fungicide.",
    "Tomato___Late_blight": "Remove and destroy infected plants. Apply copper fungicide immediately.",
    "Tomato___Leaf_Mold": "Improve ventilation in greenhouse. Apply fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Apply fungicide every 7–10 days.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap or neem oil spray.",
    "Tomato___Target_Spot": "Improve air circulation. Apply fungicide at first sign.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly population. Remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Disinfect tools. Control aphids.",
    "Tomato___healthy": "Your plant looks healthy! Keep monitoring for pests.",
    "Potato___Early_blight": "Remove infected leaves. Apply mancozeb or chlorothalonil.",
    "Potato___Late_blight": "Apply fungicide immediately. Destroy severely infected plants.",
    "Potato___healthy": "Your plant looks healthy! Ensure good drainage.",
    "Pepper,_bell___Bacterial_spot": "Avoid overhead watering. Apply copper bactericide.",
    "Pepper,_bell___healthy": "Your plant looks healthy! Ensure consistent watering.",
}

DEFAULT_TIP = "Consult your local agricultural extension officer for treatment advice."

# Load model once when the module is imported
print("Loading plant disease model...")
classifier = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)
print("Model loaded successfully.")


def predict_disease(img: Image.Image) -> dict:
    results = classifier(img)
    top = results[0]
    confidence = round(float(top["score"]), 3)
    label = top["label"]

    if confidence < 0.5:
        return {
            "disease": "Unknown — confidence too low",
            "confidence": confidence,
            "treatment_tip": "Please take a clearer photo or consult an agricultural expert.",
            "status": "low_confidence"
        }

    tip = TREATMENT_TIPS.get(label, DEFAULT_TIP)
    is_healthy = "healthy" in label.lower()

    return {
        "disease": label.replace("_", " ").replace("  ", " "),
        "confidence": confidence,
        "treatment_tip": tip,
        "status": "healthy" if is_healthy else "disease_detected"
    }
