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

# Lazy loading for model to reduce memory usage
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        print("Loading plant disease model...")
        classifier = pipeline(
            "image-classification",
            model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
            device_map="auto",  # Use GPU if available, otherwise CPU
            torch_dtype="auto",  # Use automatic dtype for memory efficiency
            model_kwargs={"low_cpu_mem_usage": True}
        )
        print("Model loaded successfully.")
    return classifier


def unload_classifier():
    """Unload the classifier to free memory"""
    global classifier
    if classifier is not None:
        del classifier
        classifier = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Disease model unloaded to free memory.")


def predict_disease(img: Image.Image) -> dict:
    classifier = get_classifier()
    results = classifier(img)
    top = results[0]
    confidence = round(float(top["score"]), 3)
    label = top["label"]

    # Debug: Print the actual label returned by the model
    print(f"Model returned label: '{label}'")
    # Show first 5
    print(
        f"Available labels in TREATMENT_TIPS: {list(TREATMENT_TIPS.keys())[:5]}...")

    if confidence < 0.5:
        return {
            "disease": "Unknown — confidence too low",
            "confidence": confidence,
            "treatment_tip": "Please take a clearer photo or consult an agricultural expert.",
            "status": "low_confidence"
        }

    # Map model output to dictionary keys
    label_mapping = {
        "Corn (Maize) with Cercospora and Gray Leaf Spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn (Maize) with Common Rust": "Corn_(maize)___Common_rust_",
        "Corn (Maize) with Northern Leaf Blight": "Corn_(maize)___Northern_Leaf_Blight",
        "Corn (Maize) Healthy": "Corn_(maize)___healthy",
        "Apple with Apple Scab": "Apple___Apple_scab",
        "Apple with Black Rot": "Apple___Black_rot",
        "Apple with Cedar Apple Rust": "Apple___Cedar_apple_rust",
        "Apple Healthy": "Apple___healthy",
        "Grape with Black Rot": "Grape___Black_rot",
        "Grape with Esca (Black Measles)": "Grape___Esca_(Black_Measles)",
        "Grape with Leaf Blight (Isariopsis Leaf Spot)": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape Healthy": "Grape___healthy",
        "Tomato with Bacterial Spot": "Tomato___Bacterial_spot",
        "Tomato with Early Blight": "Tomato___Early_blight",
        "Tomato with Late Blight": "Tomato___Late_blight",
        "Tomato with Leaf Mold": "Tomato___Leaf_Mold",
        "Tomato with Septoria Leaf Spot": "Tomato___Septoria_leaf_spot",
        "Tomato with Spider Mites (Two-spotted Spider Mite)": "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato with Target Spot": "Tomato___Target_Spot",
        "Tomato with Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato with Mosaic Virus": "Tomato___Tomato_mosaic_virus",
        "Tomato Healthy": "Tomato___healthy",
        "Potato with Early Blight": "Potato___Early_blight",
        "Potato with Late Blight": "Potato___Late_blight",
        "Potato Healthy": "Potato___healthy",
        "Pepper (Bell) with Bacterial Spot": "Pepper,_bell___Bacterial_spot",
        "Pepper (Bell) Healthy": "Pepper,_bell___healthy",
        "Bell Pepper with Bacterial Spot": "Pepper,_bell___Bacterial_spot",
        "Bell Pepper Healthy": "Pepper,_bell___healthy"
    }

    dict_key = label_mapping.get(label, label)
    tip = TREATMENT_TIPS.get(dict_key, DEFAULT_TIP)
    is_healthy = "healthy" in label.lower()

    return {
        "disease": label.replace("_", " ").replace("  ", " "),
        "confidence": confidence,
        "treatment_tip": tip,
        "status": "healthy" if is_healthy else "disease_detected"
    }
