# FarmWise AI API

AI-powered plant disease detection and farming chatbot API built with FastAPI.

## Endpoints

### GET /health

Returns API health status.

### POST /predict

Detects plant disease from a leaf image.

**Request:** multipart/form-data

- `file` — JPEG or PNG image of a plant leaf

**Response:**

```json
{
  "disease": "Tomato Late Blight",
  "confidence": 0.94,
  "treatment_tip": "Remove and destroy infected plants. Apply copper fungicide immediately.",
  "status": "disease_detected"
}
```

### POST /chat

Returns farming advice for a text question.

**Request:** application/json

```json
{
  "question": "How do I treat tomato late blight?"
}
```

**Response:**

```json
{
  "answer": "Apply copper-based fungicide and remove infected leaves immediately.",
  "is_ai": true
}
```

---

## API Usage Guide

### Predictor API

**Endpoint:** `POST /predict`

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: file (JPEG/PNG image)

**Example:**

```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:8000/predict
```

### Chatbot API

**Endpoint:** `POST /chat`

**Request:**

- Method: POST
- Content-Type: application/json
- Body: {"question": "your farming question"}

**Example:**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"question":"How to treat tomato blight?"}' http://localhost:8000/chat
```

### Quick Node.js Example

```javascript
// Disease prediction
const formData = new FormData();
formData.append("file", imageFile);
fetch("http://localhost:8000/predict", { method: "POST", body: formData });

// Chatbot query
fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "farming question" }),
});
```

---

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn main:app --reload
```

API runs at: http://localhost:8000
Docs at: http://localhost:8000/docs

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set the following:
   - **Environment:** Docker
   - **Branch:** main
5. Click Deploy

Your API will be live at: `https://farmwise-ai.onrender.com`

---

## Models Used

- **Disease Detection:** `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (HuggingFace)
- **Chatbot:** `google/flan-t5-base` (HuggingFace)
