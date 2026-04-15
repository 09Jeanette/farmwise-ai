from transformers import pipeline

SYSTEM_CONTEXT = (
    "You are FarmWise, a helpful agricultural advisor for smallholder farmers in Africa. "
    "Answer questions about crop diseases, planting seasons, soil preparation, irrigation, "
    "pest control, and farming best practices. "
    "Keep answers short, practical, and safe. "
    "If you are unsure, say so clearly and recommend consulting a local agricultural officer."
)

# Predefined safe responses for common questions
PREDEFINED = {
    "hello": "Hello! I am FarmWise, your farming assistant. Ask me anything about crops, diseases, or farming practices.",
    "hi": "Hi there! I am FarmWise. How can I help you with your farm today?",
    "help": "I can help you with: crop diseases, planting advice, soil preparation, pest control, and irrigation tips. Just ask!",
}

# Lazy loading for chatbot model to reduce memory usage
chatbot = None


def get_chatbot():
    global chatbot
    if chatbot is None:
        print("Loading chatbot model...")
        chatbot = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device_map="auto",  # Use GPU if available, otherwise CPU
            torch_dtype="auto",  # Use automatic dtype for memory efficiency
            model_kwargs={"low_cpu_mem_usage": True}
        )
        print("Chatbot model loaded successfully.")
    return chatbot


def unload_chatbot():
    """Unload the chatbot to free memory"""
    global chatbot
    if chatbot is not None:
        del chatbot
        chatbot = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Chatbot model unloaded to free memory.")


def get_farming_advice(question: str) -> str:
    question_lower = question.lower().strip()

    # Check predefined responses first
    for key, response in PREDEFINED.items():
        if key in question_lower:
            return response

    # Build prompt with context
    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"Farmer asks: {question}\n"
        f"Answer:"
    )

    try:
        chatbot = get_chatbot()
        result = chatbot(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            truncation=True
        )
        answer = result[0]["generated_text"].strip()

        if len(answer) < 10:
            return (
                "I am not confident about that answer. "
                "Please consult your local agricultural extension officer for accurate advice."
            )

        return answer

    except Exception as e:
        print(f"Chatbot error: {e}")
        return (
            "I am having trouble answering that right now. "
            "Please try again or consult your local agricultural officer."
        )
