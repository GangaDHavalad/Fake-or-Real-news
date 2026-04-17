from flask import Flask, render_template, request
import pickle
import nltk
from utils import clean_text
from explanation import get_important_words

# Download stopwords
nltk.download('stopwords')

app = Flask(__name__)

# -------------------------------
# 📥 Load model & vectorizer
# -------------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# -------------------------------
# 🧠 Summary function
# -------------------------------
def generate_summary(prediction, words):
    if prediction == 1:
        return f"This news appears real because of words like {', '.join([w for w, _ in words[:3]])}."
    else:
        return f"This news appears fake due to words like {', '.join([w for w, _ in words[:3]])}."

# -------------------------------
# 🌐 Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news', '')

    if not text.strip():
        return render_template('index.html', prediction="⚠ Please enter news text")

    # Clean text
    cleaned = clean_text(text)

    # Vectorize
    vector = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vector)[0]
    result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"

    # Explanation
    important_words = get_important_words(cleaned, vectorizer, model)

    # Summary
    summary = generate_summary(prediction, important_words)

    return render_template(
        'index.html',
        prediction=result,
        explanation=important_words,
        summary=summary
    )

# -------------------------------
# ▶ Run app on localhost
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Server starting...")
    print("👉 Open this in browser: http://127.0.0.1:5000/\n")
    app.run(host="127.0.0.1", port=5000, debug=True)