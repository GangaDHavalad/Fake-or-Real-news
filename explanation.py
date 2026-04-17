def get_important_words(text, vectorizer, model, top_n=5):
    """
    Returns top important words influencing prediction
    """

    # Transform text
    vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    vector_array = vector.toarray()[0]

    # Get weights
    try:
        weights = model.coef_[0]   # Logistic Regression
    except:
        weights = model.feature_log_prob_[0]  # Naive Bayes

    important_words = []

    # Calculate importance
    for i in range(len(vector_array)):
        if vector_array[i] > 0:
            score = vector_array[i] * weights[i]
            word = feature_names[i]
            important_words.append((word, round(float(score), 3)))

    # Sort by importance
    important_words = sorted(
        important_words,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return important_words[:top_n]


# -------------------------------
# 🧠 Summary Generator (IMPORTANT)
# -------------------------------
def generate_summary(prediction, important_words):
    """
    Generate explanation summary based on prediction
    """

    # Extract top words
    top_words = [word for word, score in important_words[:3]]

    if not top_words:
        return "The prediction is based on overall text patterns learned by the model."

    word_text = ", ".join(top_words)

    if prediction == 1:
        return (
            f"This article is classified as REAL news because it contains reliable "
            f"and commonly used terms such as {word_text}, which are often found in genuine news reports."
        )
    else:
        return (
            f"This article is classified as FAKE news because it includes suspicious "
            f"or misleading terms like {word_text}, which are frequently associated with false or misleading content."
        )