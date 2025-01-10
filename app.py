import joblib
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = joblib.load('./model/model.pkl')
vectorizer = joblib.load('./model/vectorizer.pkl')

# Flask app initialization
app = Flask(__name__)

def predict_new_text(new_text, model, vectorizer):
    """
    Predict whether a given text is 'fake' or 'real'.
    
    Parameters:
    - new_text (str): The text to predict on.
    - model: The trained classifier model.
    - vectorizer: The trained TF-IDF vectorizer.
    
    Returns:
    - str: 'Real' or 'Fake' depending on the prediction.
    """
    # Preprocess the new text (same preprocessing applied to the training data)
    new_text_cleaned = new_text.lower()  # Convert to lowercase
    new_text_cleaned = ''.join([char for char in new_text_cleaned if char.isalpha() or char.isspace()])  # Remove non-alphabetic characters

    # Convert the new text into TF-IDF features
    new_text_tfidf = vectorizer.transform([new_text_cleaned])

    # Make the prediction using the model
    prediction = model.predict(new_text_tfidf)

    # Map prediction to 'Real' or 'Fake'
    if prediction == 0:
        return 'Real'
    else:
        return 'Fake'


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = " "  # Initial empty prediction
    if request.method == "POST":
        # Get the text from the form input
        user_input = request.form["user_text"]
        # Predict the label using the model and vectorizer
        prediction = predict_new_text(user_input, model, vectorizer)
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)



# pip freeze > requirements.txt 