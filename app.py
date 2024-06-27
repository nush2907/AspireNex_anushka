import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    file_path = 'spam.csv'  
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

# Preprocess and split the data
def preprocess_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Train the model
def train_model(X_train_tfidf, y_train):
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    return classifier

# Evaluate the model
def evaluate_model(classifier, X_test_tfidf, y_test):
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_matrix, y_pred

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Plot ROC curve
def plot_roc_curve(classifier, X_test_tfidf, y_test):
    fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test_tfidf)[:, 1])
    roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test_tfidf)[:, 1])
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Predict single message
def predict_message(classifier, vectorizer, message):
    message_tfidf = vectorizer.transform([message])
    prediction = classifier.predict(message_tfidf)[0]
    return 'Spam' if prediction == 1 else 'Ham'

# Streamlit app
def main():
    st.title('SMS Spam Classifier')
    data = load_data()
    st.write(data.head())

    if st.button('Train Model'):
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(data)
        classifier = train_model(X_train_tfidf, y_train)
        accuracy, report, conf_matrix, y_pred = evaluate_model(classifier, X_test_tfidf, y_test)
        
        st.write(f'Accuracy: {accuracy}')
        st.write('Classification Report:')
        st.text(report)
        st.write('Confusion Matrix:')
        plot_confusion_matrix(conf_matrix)
        st.write('ROC Curve:')
        plot_roc_curve(classifier, X_test_tfidf, y_test)

    st.write("\n")
    st.write("### Classify a new message")
    user_message = st.text_area("Enter the message:")
    if st.button('Classify'):
        if user_message:
            X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(data)
            classifier = train_model(X_train_tfidf, y_train)
            prediction = predict_message(classifier, vectorizer, user_message)
            st.write(f'The message is classified as: {prediction}')
        else:
            st.write("Please enter a message to classify")

if __name__ == '__main__':
    main()
