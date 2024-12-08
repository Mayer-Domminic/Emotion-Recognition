from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class EmotionLogisticRegression:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.accuracy = None
    
    def train(self, X_train, y_train):
        """
        Train the model on the given data
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and return accuracy
        """
        predictions = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions)
        return self.accuracy
    
    def get_detailed_report(self, X_test, y_test):
        """
        Get detailed classification report
        """
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)