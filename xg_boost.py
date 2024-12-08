from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class EmotionXGBoost:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            random_state=42,
            objective='multi:softprob',
            num_class=6
        )
        self.accuracy = None
    
    def train(self, X_train, y_train):
        """
        Train the XGBoost model on the given data
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        X_train_dense = X_train_tfidf.toarray()
        
        self.model.fit(X_train_dense, y_train)
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        X_test_tfidf = self.vectorizer.transform(X_test)
        X_test_dense = X_test_tfidf.toarray()
        return self.model.predict(X_test_dense)
    
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