from loader import EmotionDataLoader
from logistic_regression import EmotionLogisticRegression
from xg_boost import EmotionXGBoost
from rf_classifier import EmotionRandomForest
from svm_classifier import EmotionSVM
from neural_network import EmotionNeuralNet
from vis import plot_model_comparison, plot_nn_training
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

os.makedirs('emotion_plots', exist_ok=True)

data_loader = EmotionDataLoader()
X_train_full, X_test, y_train_full, y_test = data_loader.load_data(split_type="split")

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42,
    stratify=y_train_full
)

lr_model = EmotionLogisticRegression(max_features=5000)
lr_model.train(X_train, y_train)
lr_accuracy = lr_model.evaluate(X_test, y_test)
print("\nLogistic Regression Results:")
print(f"Accuracy: {lr_accuracy:.4f}")
print("\nDetailed Report:")
print(lr_model.get_detailed_report(X_test, y_test))

xgb_model = EmotionXGBoost(max_features=5000)
xgb_model.train(X_train, y_train)
xgb_accuracy = xgb_model.evaluate(X_test, y_test)
print("\nXGBoost Results:")
print(f"Accuracy: {xgb_accuracy:.4f}")
print("\nDetailed Report:")
print(xgb_model.get_detailed_report(X_test, y_test))

rf_model = EmotionRandomForest(max_features=5000)
rf_model.train(X_train, y_train)
rf_accuracy = rf_model.evaluate(X_test, y_test)
print("\nRandom Forest Results:")
print(f"Accuracy: {rf_accuracy:.4f}")
print("\nDetailed Report:")
print(rf_model.get_detailed_report(X_test, y_test))

svm_model = EmotionSVM(max_features=5000)
svm_model.train(X_train, y_train)
svm_accuracy = svm_model.evaluate(X_test, y_test)
print("\nSVM Results:")
print(f"Accuracy: {svm_accuracy:.4f}")
print("\nDetailed Report:")
print(svm_model.get_detailed_report(X_test, y_test))

nn_model = EmotionNeuralNet(max_features=5000)
nn_model.train(X_train, y_train, X_test, y_test)
nn_accuracy = nn_model.evaluate(X_test, y_test)
print("\nNeural Network Results:")
print(f"Accuracy: {nn_accuracy:.4f}")
print("\nDetailed Report:")
print(nn_model.get_detailed_report(X_test, y_test))

classification_results = {
    'LinearRegression': {
        'train_acc': lr_model.evaluate(X_train, y_train),
        'test_acc': lr_model.evaluate(X_test, y_test),
        'val_acc': lr_model.evaluate(X_val, y_val)
    },
    'XGBoost': {
        'train_acc': xgb_model.evaluate(X_train, y_train),
        'test_acc': xgb_model.evaluate(X_test, y_test),
        'val_acc': xgb_model.evaluate(X_val, y_val)
    },
    'RandomForest': {
        'train_acc': rf_model.evaluate(X_train, y_train),
        'test_acc': rf_model.evaluate(X_test, y_test),
        'val_acc': rf_model.evaluate(X_val, y_val)
    },
    'SVM': {
        'train_acc': svm_model.evaluate(X_train, y_train),
        'test_acc': svm_model.evaluate(X_test, y_test),
        'val_acc': svm_model.evaluate(X_val, y_val)
    },
    'NeuralNetwork': {
        'train_acc': nn_model.evaluate(X_train, y_train),
        'test_acc': nn_model.evaluate(X_test, y_test),
        'val_acc': nn_model.evaluate(X_val, y_val)
    }
}

fig_classification = plot_model_comparison(classification_results)
fig_classification.write_image("emotion_plots/model_comparison.png")
fig_classification.show()

if hasattr(nn_model, 'history'):
    fig_nn = plot_nn_training(nn_model.history)
    fig_nn.write_image("emotion_plots/neural_network_training.png")
    fig_nn.show()

def predict_emotion(text, models):
    emotion_labels = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    
    text_input = [text]
    separator = "=" * 60
    
    print(separator)
    print(f"📝 Analyzing Text: '{text}'")
    print(separator)
    
    predictions = {}
    
    for model_name, model in models.items():
        prediction = model.predict(text_input)
        predicted_label = prediction[0]
        
        confidence = None
        if hasattr(model.model, 'predict_proba'):
            probabilities = model.model.predict_proba(model.vectorizer.transform(text_input))
            confidence = probabilities[0][predicted_label]
            
        predictions[model_name] = {
            'prediction': predicted_label,
            'confidence': confidence
        }
        
        print(f"\n🤖 {model_name} Model")
        print("-" * 30)
        emotion_name = emotion_labels.get(predicted_label, f"Unknown ({predicted_label})")
        print(f"Prediction: {emotion_name}")
    
    print(f"\n{separator}")
    
    return predictions

trained_models = {
    'LinearRegression': lr_model,
    'XGBoost': xgb_model,
    'RandomForest': rf_model,
    'SVM': svm_model,
    'NeuralNetwork': nn_model
}

text = "im feeling bitter"
predict_emotion(text, trained_models)