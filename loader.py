from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class EmotionDataLoader:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.val_data = None
    
    def load_data(self, split_type="split"):
        """
        Load emotion dataset from Hugging Face.
        Args:
            split_type (str): Either "split" or "unsplit"
        """
        if split_type == "split":
            dataset = load_dataset("dair-ai/emotion", "split")
            self.train_data = pd.DataFrame(dataset['train'])
            self.test_data = pd.DataFrame(dataset['test'])
            self.val_data = pd.DataFrame(dataset['validation'])
            
            self.train_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
            
            X_train = self.train_data['text'].values
            y_train = self.train_data['label'].values
            X_test = self.test_data['text'].values
            y_test = self.test_data['label'].values
            
        else:
            dataset = load_dataset("dair-ai/emotion", "unsplit")
            data = pd.DataFrame(dataset['train'])
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['text'].values,
                data['label'].values,
                test_size=0.2,
                random_state=42,
                stratify=data['label'].values
            )
        
        return X_train, X_test, y_train, y_test
    
    def get_label_names(self):
        """Return the emotion label names"""
        return ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']