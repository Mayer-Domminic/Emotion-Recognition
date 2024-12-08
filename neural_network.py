import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EmotionNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=6):
        super(EmotionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class EmotionNeuralNet:
    def __init__(self, max_features=5000, hidden_size=256, num_epochs=10, batch_size=32, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.max_features = max_features
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.accuracy = None

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def _prepare_data(self, X, y):
        X_tfidf = self.vectorizer.transform(X).toarray()
        dataset = EmotionDataset(X_tfidf, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _calculate_epoch_metrics(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the neural network on the given data
        """
        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()
        
        self.model = EmotionNet(
            input_size=self.max_features,
            hidden_size=self.hidden_size
        ).to(self.device)
        
        train_loader = self._prepare_data(X_train, y_train)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if X_test is not None and y_test is not None:
            test_loader = self._prepare_data(X_test, y_test)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = correct / total
            
            if X_test is not None and y_test is not None:
                epoch_test_loss, epoch_test_acc = self._calculate_epoch_metrics(test_loader, criterion)
            else:
                epoch_test_loss, epoch_test_acc = None, None
            
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            if epoch_test_loss is not None:
                self.history['test_loss'].append(epoch_test_loss)
                self.history['test_acc'].append(epoch_test_acc)
            
            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}]')
                print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
                if epoch_test_loss is not None:
                    print(f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}')
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        self.model.eval()
        X_test_tfidf = self.vectorizer.transform(X_test).toarray()
        test_loader = DataLoader(
            EmotionDataset(X_test_tfidf, np.zeros(len(X_test_tfidf))),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
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