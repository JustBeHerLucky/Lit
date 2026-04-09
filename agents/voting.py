import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from config import CLASSES

class VotingAgent:
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss')
        self.scaler = StandardScaler()
        self.memory = deque(maxlen=200)
        self.classes_seen = False

    def update(self, result):
        self.memory.append(CLASSES.index(result))
        if len(self.memory) >= 5:
            X = self.scaler.fit_transform(
                [list(self.memory)[i:i+3] for i in range(len(self.memory) - 3)]
            )
            y = list(self.memory)[3:]
            if not self.classes_seen:
                self.model.partial_fit(X, y, classes=np.arange(len(CLASSES)))
                self.classes_seen = True
            else:
                self.model.partial_fit(X, y)

    def predict_proba(self, X):
        if not self.classes_seen:
            return np.ones(len(CLASSES)) / len(CLASSES)
        return self.model.predict_proba(self.scaler.transform(X))[0]