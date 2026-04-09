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

    def reset(self):
        self.memory.clear()
        self.classes_seen = False
        self.model = SGDClassifier(loss='log_loss')
        self.scaler = StandardScaler()

    def update(self, result):
        self.memory.append(CLASSES.index(result))
        if len(self.memory) >= 5:
            memory_list = list(self.memory)
            X = self.scaler.fit_transform(
                [memory_list[i:i+3] for i in range(len(memory_list) - 3)]
            )
            y = memory_list[3:]
            if not self.classes_seen:
                self.model.partial_fit(X, y, classes=np.arange(len(CLASSES)))
                self.classes_seen = True
            else:
                self.model.partial_fit(X, y)

    def predict_proba(self, X):
        if not self.classes_seen:
            return np.ones(len(CLASSES)) / len(CLASSES)
        return self.model.predict_proba(self.scaler.transform(X))[0]
