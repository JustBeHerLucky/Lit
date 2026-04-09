from collections import deque, Counter
from config import CLASSES

class FrequencyEstimator:
    def __init__(self):
        self.memory = deque(maxlen=200)
        self.counter = Counter()

    def update(self, result):
        self.memory.append(result)
        self.counter[result] += 1

    def get_probabilities(self):
        total = len(self.memory)
        return {
            cls: self.counter[cls] / total if total else 1 / len(CLASSES)
            for cls in CLASSES
        }
