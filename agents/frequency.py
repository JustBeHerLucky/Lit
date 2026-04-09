from collections import deque, Counter
from config import CLASSES

class FrequencyEstimator:
    def __init__(self):
        self.memory = deque(maxlen=200)
        self.counter = Counter()

    def reset(self):
        self.memory.clear()
        self.counter.clear()

    def update(self, result):
        if len(self.memory) == self.memory.maxlen:
            oldest = self.memory[0]
            self.counter[oldest] -= 1
            if self.counter[oldest] <= 0:
                del self.counter[oldest]
        self.memory.append(result)
        self.counter[result] += 1

    def get_probabilities(self):
        total = len(self.memory)
        return {
            cls: self.counter[cls] / total if total else 1 / len(CLASSES)
            for cls in CLASSES
        }
