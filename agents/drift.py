from river import drift as river_drift

class SimpleDDM:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 1
        self.error_sum = 0
        self.min_error = float('inf')
        self.min_std = float('inf')
        self.drift_detected = False

    def update(self, is_correct):
        error = 1 - is_correct
        self.error_sum += error
        p = self.error_sum / self.n
        s = (p * (1 - p) / self.n) ** 0.5

        if p + s < self.min_error + self.min_std:
            self.min_error = p
            self.min_std = s

        if p + s > self.min_error + 2 * self.min_std:
            self.drift_detected = True
        else:
            self.drift_detected = False

        self.n += 1
        return self.drift_detected

# Instantiate drift detectors
ph_drift = river_drift.PageHinkley()
adwin_drift = river_drift.ADWIN()
simple_ddm = SimpleDDM()

def reset_drift_detectors():
    global ph_drift, adwin_drift
    ph_drift = river_drift.PageHinkley()
    adwin_drift = river_drift.ADWIN()
    simple_ddm.reset()

def detect_drift(predicted_correct, consecutive_losses):
    return (
        adwin_drift.update(int(predicted_correct)) or
        (ph_drift.update(int(predicted_correct)) is not None) or
        simple_ddm.update(int(predicted_correct)) or
        (consecutive_losses >= 3)
    )
