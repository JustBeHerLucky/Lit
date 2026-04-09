import numpy as np
from collections import defaultdict
from config import CLASSES, PATTERN_LENGTH, PATTERN_DECAY, STUCK_THRESHOLD, BAIT_THRESHOLD
from state import history, stats

def pattern_inner_default():
    return [0.0, 0]

def pattern_outer_default():
    return defaultdict(pattern_inner_default)

pattern_memory = defaultdict(pattern_outer_default)

def adjust_probs(combined_probs, pattern_memory, stuck_counter, bait_counter, current_round):
    if len(history) >= PATTERN_LENGTH:
        pattern = tuple(history)[-PATTERN_LENGTH:]
        total_pattern = sum([score for score, _ in pattern_memory[pattern].values()])
        if total_pattern > 0:
            for cls, (score, last_seen) in pattern_memory[pattern].items():
                age = current_round - last_seen
                decayed = score * (PATTERN_DECAY ** age)
                winrate = stats["class_correct"][cls] / stats["class_called"][cls] if stats["class_called"][cls] else 0.5
                boost = 0.3 + (winrate - 0.5) * 0.5
                combined_probs[CLASSES.index(cls)] += (decayed / total_pattern) * boost

    for idx, cls in enumerate(CLASSES):
        if stuck_counter[cls] >= STUCK_THRESHOLD:
            combined_probs[idx] *= 0.5
        if bait_counter[cls] >= BAIT_THRESHOLD:
            combined_probs[idx] += 0.05

    combined_probs /= combined_probs.sum()
    return combined_probs
