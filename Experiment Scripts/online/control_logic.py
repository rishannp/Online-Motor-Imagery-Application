from collections import deque

class ControlLogic:
    def __init__(self, buffer_size, threshold):
        """ buffer the last N preds and emit only on majority."""
        self.buffer = deque(maxlen=buffer_size)
        self.threshold = threshold

    def update(self, pred: int):
        """
        Append the latest pred; if any class reaches `threshold`
        Return that class, else None.
        """
        self.buffer.append(pred)
        if len(self.buffer) == self.buffer.maxlen:
            counts = {lab: self.buffer.count(lab) for lab in set(self.buffer)}
            top, cnt = max(counts.items(), key=lambda x: x[1])
            if cnt >= self.threshold:
                return top
        return None
