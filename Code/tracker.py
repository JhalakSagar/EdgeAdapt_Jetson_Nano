import numpy as np

class Tracker:
    def __init__(self, max_lost=5):
        self.next_id = 0
        self.objects = {}
        self.lost = {}
        self.max_lost = max_lost

    def update(self, detections):
        updated_objects = {}

        for det in detections:
            x1, y1, x2, y2, score = det
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            matched_id = None

            for obj_id, (px, py) in self.objects.items():
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < 50:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            updated_objects[matched_id] = (cx, cy)
            self.lost[matched_id] = 0

        # Increase lost count
        for obj_id in list(self.objects.keys()):
            if obj_id not in updated_objects:
                self.lost[obj_id] += 1
                if self.lost[obj_id] > self.max_lost:
                    self.objects.pop(obj_id, None)
                    self.lost.pop(obj_id, None)

        self.objects.update(updated_objects)

        # Return tracked boxes
        results = []
        for det in detections:
            x1, y1, x2, y2, score = det
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            for obj_id, (px, py) in self.objects.items():
                if abs(cx - px) < 5 and abs(cy - py) < 5:
                    results.append([x1, y1, x2, y2, obj_id])

        return results
