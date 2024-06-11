from panda_vision_imitation.geometry import Box

class Tracker:

    def __init__(self, store_hist=True):

        # Bounding boxes for currious and current frame
        self.prev_bbox = None
        self.curr_bbox = None

        # Counter for frame
        self.frame_id = 0

        self.store_hist = store_hist

        if self.store_hist:
            self.hist = list()

    def track(self, bbox: Box):

        if self.frame_id == 0:
            # For the first frame previous and current are the same
            self.prev_bbox = bbox
            self.curr_bbox = bbox
        else:
            # For every subsequent frame
            # set curr to prev
            self.prev_bbox = self.curr_bbox
            self.curr_bbox = bbox

            # If current bbox is None set it to previous
            if self.curr_bbox is None:
                self.curr_bbox = self.prev_bbox

        # Store history if needed
        if self.store_hist:
            self.hist.append(dict(frame_id=self.frame_id,
                                  prev_bbox=self.prev_bbox,
                                  curr_bbox=self.curr_bbox))

        self.frame_id += 1
        return self.prev_bbox, self.curr_bbox
