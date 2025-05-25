import numpy as np

class STrack:
    """
    Single track object with state, position, and motion information
    """
    _count = 0
    
    def __init__(self, tlbr, score, cls_id):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = score
        self.cls_id = cls_id
        self.is_activated = False
        
        self.track_id = 0
        self.state = 'new'
        
        self.age = 0
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0
        
    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def mark_lost(self):
        """Mark track as lost (not tracked but possibly reidentified later)"""
        self.state = 'lost'
        self.end_frame = self.frame_id
        
    def update(self, new_track, frame_id):
        """
        Update a matched track
        Args:
            new_track (STrack): New STrack object with new detection
            frame_id (int): Current frame id
        """
        self.frame_id = frame_id
        self.tlbr = new_track.tlbr
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        self.state = 'tracked'
        self.is_activated = True
        self.age = 0
        
    @property
    def tlwh(self):
        """Get current position in (top, left, width, height) format."""
        x1, y1, x2, y2 = self.tlbr
        return np.array([x1, y1, x2 - x1, y2 - y1])
        
    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count


class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.tracked_tracks = []  # Type: List[STrack]
        self.lost_tracks = []     # Type: List[STrack]
        self.removed_tracks = []  # Type: List[STrack]
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
    def update(self, dets: np.ndarray):
        """
        Update tracks using detection results of the current frame
        
        Args:
            dets: shape (N, 6), [x1, y1, x2, y2, score, class]
        
        Returns:
            List[STrack]: The active tracks with tracking info
        """
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Convert detections to STrack objects
        detections = []
        if len(dets) > 0:
            for det in dets:
                bbox = det[:4]
                score = det[4]
                cls = det[5]
                if score >= self.track_thresh:
                    detections.append(STrack(bbox, score, cls))
        
        # Add newly detected tracks to tracked_tracks
        for track in self.tracked_tracks:
            if not track.is_activated:
                # previous track which is not active in the last frame but is recorded in tracked_tracks
                self.tracked_tracks.remove(track)
                
        # First association, with high score detection boxes
        track_pool = self.tracked_tracks
        
        # Simple association based on IoU matching
        matched_pairs, unmatched_tracks, unmatched_detections = self._associate_tracks_with_detections(track_pool, detections)
        
        for track_idx, det_idx in matched_pairs:
            track = track_pool[track_idx]
            det = detections[det_idx]
            
            # If they match, update the track
            track.update(det, self.frame_id)
            activated_tracks.append(track)
        
        # Handle unmatched tracks
        for i in unmatched_tracks:
            track = track_pool[i]
            
            # Mark unmatched track as lost
            if not track.state == 'lost':
                track.mark_lost()
                lost_tracks.append(track)
        
        # Handle unmatched detections
        for i in unmatched_detections:
            det = detections[i]
            # Initialize a new track
            new_track = STrack(det.tlbr, det.score, det.cls_id)
            new_track.activate(self.frame_id)
            activated_tracks.append(new_track)
        
        # Remove lost tracks that have been lost for too long
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.track_buffer:
                removed_tracks.append(track)
        
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == 'tracked']
        self.tracked_tracks = self.tracked_tracks + activated_tracks
        
        self.lost_tracks = [t for t in self.lost_tracks if t not in removed_tracks]
        self.lost_tracks = self.lost_tracks + lost_tracks
        
        self.removed_tracks = removed_tracks
        
        output_tracks = [track for track in self.tracked_tracks if track.is_activated]
        
        return output_tracks
    
    def _associate_tracks_with_detections(self, tracks, detections):
        """
        Simple IoU-based matching
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._calc_iou(track.tlbr, det.tlbr)
        
        # Hungarian algorithm to solve assignment problem
        matched_indices = []
        
        # Use greedy matching for simplicity
        while True:
            # Find highest IoU
            if iou_matrix.size == 0:
                break
            
            # Get index of max IoU
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Check if this is a valid match
            if iou_matrix[i, j] < self.match_thresh:
                break
            
            matched_indices.append((i, j))
            
            # Remove matched possibilities
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        matched_tracks = []
        matched_detections = []
        for m in matched_indices:
            matched_tracks.append(m[0])
            matched_detections.append(m[1])
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        
        return matched_indices, unmatched_tracks, unmatched_detections
    
    def _calc_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes
        """
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        
        # Calculate intersection area
        xx1 = max(x1, x1_)
        yy1 = max(y1, y1_)
        xx2 = min(x2, x2_)
        yy2 = min(y2, y2_)
        
        # Check if there is an intersection
        if xx2 < xx1 or yy2 < yy1:
            return 0.0
        
        # Calculate areas
        inter_area = (xx2 - xx1) * (yy2 - yy1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        
        # Calculate IoU
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou