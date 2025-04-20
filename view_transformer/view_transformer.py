import cv2
import numpy as np

class ViewTransformer:
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915],
        ], dtype=np.float32)

        self.world_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width],
        ], dtype=np.float32)

        self.transform_matrix = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.world_vertices
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
            return None

        src = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.transform_matrix)
        return dst.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for obj_name, frames in tracks.items():
            for frame_idx, frame_tracks in enumerate(frames):
                for track_id, info in frame_tracks.items():
                    pos = info.get('position_adjusted')
                    if pos is None:
                        continue
                    transformed = self.transform_point(pos)
                    if transformed is not None:
                        transformed = transformed.squeeze().tolist()
                    # Store back into the same structure
                    frame_tracks[track_id]['position_transformed'] = transformed
