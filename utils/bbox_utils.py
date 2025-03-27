def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_bbox_wdth(bbox):
    return bbox[2]-bbox[0]