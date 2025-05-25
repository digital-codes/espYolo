
import numpy as np

def define_cells(image_size, grid = 3):
    cells = []
    # 1x1 regions (3x3 grid)
    cell_size = image_size // grid
    for row in range(grid):
        for col in range(grid):
            x_min = col * cell_size 
            y_min = row * cell_size
            x_max = (col + 1) * cell_size - 1 if col < grid - 1 else image_size # margin
            y_max = (row + 1) * cell_size - 1 if row < grid - 1 else image_size # margin
            cells.append((x_min, y_min, x_max, y_max))
    return cells

def define_regions(grid=3):
    regions = []
    # enlarge in x and y
    for height in range(1, grid+1):
        for width in range(1, grid+1):
            for y in range(0, grid - height + 1):
                for x in range(0, grid - width + 1):
                    # print(f"Adding region {x},{y},{width}x{height} to regions")
                    start_cell = x + y * grid
                    end_cell = start_cell + width - 1 + (height - 1) * grid
                    # print("Start/End :", start_cell, end_cell)
                    regions.append((start_cell,end_cell))
    # add full size region
    return regions

def find_cell(cells, x,y):
    """
    Find the cell index for the given coordinates (x, y).
    """
    for idx, (x_min, y_min, x_max, y_max) in enumerate(cells):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return idx
    return None 

def find_region(regions, start_cell, end_cell):
    """
    Find the region index for the given start and end cell indices.
    """
    for idx, (start, end) in enumerate(regions):
        if start == start_cell and end == end_cell:
            return idx
    return None

def map_bbox(cells, regions, bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    wadjust = width // 20
    hadjust = height // 20
    # Adjust bbox to avoid edge cases
    start_cell = find_cell(cells, bbox[0] + wadjust, bbox[1] + hadjust)
    end_cell = find_cell(cells, bbox[2] - wadjust, bbox[3] - hadjust)
    region = find_region(regions, start_cell, end_cell)
    return region
    

def create_label_vector(cells, regions, bboxes, class_ids, num_classes):
    """
    Assign each bbox to the single best-matching region based on IoU.
    """
    num_regions = len(regions)
    label = np.zeros(num_classes * num_regions, dtype=np.float32)

    for bbox, class_id in zip(bboxes, class_ids):

        region = map_bbox(cells, regions, bbox)
        if region is None:
            print(f"[WARN] No region found for bbox {bbox} with class {class_id}")
            continue
        if class_id < 0 or class_id >= num_classes:
            print(f"[WARN] Invalid class_id {class_id} for bbox {bbox}")
            continue
        
        idx = class_id * num_regions + region
        label[idx] = 1.0
        #print(f"Assigned bbox {bbox} with class {class_id} to region {region}")

    return label


def convert_to_softmax_labels(label_vec, num_classes, num_regions):
    """
    Converts a flat (C × R) multi-label vector to (R,) softmax label.
    Assumes: at most 1 active class per region.
    
    Inputs:
        label_vec: (num_classes * num_regions,) with 0/1 (multi-hot per class-region)
        num_classes: number of foreground classes
        num_regions: number of spatial regions

    Returns:
        labels: (num_regions,) int32, with 0 = background, 1..C = classes
    """
    label_vec = label_vec.reshape((num_classes, num_regions))
    
    softmax_labels = np.zeros((num_regions,), dtype=np.int32)  # default: background

    for region_id in range(num_regions):
        active_classes = np.where(label_vec[:, region_id] > 0.5)[0]  # which class is set
        if len(active_classes) == 1:
            softmax_labels[region_id] = active_classes[0] + 1  # shift: 1 = class 0
        elif len(active_classes) > 1:
            # multiple labels in one region → choose highest index or warn
            softmax_labels[region_id] = active_classes[0] + 1  # or resolve differently
            print(f"[Warning] Multiple classes in region {region_id}, choosing class {active_classes[0]}")
        # else: leave as background (0)

    return softmax_labels

