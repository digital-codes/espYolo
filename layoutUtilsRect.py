
import numpy as np

def define_cells(image_size =(176,144), grid = (4,3)):
    cells = []
    # 1x1 regions (3x3 grid)
    cell_size = (image_size[0] // grid[0], image_size[1] // grid[1])
    for row in range(grid[1]):
        for col in range(grid[0]):
            x_min = col * cell_size[0] 
            y_min = row * cell_size[1]
            x_max = (col + 1) * cell_size[0] - 1 if col < grid[0] - 1 else image_size[0] # margin
            y_max = (row + 1) * cell_size[1] - 1 if row < grid[1] - 1 else image_size[1] # margin
            cells.append((x_min, y_min, x_max, y_max))
    return cells

def define_regions(cells, grid=(4,3)):
    regions = []
    # add all cells. cells are ordered row wise
    for c in cells:
        regions.append((cells.index(c), cells.index(c)))  # (start_cell, end_cell) is the same for single cell
    #enlarge in steps of 1 
    for s in range(1,min(grid)):
        for r in range(0, grid[1] - s):
            for c in range(0, grid[0] - s):
                start_cell = c + r * grid[0]
                end_cell = start_cell + (grid[0] + 1) * s
                regions.append((start_cell, end_cell))  # (start_cell, end_cell) is the same for single cell
    # add full size
    if grid[0] != grid[1]:
        regions.append((0, len(cells) - 1))            

    return regions

def find_cell(cells, x,y):
    """
    Find the cell index for the given coordinates (x, y).
    """
    for idx, (x_min, y_min, x_max, y_max) in enumerate(cells):
        if x_min <= int(x) <= x_max and y_min <= int(y) <= y_max:
            return idx
    print(f"[WARN] No cell found for coordinates ({x}, {y})")        
    return None 

def find_region(regions, start_cell, end_cell, grid=(4,3)):
    """
    Find the region index for the given start and end cell indices.
    """
    s_row = start_cell // grid[0]
    s_col = start_cell % grid[0]
    e_row = end_cell // grid[0]
    e_col = end_cell % grid[0]
    delta = max(e_row - s_row, e_col - s_col)
    if s_row + delta >= grid[1]:
        s_row -= grid[1] - delta
    if s_col + delta >= grid[0]:
        s_col -= grid[0] - delta
    e_row = s_row + delta
    e_col = s_col + delta
    start_cell = s_row * grid[0] + s_col
    end_cell = e_row * grid[0] + e_col
    for idx, (start, end) in enumerate(regions):
        if start == start_cell and end == end_cell:
            return idx
    return None

def map_bbox(cells, regions, bbox,grid=(4,3)):
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    wadjust = width // 20
    hadjust = height // 20
    # Adjust bbox to avoid edge cases
    start_cell = find_cell(cells, bbox[0] + wadjust, bbox[1] + hadjust)
    end_cell = find_cell(cells, bbox[2] - wadjust, bbox[3] - hadjust)
    region = find_region(regions, start_cell, end_cell,grid)
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

