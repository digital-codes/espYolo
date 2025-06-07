
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
    for i,(start, end) in enumerate(regions):
        if start == start_cell and end == end_cell:
            return i

    s_row = start_cell // grid[0]
    s_col = start_cell % grid[0]
    e_row = end_cell // grid[0]
    e_col = end_cell % grid[0]
    # adjust the start and end cells to ensure they are within the square region bounds   
    delta = max(e_row - s_row, e_col - s_col)
    print(f"delta: {delta} for cells ({start_cell}, {end_cell}, {s_row}, {s_col}, {e_row}, {e_col})")
    if s_row + delta >= grid[1]:
        s_row = 0
    if s_col + delta >= grid[0]:
        s_col = 0
    e_row = s_row + delta - 1
    e_col = s_col + delta - 1
    start_cell = s_row * grid[0] + s_col
    end_cell = e_row * grid[0] + e_col
    for idx, (start, end) in enumerate(regions):
        if start == start_cell and end == end_cell:
            return idx
    if grid[0] != grid[1]:
        return len(regions) - 1  # return the full size region if no other region is found
    else:
        return None

def map_bbox(cells, regions, grid, bbox):
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    wadjust = width // 20
    hadjust = height // 20
    # Adjust bbox to avoid edge cases
    start_cell = find_cell(cells, bbox[0] + wadjust, bbox[1] + hadjust)
    end_cell = find_cell(cells, bbox[2] - wadjust, bbox[3] - hadjust)
    if start_cell is None or start_cell >= len(cells) or end_cell is None or end_cell >= len(cells):
        print(f"[WARN] Invalid bbox {bbox} with start_cell {start_cell} or end_cell {end_cell}")
        return None
    region = find_region(regions, start_cell, end_cell,grid)
    return region
    

def create_label_vector(cells, regions, grid, bboxes, class_ids, num_classes,output_size,reg_items):
    """
    Assign each bbox to the single best-matching region based on IoU.
    """
    num_regions = len(regions)
    vec = np.zeros(output_size, dtype=np.float32)

    # init region items
    region_fill = np.zeros(num_regions, dtype=np.int32)

    for bbox, class_id in zip(bboxes, class_ids):

        region = map_bbox(cells, regions, grid, bbox)
        if region is None:
            print(f"[WARN] No region found for bbox {bbox} with class {class_id}")
            continue
        # region already full?
        if region_fill[region] >= reg_items:
            print(f"[WARN] Region {region} already filled for bbox {bbox} with class {class_id}")
            continue
        region_fill[region] += 1
        if class_id < 0 or class_id >= num_classes:
            print(f"[WARN] Invalid class_id {class_id} for bbox {bbox}")
            continue
        # increment class_id. 0 is invalid
        class_id += 1
        
        idx = region * (reg_items * 6) + (region_fill[region] - 1) * 6
        vec[idx] = 1.0
        vec[idx + 1] = class_id
        vec[idx + 2] = bbox[0]
        vec[idx + 3] = bbox[1]
        vec[idx + 4] = bbox[2]
        vec[idx + 5] = bbox[3]  
        #print(f"Assigned bbox {bbox} with class {class_id} to region {region}")

    return vec


