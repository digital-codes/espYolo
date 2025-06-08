
import numpy as np

def define_cells(image_size =(176,144), grid = (4,3)):
    cells = []
    # 1x1 regions (3x3 grid)
    cell_size = (image_size[0] // grid[0], image_size[1] // grid[1])
    for row in range(grid[1]):
        for col in range(grid[0]):
            x_min = col * cell_size[0] 
            y_min = row * cell_size[1]
            x_max = (col + 1) * cell_size[0] - 1 if col < grid[0] - 1 else image_size[0] - 1 # margin
            y_max = (row + 1) * cell_size[1] - 1 if row < grid[1] - 1 else image_size[1] - 1 # margin
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
    if bbox[2] >= cells[grid[0]-1][2]:
        bbox[2] = cells[grid[0]-1][2] - 1  # adjust right edge to avoid out of bounds
    if bbox[3] >= cells[-1][3]:
        bbox[3] = cells[-1][3] - 1  # adjust bottom edge to avoid out of bounds
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
    

def create_label_vector(cells, regions, grid, bboxes, class_ids, class_num, reg_items=1, item_size=1,mode="region"):
    if mode == "region":
        num_regions = len(regions)
        # all zero
        set_classes = np.zeros(class_num*num_regions, dtype=np.int8) 
        vec = np.zeros(class_num * num_regions, dtype=np.float32) 
        for bbox, class_id in zip(bboxes, class_ids):
            region = map_bbox(cells, regions, grid, bbox)
            if region is None:
                print(f"[WARN] No region found for bbox {bbox} with class {class_id}")
                continue
            if set_classes[class_id*region] > 0:
                # first entry per class is good. input sorted by distance
                continue
            if class_id < 0 or class_id >= class_num:
                print(f"[WARN] Invalid class_id {class_id} for bbox {bbox}")
                continue
            idx = class_id + region * class_num
            vec[idx] = 1.0
            set_classes[class_id*region] = 1

        
    elif mode == "yolo":    
        # class 0 (empty) probability 1
        img_width = cells[-1][2] + 1
        img_height = cells[-1][3] + 1
        itemVec = np.zeros(item_size, dtype=np.float32)
        itemVec[0] = 1.0 
        vec = itemVec.copy()
        for i in range(1, num_regions * reg_items):
            vec = np.concatenate((vec, itemVec), axis=0)
        
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
            if class_id < 0 or class_id >= class_num:
                print(f"[WARN] Invalid class_id {class_id} for bbox {bbox}")
                continue
            # increment class_id. 0 is invalid
            class_id += 1
            
            idx = region * (reg_items * 6) + (region_fill[region]) * 6
            vec[idx] = 1.0
            vec[idx + 1] = class_id / class_num
            vec[idx + 2] = bbox[0]/img_width
            vec[idx + 3] = bbox[1]/img_height
            vec[idx + 4] = bbox[2]/img_width
            vec[idx + 5] = bbox[3]/img_height  
            #print(f"Assigned bbox {bbox} with class {class_id} to region {region}")
            region_fill[region] += 1

    return vec

def get_output_size(regions, reg_items=1, item_size=1, class_num = 1):
    # e.g. 6 items per region: probability, class_id, x_min, y_min, x_max, y_max
    return len(regions) * reg_items * item_size * class_num

def decode_label_vector(vec, cells, regions, reg_items=1, item_size=1, class_num = 1, mode="region"):
    items = []
    if mode == "region":
        num_regions = len(regions)
        for r in range(num_regions):
            for c in range(class_num):
                idx = r * class_num + c
                if vec[idx] < .5:
                    continue
                region = regions[r]
                start = cells[region[0]]
                end = cells[region[1]]
                x_min = start[0]
                y_min = start[1]
                x_max = end[2]
                y_max = end[3]
                items.append((float(vec[idx]), c, float(x_min), float(y_min), float(x_max), float(y_max)))                  

    elif mode == "yolo":
        img_width = cells[-1][2] + 1
        img_height = cells[-1][3] + 1
        itemNum = len(vec) // item_size
        if len(vec) % item_size != 0:
            raise ValueError(f"Invalid vector length {len(vec)} for item size {item_size}")
        for i in range(itemNum):
            idx = i * item_size
            if vec[idx] < .2:
                continue
            class_id = round(vec[idx + 1] * 5) - 1
            if class_id < 1:
                continue
            x_min = int(vec[idx + 2] * img_width)
            y_min = int(vec[idx + 3] * img_height)
            x_max = int(vec[idx + 4] * img_width)
            y_max = int(vec[idx + 5] * img_height)
            items.append((float(vec[idx]),int(class_id), float(x_min), float(y_min), float(x_max), float(y_max)))

    return items

