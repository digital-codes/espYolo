import numpy as np


def define_cells(image_size=(176, 144), grid=(4, 3)):
    cells = []
    # 1x1 regions (3x3 grid)
    cell_size = (image_size[0] // grid[0], image_size[1] // grid[1])
    for row in range(grid[1]):
        for col in range(grid[0]):
            x_min = col * cell_size[0]
            y_min = row * cell_size[1]
            x_max = (
                (col + 1) * cell_size[0] - 1 if col < grid[0] - 1 else image_size[0] - 1
            )  # margin
            y_max = (
                (row + 1) * cell_size[1] - 1 if row < grid[1] - 1 else image_size[1] - 1
            )  # margin
            cells.append((x_min, y_min, x_max, y_max))
    return cells


def define_regions(cells, grid=(4, 3), square=True):
    regions = []

    if square:
        # add all cells. cells are ordered row wise
        for c in cells:
            regions.append(
                (cells.index(c), cells.index(c))
            )  # (start_cell, end_cell) is the same for single cell
        # enlarge in steps of 1
        for s in range(1, min(grid)):
            for r in range(0, grid[1] - s):
                for c in range(0, grid[0] - s):
                    start_cell = c + r * grid[0]
                    end_cell = start_cell + (grid[0] + 1) * s
                    regions.append(
                        (start_cell, end_cell)
                    )  # (start_cell, end_cell) is the same for single cell
        # add full size
        if grid[0] != grid[1]:
            regions.append((0, len(cells) - 1))
    else:
        # enlarge in x and y
        for height in range(1, grid[1] + 1):
            for width in range(1, grid[0] + 1):
                for y in range(0, grid[1] - height + 1):
                    for x in range(0, grid[0] - width + 1):
                        # print(f"Adding region {x},{y},{width}x{height} to regions")
                        start_cell = x + y * grid[0]
                        end_cell = start_cell + width - 1 + (height - 1) * grid[0]
                        # print("Start/End :", start_cell, end_cell)
                        regions.append((start_cell, end_cell))

    return regions


def find_cell(cells, x, y):
    """
    Find the cell index for the given coordinates (x, y).
    """
    for idx, (x_min, y_min, x_max, y_max) in enumerate(cells):
        if x_min <= int(x) <= x_max and y_min <= int(y) <= y_max:
            return idx
    print(f"[WARN] No cell found for coordinates ({x}, {y})")
    return None


def find_region(regions, start_cell, end_cell, grid=(4, 3)):
    """
    Find the region index for the given start and end cell indices.
    """
    for i, (start, end) in enumerate(regions):
        if start == start_cell and end == end_cell:
            return i

    s_row = start_cell // grid[0]
    s_col = start_cell % grid[0]
    e_row = end_cell // grid[0]
    e_col = end_cell % grid[0]
    # adjust the start and end cells to ensure they are within the square region bounds
    delta = max(e_row - s_row, e_col - s_col)
    #print(f"delta: {delta} for cells ({start_cell}, {end_cell}, {s_row}, {s_col}, {e_row}, {e_col})" )
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
        return (
            len(regions) - 1
        )  # return the full size region if no other region is found
    else:
        return None


def map_bbox(cells, regions, grid, bbox):
    if bbox[2] >= cells[grid[0] - 1][2]:
        bbox[2] = cells[grid[0] - 1][2] - 1  # adjust right edge to avoid out of bounds
    if bbox[3] >= cells[-1][3]:
        bbox[3] = cells[-1][3] - 1  # adjust bottom edge to avoid out of bounds
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    wadjust = width // 20
    hadjust = height // 20
    # Adjust bbox to avoid edge cases
    start_cell = find_cell(cells, bbox[0] + wadjust, bbox[1] + hadjust)
    end_cell = find_cell(cells, bbox[2] - wadjust, bbox[3] - hadjust)
    if (
        start_cell is None
        or start_cell >= len(cells)
        or end_cell is None
        or end_cell >= len(cells)
    ):
        print(
            f"[WARN] Invalid bbox {bbox} with start_cell {start_cell} or end_cell {end_cell}"
        )
        return None
    region = find_region(regions, start_cell, end_cell, grid)
    return region


def get_item_size(mode="region"):
    # 6 items per class: probability, x_min, y_min, x_max, y_max, mult (if more than 1 occurance)
    return 1 if mode == "region" else 6


def get_output_size(regions, class_num=1, mode="region"):
    # 6 items per class: probability, x_min, y_min, x_max, y_max, mult (if more than 1 occurance)
    return len(regions) * class_num * get_item_size(mode)



def create_label_vector(
    cells, regions, grid, bboxes, class_ids, class_num, mode="region"
):
    num_regions = len(regions)
    # yolo item size: x_min, y_min, x_max, y_max, mult
    item_size = get_item_size(mode)
    vec = np.zeros(class_num * num_regions * item_size, dtype=np.float32)

    if mode == "region":
        # all zero
        set_classes = np.zeros(class_num * num_regions, dtype=np.int8)
        for bbox, class_id in zip(bboxes, class_ids):
            region = map_bbox(cells, regions, grid, bbox)
            if region is None:
                print(f"[WARN] No region found for bbox {bbox} with class {class_id}")
                continue
            if set_classes[class_id * region] > 0:
                # first entry per class is good. input sorted by distance
                continue
            if class_id < 0 or class_id >= class_num:
                print(f"[WARN] Invalid class_id {class_id} for bbox {bbox}")
                continue
            idx = class_id + region * class_num
            vec[idx] = 1.0
            set_classes[class_id * region] = 1

    elif mode == "yolo":
        # use bbox instead dof region
        img_width = cells[-1][2] + 1
        img_height = cells[-1][3] + 1
        # init region items
        region_fill = np.zeros((num_regions * class_num), dtype=np.int32)

        for bbox, class_id in zip(bboxes, class_ids):

            region = map_bbox(cells, regions, grid, bbox)
            # check region
            if region is None:
                print(f"[WARN] No region found for bbox {bbox} with class {class_id}")
                continue
            # check class
            if class_id < 0 or class_id >= class_num:
                print(f"[WARN] Invalid class_id {class_id} for bbox {bbox}")
                continue
            # find index
            idx = (region * class_num + class_id) * item_size
            # class already present in region?
            if region_fill[region * class_id] >= 0:
                #print(f"[WARN] Class {class_id} already present in Region {region}")
                # update mult indicator
                vec[idx + 5] = 1
                continue
            else:
                region_fill[region * class_id] = 1
                
            # class probability
            vec[idx + 0] = 1
            # set bbox
            vec[idx + 1] = bbox[0] / img_width
            vec[idx + 2] = bbox[1] / img_height
            vec[idx + 3] = bbox[2] / img_width
            vec[idx + 4] = bbox[3] / img_height
            # print(f"Assigned bbox {bbox} with class {class_id} to region {region}")

    return vec


def decode_label_vector(vec, cells, regions, class_num=1, mode="region"):
    items = []
    item_size = get_item_size(mode)
    img_width = cells[-1][2] + 1
    img_height = cells[-1][3] + 1
    num_regions = len(regions)
    for r in range(num_regions):
        for c in range(class_num):
            idx = (r * class_num + c) * item_size
            if vec[idx] < 0.5:
                continue
            if mode == "region":
                region = regions[r]
                start = cells[region[0]]
                end = cells[region[1]]
                x_min = start[0]
                y_min = start[1]
                x_max = end[2]
                y_max = end[3]
                items.append(
                    (
                        float(vec[idx]),
                        c,
                        float(x_min),
                        float(y_min),
                        float(x_max),
                        float(y_max),
                    )
                )

            elif mode == "yolo":
                items.append(
                    (
                        float(vec[idx]),
                        c,
                        float(vec[idx + 1]) * img_width,
                        float(vec[idx + 2]) * img_height,
                        float(vec[idx + 1]+vec[idx + 3]) * img_width,
                        float(vec[idx + 2]+vec[idx + 4]) * img_height,
                        float(vec[idx + 5]),
                    )
                )

    return items
