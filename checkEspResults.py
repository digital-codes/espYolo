import layoutUtils as lu
import json
with open("espyolo/main/robot_204_labels.json") as f:
    boxes = json.load(f)

grid = 5
imgSize = 160
cells = lu.define_cells(imgSize,grid)
regions = lu.define_regions(grid)
numClasses = 5

# predict code:
#for region_id in range(NUM_REGIONS):
#    index = class_id * NUM_REGIONS + region_id
#    score = pred_vec[index]


results = []
for i,bb in enumerate(boxes["bboxes"]):
    c0 = lu.find_cell(cells,bb[0],bb[1])
    c1 = lu.find_cell(cells,bb[2],bb[3])
    r = lu.find_region(regions,c0,c1)
    cls = boxes["labels"][i]
    print("BBox:", bb, "Class:",cls, "Cell:", c0, c1, "Region:", r, "VectorItem:", r * numClasses + cls)
    segment = cls * len(regions)
    print("Segment:", segment + r)
    #for c in range(numClasses):
    #   print(f"Segment2 ({c}):", c * len(regions) + r)
        

esp = [6,43,109,707,723,725,778,805]
for e in esp:
    print("ESP:", e, "Class:", e % numClasses, "Region:", e // numClasses)
for e in esp:
    print("ESP2:", e, "Region:", e % len(regions), "Class:", e // len(regions))
    