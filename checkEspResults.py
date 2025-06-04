import layoutUtils as lu
import json
import sys 

if len(sys.argv) < 2:
    print("Usage: python checkEspResults.py <path_to_json_file>")
    sys.exit()


with open(sys.argv[1]) as f:
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
    segment = cls * len(regions) + r
    print("BBox:", bb, "Class:",cls, "Cells:", c0, c1, "Region:", r, "VectorItem:", r * numClasses + cls, "Segment:", segment)
    results.append((bb, cls, c0, c1, r, segment))
    #print("Segment:", segment + r)
    #for c in range(numClasses):
    #   print(f"Segment2 ({c}):", c * len(regions) + r)
        

#esp = [6,43,109,707,723,725,778,805]
esp = [5,29,48,99,236,238,242,258,261,263,305,309,310,317,324,325,327,694,735,758]
for e in esp:
    print("ESP:", e,  "Region:", e % len(regions), "Class:", e // len(regions))
    