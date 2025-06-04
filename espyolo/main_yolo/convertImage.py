from PIL import Image
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <image path>")
    sys.exit()

img = Image.open(sys.argv[1]).resize((160, 160)).convert("RGB")
arr = np.asarray(img, dtype=np.uint8)
arr.tofile("robot_image.rgb")  # 160×160×3 bytes = 76,800 bytes

os.system("xxd -i robot_image.rgb > temp.inc")
with open("temp.inc") as f:
    data = f.read()

# select between brackets
raw = data.split("{")[1].split("}")[0].strip()
with open("robot_image.inc","w") as f:
    f.write(raw)


