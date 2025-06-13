
""" Defs for fomo train and predict"""
OBJ_SIZES = [2.5,6.0]  # Sizes in meters, -1.0 for empty class
NUM_SIZES = len(OBJ_SIZES)  # Number of sizes
NUM_TYPES = 5
NUM_CLASSES = NUM_TYPES * (NUM_SIZES + 1) # 5 classes, 3 sizes

