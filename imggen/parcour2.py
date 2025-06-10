import os
import numpy as np
import json
from PIL import Image
from PIL import ImageDraw
import sys
import random

from vapory import (
    Scene,
    Camera,
    LightSource,
    Background,
    Box,
    Cone,
    Sphere,
    Cylinder,
    Union,
    Plane,
    Pigment,
    Finish,
    Normal,
    Texture,
    ImageMap,
    ColorMap,
)

# Output directory for frames
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
else:
    output_dir = "output_frames"

os.makedirs(output_dir, exist_ok=True)

included = ["textures.inc"]
# see https://www.f-lohmueller.de/pov_tut/tex/tex_160d.htm

textureNames = [
    "Red_Marble",
    "White_Marble",
    "Blood_Marble",
    "Blue_Agate",
    "Cherry_Wood",
    "Pine_Wood",
    "Dark_Wood",
    "Cork",
    "Ruby_Glass",
    "Dark_Green_Glass",
]

# A map_type 0 gives the default planar mapping.
# A map_type 1 gives a spherical mapping (maps the image onto a sphere).
# With map_type 2 you get a cylindrical mapping (maps the image onto a cylinder).
# Finally map_type 5 is a torus or donut shaped mapping (maps the image onto a torus).
# (Note that in order for the image to be aligned properly, either the object
# has to be located at the origin when applying the pigment or the pigment
# has to be transformed to align with the object.
# It is generally easiest to create the object at the origin, apply the texture,
# then move it to wherever you want it.)

green_shades = [
    [0.0, 0.3, 0.0],  # Dark green
    [0.1, 0.4, 0.1],  # Forest green
    [0.2, 0.5, 0.2],  # Medium green
    [0.3, 0.6, 0.3],  # Lime green
    [0.4, 0.7, 0.4],  # Bright green
    [0.5, 0.8, 0.5],  # Light green
    [0.6, 0.9, 0.6],  # Pale green
    [0.7, 1.0, 0.7],  # Mint green
    [0.8, 1.0, 0.8],  # Soft green
    [0.9, 1.0, 0.9],  # Pastel green
]
blue_shades = [
    [0.0, 0.0, 0.3],  # Dark blue
    [0.1, 0.1, 0.4],  # Navy blue
    [0.2, 0.2, 0.5],  # Medium blue
    [0.3, 0.3, 0.6],  # Royal blue
    [0.4, 0.4, 0.7],  # Bright blue
    [0.5, 0.5, 0.8],  # Light blue
    [0.6, 0.6, 0.9],  # Sky blue
    [0.7, 0.7, 1.0],  # Pale blue
    [0.8, 0.8, 1.0],  # Soft blue
    [0.9, 0.9, 1.0],  # Pastel blue
]

red_shades = [
    [0.3, 0.0, 0.0],  # Dark red
    [0.4, 0.1, 0.1],  # Crimson red
    [0.5, 0.2, 0.2],  # Medium red
    [0.6, 0.3, 0.3],  # Bright red
    [0.7, 0.4, 0.4],  # Scarlet red
    [0.8, 0.5, 0.5],  # Light red
    [0.9, 0.6, 0.6],  # Coral red
    [1.0, 0.7, 0.7],  # Pale red
    [1.0, 0.8, 0.8],  # Soft red
    [1.0, 0.9, 0.9],  # Pastel red
]

light_shades = [
    [0.8, 0.8, 0.8],  # Light gray
    [0.7, 0.7, 0.7],  # Medium-light gray
    [0.81, 0.83, 0.82],  # Light gray variant
    [0.72, 0.73, 0.71],  # Medium-light gray variant
    [0.6, 0.6, 0.6],  # Medium gray
    [0.5, 0.5, 0.5],  # Medium-dark gray
    [0.9, 0.9, 0.9],  # Near white
    [0.85, 0.85, 0.85],  # Soft light gray
    [0.75, 0.75, 0.75],  # Medium-light gray
    [1.0, 1.0, 1.0],  # Pure white
]

dark_gray_shades = [
    [0.1, 0.1, 0.1],  # Very dark gray
    [0.15, 0.15, 0.15],  # Dark gray
    [0.2, 0.2, 0.2],  # Medium-dark gray
    [0.25, 0.25, 0.25],  # Slightly lighter dark gray
    [0.3, 0.3, 0.3],  # Neutral gray
    [0.35, 0.35, 0.35],  # Medium gray
]

all_shades = [
    [r, g, b]
    for r in np.linspace(0.0, 1.0, 30)
    for g in np.linspace(0.0, 1.0, 30)
    for b in np.linspace(0.0, 1.0, 30)
]


def color(rgb):
    return Pigment("color", rgb)


# define object sizes
MIN_SIZE = 0.01  # Minimum size for objects
MAX_SIZE = 0.1  # Maximum size for objects

# texture files
tf = os.listdir("textures")
texFiles = [os.sep.join(["textures", f]) for f in tf if f.startswith("texture_")]


def createTexture(tx=None, color=None):
    if tx is not None:
        return Texture(tx)
    elif color is not None:
        return Texture(
            Pigment("color", color),
            Normal(
                "bumps",
                np.random.uniform(0.15, 0.35),
                "scale",
                np.random.uniform(0.05, 0.15),
            ),
            Finish(
                "ambient",
                np.random.uniform(0.2, 0.4),
                "diffuse",
                np.random.uniform(0.4, 0.6),
                "roughness",
                np.random.uniform(0.05, 0.2),
            ),
        )

    else:
        im = f'"{random.choice(texFiles)}"'
        return Texture(
            Pigment(
                ImageMap("jpeg", im, "interpolate", 2),
                "rotate",
                [90, 0, 0],
                "scale",
                [2, 2, 2],
                "translate",
                [-1, 0, -1],
            )
        )



def create_box(size, col=None):
    """
    Create a box object with specified position and color.
    """
    return Box(size,size, color(col))


# Oval path parameters
a, b = 0.25, 0.25  # 50 cm elliptical path

# Robot dims
rx, ry, rz = 0.05, 0.06, 0.08  # Robot dimensions
# camera height
cam_h = 0.02  # Camera height above robot body

camera_fov = 43.6  # horizontal FOV ~ matching 2.32 mm lens on ~1 mm sensor

def createObjects():
    numBoxes = random.randint(1, 5)
    numBalls = random.randint(1, 3)
    numPersons = random.randint(1, 4)
    numFences = random.randint(3, 6)
    numTracks = random.randint(2, 6)
    obj_coordinates = []
    for _ in range(numBoxes):
        pos0 = [np.random.uniform(-0.45, 0.45), 0, np.random.uniform(-0.45, 0.45)]
        pos1 = [pos0[0] + np.random.uniform(MIN_SIZE, MAX_SIZE), np.random.uniform(MIN_SIZE, MAX_SIZE), pos0[2] + np.random.uniform(MIN_SIZE, MAX_SIZE)]
        color = random.choice(all_shades)
        obj_coordinates.append({
            "class": "box",
            "type": "box",
            "pos0": pos0,
            "pos1": pos1,
            "color": color,
        })
    for _ in range(numFences):
        pos0 = [np.random.uniform(-0.45, 0.45), 0, np.random.uniform(-0.45, 0.45)]
        pos1 = [pos0[0] + np.random.uniform(.05, .08), 0.002, pos0[2] + np.random.uniform(.05, .08)]
        color = random.choice(red_shades)
        obj_coordinates.append({
            "class": "box",
            "type": "fence",
            "pos0": pos0,
            "pos1": pos1,
            "color": color,
        })
    for _ in range(numTracks):
        pos0 = [np.random.uniform(-0.35, 0.35), 0, np.random.uniform(-0.35, 0.35)]
        pos1 = [pos0[0] + np.random.uniform(.02, .06), 0.002, pos0[2] + np.random.uniform(.05, .08)]
        color = random.choice(light_shades)
        obj_coordinates.append({
            "class": "box",
            "type": "track",
            "pos0": pos0,
            "pos1": pos1,
            "color": color,
        })
        
    objects = []
    for obj in obj_coordinates:
        if obj["type"] == "box":
            if obj["class"] == "track" or obj["class"] == "fence":
                objects.append(Box(obj["pos0"], obj["pos1"], color(obj["color"])))
            else:
                objects.append(Box(obj["pos0"], obj["pos1"], createTexture(color = random.choice(random.choice([green_shades, blue_shades])))))
        elif obj["type"] == "cone":
            objects.append(
                Cone(obj["pos0"], obj["r0"], obj["pos1"], obj["r1"], createTexture(tx = random.choice(textureNames)))
            )
        elif obj["type"] == "sphere":
            objects.append(Sphere(obj["pos0"], obj["r0"], createTexture(color=random.choice(blue_shades))))
        

    return objects, obj_coordinates


birds_eye_camera = Camera(
    "location",
    [0, 1.5, 0],  # 1.5 m above the center
    "look_at",
    [0, 0, 0],  # look at the center of the arena
    "angle",
    60,  # wider FOV to capture full track
)

side_camera = Camera(
    "location",
    [0.7, 0.4, 0.7],  # 1.5 m above the center
    "look_at",
    [0, 0, 0],  # look at the center of the arena
    "angle",
    60,  # wider FOV to capture full track
)


def robot_position():
    angle = np.random.uniform(0, 2 * np.pi)
    x = np.random.uniform(-0.45, 0.45)
    y = np.random.uniform(0, 0.02)
    z = np.random.uniform(-0.45, 0.45)
    return [x, y, z, np.degrees(angle)]


def cam_pos(x, y, z, ry, rz, h=0.02):
    cx = 0
    cy = ry  #  + .02
    cz = rz
    return (
        [x - cx - 0.005, y + cy + h / 4, z + cz - 0.005],
        [x + cx + 0.005, y + cy + 3 * h / 4, z + cz + 0.005],
        [x + cx, y + cy + 5 * h / 4, z + cz],  # make sure camera is above box
    )


def robot_union(x, z, rx=0.03, ry=0.025, rz=0.05, rot=0):
    cam0 = cam_pos(0, 0, 0, ry, 0, cam_h)[0]
    cam1 = cam_pos(0, 0, 0, ry, 0, cam_h)[1]
    return Union(
        # Body box
        Box([-rx / 2, 0, 0], [rx / 2, ry, -rz], color([0.4, 0.4, 0.4])),
        # Wheels (rear left and right)
        Cylinder(
            [-0.01, 0.0, 0],
            [0.01, 0.0, 0],
            0.02,
            "rotate",
            [0, 0, 0],
            "translate",
            [-rx / 2 - 0.015, 0.02, 0],
            color([1, 1, 1]),
        ),
        Cylinder(
            [-0.01, 0.0, 0],
            [0.01, 0.0, 0],
            0.02,
            "rotate",
            [0, 0, 0],
            "translate",
            [rx / 2 + 0.015, 0.02, 0],
            color([1, 1, 1]),
        ),
        # Camera mount
        # Box(
        #     [-0.01, ry, -0.01],
        #     [+0.01, ry + 0.02, 0],
        #     color([1.2, 0.2, 0.2]),
        # ),
        # Camera box. camera sits on top!
        Box(
            [cam0[0], cam0[1], cam0[2]],
            [cam1[0], cam1[1], cam1[2]],
            # cam_pos(0, 0, 0, ry*.9, 0,cam_h)[0],
            # cam_pos(0, 0, 0, ry*.9, 0,cam_h)[1],
            color([0.1, 1.1, 0.1]),
        ),
        "rotate",
        [0, -rot, 0],
        "translate",
        [x, 0, z],
    )


def get_camera_basis(cam_pos, look_at):
    """
    Computes the camera's coordinate system (right, up, forward).
    Returns right, up, forward vectors for use in projection.
    """
    forward = np.array(look_at) - np.array(cam_pos)
    forward /= np.linalg.norm(forward)

    world_up = np.array([0, 1, 0])
    # Handle edge case where forward is parallel to world_up
    if np.allclose(forward, world_up) or np.allclose(forward, -world_up):
        world_up = np.array([0, 0, 1])

    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    return right, up, forward


def project_point(point, cam_pos, look_at, fov_deg, img_width, img_height):
    right, up, forward = get_camera_basis(cam_pos, look_at)
    R = np.stack([right, up, forward], axis=1)
    p_world = np.array(point) - np.array(cam_pos)
    p_cam = np.dot(R.T, p_world)

    if p_cam[2] <= 0:
        return None, p_cam

    fov_rad = np.radians(fov_deg)
    aspect = img_width / img_height

    x_ndc = p_cam[0] / (p_cam[2] * np.tan(fov_rad / 2))
    y_ndc = p_cam[1] / (p_cam[2] * np.tan(fov_rad / 2) / aspect)

    x_screen = int((x_ndc + 1) * img_width / 2)
    y_screen = int((1 - y_ndc) * img_height / 2)

    # don't do this here. clip later!
    # if not (0 <= x_screen < img_width and 0 <= y_screen < img_height):
    #    return None, p_cam

    return (x_screen, y_screen), p_cam


def estimate_bounding_box(
    center, size, cam_pos, look_at, fov_deg, img_width, img_height, pad=2
):
    dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2
    corners = [
        [center[0] + sx * dx, center[1] + sy * dy, center[2] + sz * dz]
        for sx in [-1, 1]
        for sy in [-1, 1]
        for sz in [-1, 1]
    ]

    fov_rad = np.radians(fov_deg)
    aspect = img_width / img_height
    projected_points = []
    drop_object = False

    for c in corners:
        _, p_cam = project_point(c, cam_pos, look_at, fov_deg, img_width, img_height)
        if p_cam[2] > 0:
            x_ndc = p_cam[0] / (p_cam[2] * np.tan(fov_rad / 2))
            y_ndc = p_cam[1] / (p_cam[2] * np.tan(fov_rad / 2) / aspect)
            x_screen = int((x_ndc + 1) * img_width / 2)
            y_screen = int((1 - y_ndc) * img_height / 2)
            projected_points.append((x_screen, y_screen))

    if not projected_points:
        return None, None, drop_object  # All corners behind the camera

    xs, ys = zip(*projected_points)
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad

    # inital area
    init_area = abs(x_max - x_min) * abs(y_max - y_min)

    # Clamp to screen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)
    # final_area
    final_area = abs(x_max - x_min) * abs(y_max - y_min)
    # ignore if fraction below .5
    if final_area < 0.5 * init_area:
        # print(f"Visible bounding box too small: initial area {init_area}, final area {final_area}")
        drop_object = True

    if x_min >= x_max or y_min >= y_max:
        return None, None, drop_object  # Outside screen entirely

    cutoff_x = img_width // 40
    cutoff_y = img_height // 30
    if x_max - x_min < cutoff_x or y_max - y_min < cutoff_y:
        return None, None, drop_object

    # return x,y,w,h
    return (
        [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
        p_cam,
        drop_object,
    )


def lookat_point(pos, cam_rot=1):
    """
    Returns a camera object that looks at a specified point.
    cam_pos: camera position in world
    """
    angle = pos[3] * cam_rot  # angle in degrees
    cam_dz = np.cos(np.radians(-angle))
    cam_dx = np.sin(np.radians(-angle))
    x = pos[0] + cam_dx * 5 * rz
    z = pos[2] + cam_dz * 5 * rz
    return [x, 0, z]  # Look at point in front of the robot


def create_scene(view="robot"):
    pos = robot_position()
    cam_h = np.random.uniform(0.02, 0.04)  # Random camera height
    camera_pos = cam_pos(pos[0], 0, pos[2], ry, 0, cam_h)[2]
    look_at = lookat_point(pos, 1)  # Look at point in front of the robot
    angle = pos[3]  # angle in degrees

    # print("Camera position:", camera_pos)
    # def cam_pos(x,y,z,ry,rz,cam_h):

    if view == "bird":
        camera = birds_eye_camera
    elif view == "robot":
        camera = Camera("location", camera_pos, "look_at", look_at, "angle", camera_fov)
    elif view == "side":
        camera = side_camera

    # Create plane with texture1
    plane = Plane([0, 1, 0], 0, createTexture(), "translate", [0, -0.01, 0])
    floor = Box([-0.5, -0.01, -0.5], [0.5, 0, 0.5], createTexture(color=(random.choice(dark_gray_shades))))

    objects, obj_coordinates = createObjects()

    globalLight = LightSource(
        [
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(9.5, 10.5),
            np.random.uniform(-0.1, 0.1),
        ],
        "color",
        [
            np.random.uniform(0.9, 1.1),
            np.random.uniform(0.9, 1.1),
            np.random.uniform(0.9, 1.1),
        ],
        "shadowless",
    )
    spotLight = LightSource(
        [
            np.random.uniform(0.6, 0.8),
            np.random.uniform(0.3, 0.5),
            np.random.uniform(0.6, 0.8),
        ],
        "color",
        [
            np.random.uniform(0.4, 0.6),
            np.random.uniform(0.4, 0.6),
            np.random.uniform(0.4, 0.6),
        ],
        "spotlight",
        "radius",
        40,
        "point_at",
        [0, 0, 0],
    )

    sc = Scene(
        camera,
        objects=[
            globalLight,
            spotLight,
            # robot
            robot_union(pos[0], pos[2], rx, ry, rz, angle),
            # background plane
            plane,
            # Arena floor
            floor,
            # objects
            *objects,
            Background("color", [1, 10, 1]),
        ],
        included=included,
    )
    return sc, pos, camera_pos, look_at, obj_coordinates


################################

img_width = 176  # 600
img_height = 144  # 450
frames = 10
unique_classes = {"person": 0, "box":1, "ball":2, "fence":3, "track":4}
class_map = {key: idx for idx, key in enumerate(unique_classes)}
print("Unique classes:", class_map)
with open(os.path.join(output_dir, "label_map.json"), "w") as f:
    json.dump(class_map, f, indent=4)

for i in range(frames):
    print("\n\nRendering frame", i, "of", frames)
    for view in ["robot", "bird", "side"]:  # , "bird", "side"]:
        scene, pos, camera_pos, look_at, obj_coordinates = create_scene(view)

        scene.render(
            os.path.join(output_dir, f"{view}_{i:04d}.png"),
            width=img_width,
            height=img_height,
            quality=9,
            antialiasing=0.01,
        )

    angle = pos[3]  # angle in degrees

    visible_obj = []
    img_name = os.path.join(f"robot_{i:04d}.png")

    for obj in obj_coordinates:
        # Calculate the center position of the bounding box
        if obj["type"] == "box":
            pnt = [
                obj["pos1"][i] - (obj["pos1"][i] - obj["pos0"][i]) / 2 for i in range(3)
            ]
            size = [(obj["pos1"][i] - obj["pos0"][i]) for i in range(3)]
        elif obj["type"] == "cone":
            pnt = [
                obj["pos0"][0],
                obj["pos1"][1] - (obj["pos1"][1] - obj["pos0"][1]) / 2,
                obj["pos0"][2],
            ]
            size = [obj["r0"] * 1.7, (obj["pos1"][1] - obj["pos0"][1]), obj["r0"] * 1.7]
        elif obj["type"] == "sphere":
            # For sphere, use the center position
            pnt = obj["pos0"]  # Sphere position is already the center
            size = [obj["r0"] * 1.7 for i in range(3)]
        # pnt = [obj["pos1"][i] - (obj["pos1"][i] - obj["pos0"][i]) / 2 for i in range(3)]
        screen_coords, rel_pos = project_point(
            pnt, camera_pos, look_at, camera_fov, img_width, img_height
        )
        if screen_coords:
            # print(
            #    f"Object {obj['type']}, {pnt} at frame {i:04d} on screen at:",
            #    screen_coords,
            #    rel_pos,
            # )
            bb, dist, drop = estimate_bounding_box(
                pnt, size, camera_pos, look_at, camera_fov, img_width, img_height
            )
            if bb is not None:
                # print(f"Object {obj['type']} bounding box at frame {i:04d}:", bb,dist)
                visible_obj.append(
                    {
                        "class": obj["class"],
                        "type": obj["type"],
                        "center": pnt,
                        "coords:": screen_coords,
                        "size": size,
                        "bounding_box": bb,
                        "distance": dist,
                        "drop": drop,
                    }
                )
        # else:
        #    print(f"Object {obj['type']}, {pnt} at frame {i:04d} not visible")

    # Open the rendered image
    img_path = os.path.join(output_dir, f"robot_{i:04d}.png")
    img = Image.open(img_path)
    img = img.convert("RGB")

    # Draw bounding boxes on the image

    draw = ImageDraw.Draw(img)
    annotations = []
    visible_obj.sort(
        key=lambda obj: (
            obj["distance"][2] if obj["distance"] is not None else float("inf")
        )
    )
    for idx, obj in enumerate(visible_obj):
        if obj["bounding_box"] is None:
            continue
        bb = obj["bounding_box"]
        bb_width = bb[2]
        bb_height = bb[3]
        print(
            f"Processing object {obj['class']} with bounding box {bb} at frame {i:04d}"
        )
        for earlier_obj in visible_obj[:idx]:
            if earlier_obj["bounding_box"] is None:
                continue
            earlier_bb = earlier_obj["bounding_box"]

            print(f"Earlier object {earlier_obj['class']} bounding box: {earlier_bb}")
            print(f"Current object {obj['class']} bounding box: {bb}")

            # try to skip if same class
            if earlier_obj["class"] == obj["class"]:
                print(f"Skipping overlap check for same class {earlier_obj['class']}")
                continue

            # check for disjunct
            if bb[0] + bb[2] <= earlier_bb[0]:
                # object completely left to earlier object
                continue
            if bb[1] + bb[3] <= earlier_bb[1]:
                # object completely above to earlier object
                continue
            if bb[0] >= earlier_bb[0] + earlier_bb[2]:
                # object completely right to earlier object
                continue
            if bb[1] >= earlier_bb[1] + earlier_bb[3]:
                # object completely below to earlier object
                continue

            # Check if the object is completely inside the earlier object
            if (
                bb[0] >= earlier_bb[0]
                and bb[1] >= earlier_bb[1]
                and bb[0] + bb[2] <= earlier_bb[0] + earlier_bb[2]
                and bb[1] + bb[3] <= earlier_bb[1] + earlier_bb[3]
            ):
                print(
                    f"Object {obj['class']} completely inside earlier object, dropping"
                )
                obj["bounding_box"] = None
                break

            print("Clipping bounding box for overlap with earlier object")

            # Calculate overlap area
            # if both dimensions overlap, clip both dimensions
            # if width completly inside earlier width, clip only height
            if (
                bb[0] >= earlier_bb[0]
                and bb[0] + bb[2] <= earlier_bb[0] + earlier_bb[2]
            ):
                print(
                    f"Object {obj['class']} width completely inside earlier object, clipping height only"
                )
            else:
                print(
                    f"Object {obj['class']} width overlaps with earlier object, clipping width"
                )
                if bb[0] < earlier_bb[0]:
                    bb[2] = earlier_bb[0] - bb[0]  # cut right
                else:
                    temp = bb[0] + bb[2]  # right border
                    bb[0] = earlier_bb[0] + earlier_bb[2]  # cut left
                    bb[2] = temp - bb[0]

            # if height completly inside earlier height, clip only width
            if (
                bb[1] >= earlier_bb[1]
                and bb[1] + bb[3] <= earlier_bb[1] + earlier_bb[3]
            ):
                print(
                    f"Object {obj['class']} height completely inside earlier object, clipping width only"
                )
            else:
                print(
                    f"Object {obj['class']} height overlaps with earlier object, clipping height"
                )
            if bb[1] < earlier_bb[1]:
                bb[3] = earlier_bb[1] - bb[1]  # cut bottom
            else:
                temp = bb[1] + bb[3]
                bb[1] = earlier_bb[1] + earlier_bb[3]  # cut top
                bb[3] = temp - bb[1]

            print(
                f"new width: {bb[2]}, new height: {bb[3]}, old width,height: {bb_width}, {bb_height}"
            )

            # Drop item if remaining width or height is smaller than 40% of original
            cutoff = 0.4
            if bb[2] < cutoff * bb_width or bb[3] < cutoff * bb_height:
                print(
                    f"Object {obj['class']} bounding box too small after overlap check, dropping"
                )
                obj["bounding_box"] = None

    for obj in visible_obj:
        if obj["bounding_box"] is None:
            print(f"Object {obj['class']} bounding box not visible")
        elif obj["drop"]:
            print(f"Object {obj['class']} bounding box dropped due to fractional area")
        else:
            bb = obj["bounding_box"]
            dist = obj["distance"]
            bbox = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
            print(f"Object {obj['class']} bounding box at frame {i:04d}:", bbox)
            draw.rectangle(
                bbox,
                outline="red",
                width=2,
            )
            draw.text((bbox[2], bbox[3]), obj["class"], fill="red")
            annotations.append(
                {
                    "label": class_map[obj["class"]],
                    "class": obj["class"],
                    "type": obj["type"],
                    "bounding_box": bbox,
                    "distance": dist.tolist() if isinstance(dist, np.ndarray) else dist,
                }
            )

    with open(os.path.join(output_dir, f"visible_objects_{i:04d}.json"), "w") as f:
        json.dump(annotations, f)

    labels = []
    bboxes = []
    zdistance = []
    for a in annotations:
        bboxes.append(a["bounding_box"])
        labels.append(a["label"])
        zdistance.append(a["distance"][2])

    with open(os.path.join(output_dir, f"robot_{i:04d}_labels.json"), "w") as f:
        json.dump(
            {
                "img": img_name,
                "bboxes": bboxes,
                "labels": labels,
                "zdistance": zdistance,
            },
            f,
            indent=4,
        )

    # Save the annotated image
    annotated_img_path = os.path.join(output_dir, f"robot_ann{i:04d}.png")
    img.save(annotated_img_path)


print("Rendering complete. Frames saved to:", output_dir)
