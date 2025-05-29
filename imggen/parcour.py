import os
import numpy as np
import json
from PIL import Image
from PIL import ImageDraw

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
    Pigment,
    Texture,
    ImageMap,
)

# A map_type 0 gives the default planar mapping.
# A map_type 1 gives a spherical mapping (maps the image onto a sphere).
# With map_type 2 you get a cylindrical mapping (maps the image onto a cylinder).
# Finally map_type 5 is a torus or donut shaped mapping (maps the image onto a torus). 
# (Note that in order for the image to be aligned properly, either the object 
# has to be located at the origin when applying the pigment or the pigment 
# has to be transformed to align with the object. 
# It is generally easiest to create the object at the origin, apply the texture, 
# then move it to wherever you want it.) 

def color(rgb):
    return Pigment("color", rgb)


# Output directory for frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Oval path parameters
a, b = 0.25, 0.25  # 50 cm elliptical path

# Robot dims
rx, ry, rz = 0.05, 0.06, 0.08  # Robot dimensions

camera_fov = 43.6  # horizontal FOV ~ matching 2.32 mm lens on ~1 mm sensor

object_coords = [
    {
        "class": "box",
        "type": "box",
        "pos0": [0.2, 0, 0.2],
        "pos1": [0.24, 0.04, 0.24],
        "color": [1, 0.6, 0.5],
    },  # Box
    {
        "class": "person",
        "type": "cone",
        "pos0": [0.1, 0, 0.1],
        "r0": 0.03,
        "pos1": [0.1, 0.05, 0.1],
        "r1": 0.01,
        "color": [1, 0.8, 0],
    },  # Cone
    {
        "class": "ball",
        "type": "sphere",
        "pos0": [-0.2, 0.04, 0.25],
        "r0": 0.04,
        "color": [0.4, 0.6, 1],
    },  # Sphere
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.5, 0, -0.5],
        "pos1": [-0.5, 0.02, -0.25],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.5, 0, -0.25],
        "pos1": [-0.5, 0.02, 0],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.5, 0, 0],
        "pos1": [-0.5, 0.02, 0.25],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.5, 0, 0.25],
        "pos1": [-0.5, 0.02, 0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.5, 0, 0.5],
        "pos1": [-0.25, 0.02, 0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.25, 0, 0.5],
        "pos1": [0, 0.02, 0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0, 0, 0.5],
        "pos1": [0.25, 0.02, 0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.25, 0, 0.5],
        "pos1": [0.5, 0.02, 0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.5, 0, 0.5],
        "pos1": [0.5, 0.02, 0.25],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.5, 0, 0.25],
        "pos1": [0.5, 0.02, 0],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.5, 0, 0],
        "pos1": [0.5, 0.02, -0.25],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.5, 0, -0.25],
        "pos1": [0.5, 0.02, -0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.5, 0, -0.5],
        "pos1": [0.25, 0.02, -0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0.25, 0, -0.5],
        "pos1": [0, 0.02, -0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [0, 0, -0.5],
        "pos1": [-0.25, 0.02, -0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
    {
        "class": "fence",
        "type": "box",
        "pos0": [-0.25, 0, -0.5],
        "pos1": [-0.5, 0.02, -0.5],
        "color": [1, 0.1, 0.1],
    },  # fence
]


def oval_track_segments(
    radius=0.25, width=0.02, gap = 0 , height=0.001, segments=60
):
    objects = []
    length = 2 * np.pi * radius / segments - gap  # Length of each segment
    for i in range(segments):
        theta1 = 2 * np.pi * i / segments
        # Outer segment
        x1, z1 = radius * np.cos(theta1), radius * np.sin(theta1)
        w = length * 1.1
        l = length * 1.1
        objects.append(
            {
                "class": "track",
                "type": "box",
                "pos0": [x1 - w / 2, height, z1 - l / 2],
                "pos1": [x1 + w / 2, height, z1 + l / 2],
                "color": [1, 0.1, 0.1],
                "rotate": [0, 0, 0],
            },
        )
    return objects


trackLines = oval_track_segments()

for t in trackLines:
    object_coords.append(
        {
            "class": "track",
            "type": "box",
            "rot": t.get("rotate", [0, 0, 0]),
            "pos0": t["pos0"],
            "pos1": t["pos1"],
            "color": [1, 1, 1],
        }
    )


objects = []
texture1 = Texture(Pigment(ImageMap('jpeg', '"textures/img1.jpg"','map_type', 0)))
texture2 = Texture(Pigment(ImageMap('jpeg', '"textures/img1.jpg"','map_type', 1)))
texture3 = Texture(Pigment(ImageMap('jpeg', '"textures/img2.jpg"','map_type', 2)))
texture4 = Texture(Pigment(ImageMap('jpeg', '"textures/img2.jpg"','map_type', 1)))


for obj in object_coords:
    if obj["type"] == "box":
        objects.append(Box(obj["pos0"], obj["pos1"], color(obj["color"])))
    elif obj["type"] == "cone":
        objects.append(
            Cone(obj["pos0"], obj["r0"], obj["pos1"], obj["r1"], color(obj["color"]))
        )
    elif obj["type"] == "sphere":
        #objects.append(Sphere(obj["pos0"], obj["r0"], color(obj["color"])))
        objects.append(Sphere(obj["pos0"], obj["r0"], texture2))


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


def robot_position(t, duration):
    angle = 2 * np.pi * (1.5 * t / duration)
    return [a * np.cos(angle), 0, b * np.sin(angle), np.degrees(angle)]


def cam_pos(x, y, z, ry, rz, h=0.02):
    cx = 0
    cy = ry  #  + .02
    cz = rz
    return (
        [x - cx - 0.01, y + cy + h / 4, z + cz - 0.01],
        [x + cx + 0.01, y + cy + 3 * h / 4, z + cz + 0.01],
        [x + cx, y + cy + 5 * h / 4, z + cz],  # make sure camera is above box
    )


def robot_union(x, z, rx=0.03, ry=0.025, rz=0.05, rot=0):
    cam0 = cam_pos(0, 0, 0, ry, 0)[0]
    cam1 = cam_pos(0, 0, 0, ry, 0)[1]
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
            # cam_pos(0, 0, 0, ry*.9, 0)[0],
            # cam_pos(0, 0, 0, ry*.9, 0)[1],
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


def project_point_(point, cam_pos, look_at, fov_deg, img_width, img_height):
    """
    Projects a 3D world-space point onto 2D image space using pinhole projection.
    Returns 2D screen coordinates and camera-space coordinates (or None if not visible).
    """
    right, up, forward = get_camera_basis(cam_pos, look_at)
    # print("Camera basis vectors:", right, up, forward)

    # Build rotation matrix: columns = right, up, -forward
    R = np.stack([right, up, forward], axis=1)

    # Transform point into camera space
    p_world = np.array(point) - np.array(cam_pos)
    p_cam = np.dot(R.T, p_world)

    # print("Point in camera space:", p_cam)
    # Reject points behind the camera
    if p_cam[2] <= 0:
        return None, p_cam

    # Perspective projection
    fov_rad = np.radians(fov_deg)
    aspect = img_width / img_height

    x_ndc = p_cam[0] / (p_cam[2] * np.tan(fov_rad / 2))
    y_ndc = p_cam[1] / (p_cam[2] * np.tan(fov_rad / 2) / aspect)

    x_screen = int((x_ndc + 1) * img_width / 2)
    y_screen = int((1 - y_ndc) * img_height / 2)

    if not (0 <= x_screen < img_width and 0 <= y_screen < img_height):
        return None, p_cam  # outside image frame

    return (x_screen, y_screen), p_cam


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

    for c in corners:
        _, p_cam = project_point(c, cam_pos, look_at, fov_deg, img_width, img_height)
        if p_cam[2] > 0:
            x_ndc = p_cam[0] / (p_cam[2] * np.tan(fov_rad / 2))
            y_ndc = p_cam[1] / (p_cam[2] * np.tan(fov_rad / 2) / aspect)
            x_screen = int((x_ndc + 1) * img_width / 2)
            y_screen = int((1 - y_ndc) * img_height / 2)
            projected_points.append((x_screen, y_screen))

    if not projected_points:
        return None, None  # All corners behind the camera

    xs, ys = zip(*projected_points)
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad

    # Clamp to screen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    if x_min >= x_max or y_min >= y_max:
        return None, None  # Outside screen entirely

    cutoff_x = img_width // 40
    cutoff_y = img_height // 30
    if x_max - x_min < cutoff_x or y_max - y_min < cutoff_y:
        return None, None

    # return x,y,w,h
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)], p_cam


def lookat_point(pos):
    """
    Returns a camera object that looks at a specified point.
    cam_pos: camera position in world
    """
    cam_rot = 4  # optional speedup
    angle = pos[3] * cam_rot  # angle in degrees
    cam_dz = np.cos(np.radians(-angle))
    cam_dx = np.sin(np.radians(-angle))
    x = pos[0] + cam_dx * 5 * rz
    z = pos[2] + cam_dz * 5 * rz
    return [x, 0, z]  # Look at point in front of the robot


def create_scene(t, duration, view="robot"):
    pos = robot_position(t, duration)
    camera_pos = cam_pos(pos[0], 0, pos[2], ry, 0)[2]
    look_at = lookat_point(pos)  # Look at point in front of the robot
    angle = pos[3]  # angle in degrees

    # print("Camera position:", camera_pos)
    # def cam_pos(x,y,z,ry,rz):

    if view == "bird":
        camera = birds_eye_camera
    elif view == "robot":
        camera = Camera("location", camera_pos, "look_at", look_at, "angle", camera_fov)
    elif view == "side":
        camera = side_camera

    #track = oval_track_segments()

    pointer = Cylinder(
        [camera_pos[0], camera_pos[1] + 0.005, camera_pos[2]],  # camera_pos,
        look_at,
        0.001,  # Thin line
        color([1, 0, 0]),  # Red color for visibility
    )
    antenna = Cylinder(
        camera_pos,
        [camera_pos[0], camera_pos[1] + 0.05, camera_pos[2]],
        0.005,  # Thin antenna
        color([1, 1, 0]),  # Yellow color for visibility
    )

    # floor = Box([-0.5, -0.01, -0.5], [0.5, 0, 0.5], color([0.9, 0.9, 0.0])),
    floor = Box([-0.5, -0.01, -0.5], [0.5, 0, 0.5], texture4)


    return Scene(
        camera,
        [
            LightSource([0, 10, 0], "color", [1.0, 1.0, 1.0], "shadowless"),
            LightSource(
                [0.7, 0.4, 0.7],
                "color",
                [0.5, 0.5, 0.5],
                "spotlight",
                "radius",
                40,
                "point_at",
                [0, 0, 0],
            ),
            # LightSource([2, 4, -3], 'color', [1.5, 1.5, 1.5]),
            # LightSource(
            #     torch_pos,
            #     'color', [1, 1, 5],
            #     'spotlight',
            #     'point_at', look_at,
            #     'radius', 20,           # beam width
            #     'falloff', 30,          # soft edge
            #     'tightness', 10,        # intensity at center
            #     'shadowless'
            # ),
            # robot
            robot_union(pos[0], pos[2], rx, ry, rz, angle),
            # pointer,
            # antenna,
            ## body
            # Box([pos[0] - rx/2, 0, pos[2] - rz/2], [pos[0] + rx/2, ry, pos[2] + rz/2], color([0.2, 0.2, 0.2])),
            # camera mount
            # Box([pos[0] - rx/4, ry + .01, pos[2] - rz/4], [pos[0] + rx/4, ry + 0.05, pos[2] + rz/4], color([0.3, 0.3, 0.3])),
            ## wheels
            ##Cylinder([pos[0] + 0.025, 0.005, pos[2] - 0.05], [pos[0] + 0.025, 0.005, pos[2] - 0.07], 0.01, color([0.05, 0.05, 0.05])),
            ## Cylinder([pos[0] - 0.025, 0.005, pos[2] - 0.05], [pos[0] - 0.025, 0.005, pos[2] - 0.07], 0.01, color([0.05, 0.05, 0.05])),
            # Arena floor
            floor,
            # Oval track (50 cm diameter, 2 cm width)
            #*track,
            # objects
            *objects,
            # Box([0.2, 0, 0.2], [0.25, 0.05, 0.25], color([1, 0.6, 0.5])),
            # Cone([0.1, 0, 0.1], 0.03, [0.1, 0.08, 0.1], 0, color([1, 0.8, 0])),
            # Sphere([-0.2, 0.04, 0.25], 0.04, color([0.4, 0.6, 1])),
            # bg
            Background("color", [1, 10, 1]),
        ],
    )


# Animation parameters
duration = 4.0
fps = 15
frames = int(duration * fps)

img_width = 600
img_height = 450

for i in range(frames):
    print("\n\nRendering frame", i, "of", frames)
    t = i / fps
    for view in ["bird","robot","side"]:  # , "bird", "side"]:
        scene = create_scene(t, duration, view)

        scene.render(
            os.path.join(output_dir, f"{view}_{i:03d}.png"),
            width=img_width,
            height=img_height,
            quality=9,
            antialiasing=0.01,
        )

    # project_point(point, cam_pos, look_at, fov_deg, img_width, img_height)
    # get coordinates
    pos = robot_position(t, duration)
    camera_pos = cam_pos(pos[0], 0, pos[2], ry, 0)[2]
    look_at = lookat_point(pos)  # Look at point in front of the robot
    angle = pos[3]  # angle in degrees

    # print("Robot position at frame", i, ":", pos)
    # print("Camera position:", camera_pos)
    # print("Look at position:", look_at)

    # bounding boxes
    # Project objects onto the camera view

    visible_obj = []
    for obj in object_coords:
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
            size = [obj["r0"] * 2, (obj["pos1"][1] - obj["pos0"][1]), obj["r0"] * 2]
        elif obj["type"] == "sphere":
            # For sphere, use the center position
            pnt = obj["pos0"]  # Sphere position is already the center
            size = [obj["r0"] * 2 for i in range(3)]
        # pnt = [obj["pos1"][i] - (obj["pos1"][i] - obj["pos0"][i]) / 2 for i in range(3)]
        screen_coords, rel_pos = project_point(
            pnt, camera_pos, look_at, camera_fov, 600, 450
        )
        if screen_coords:
            #print(
            #    f"Object {obj['type']}, {pnt} at frame {i:03d} on screen at:",
            #    screen_coords,
            #    rel_pos,
            #)
            bb, dist = estimate_bounding_box(
                pnt, size, camera_pos, look_at, camera_fov, img_width, img_height
            )
            if bb is not None:
                # print(f"Object {obj['type']} bounding box at frame {i:03d}:", bb,dist)
                visible_obj.append(
                    {
                        "class": obj["class"],
                        "type": obj["type"],
                        "center": pnt,
                        "coords:": screen_coords,
                        "size": size,
                        "bounding_box": bb,
                        "distance": dist,
                    }
                )
        # else:
        #    print(f"Object {obj['type']}, {pnt} at frame {i:03d} not visible")

    # Open the rendered image
    img_path = os.path.join(output_dir, f"robot_{i:03d}.png")
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
            f"Processing object {obj['class']} with bounding box {bb} at frame {i:03d}"
        )
        for earlier_obj in visible_obj[:idx]:
            if earlier_obj["bounding_box"] is None:
                continue
            earlier_bb = earlier_obj["bounding_box"]

            print(
                f"Earlier object {earlier_obj['class']} bounding box: {earlier_bb}"
            )
            print(f"Current object {obj['class']} bounding box: {bb}")

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

            # Check overlap. if earlier object is inside with one or both dimensions, continue
            # otherwise, we would need to split the bounding box
            if (
                ((earlier_bb[0] >= bb[0])
                and (earlier_bb[0] + earlier_bb[1]) <= (bb[1] + bb[3]))
                or ((earlier_bb[1] >= bb[1]) and (earlier_bb[1] + earlier_bb[3] <= bb[1] + bb[3]))
            ):
                # earlier object is completely inside current object, continue
                print(
                    f"Earlier object {earlier_obj['class']} inside current object {obj['class']}, skipping overlap check"
                )
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
            if bb[0] < earlier_bb[0]:
                bb[2] = earlier_bb[0] - bb[0] # cut right
            else:
                temp = bb[0] + bb[2]  # right border
                bb[0] = earlier_bb[0] + earlier_bb[2] # cut left
                bb[2] = temp - (earlier_bb[1] + earlier_bb[3])

            if bb[1] < earlier_bb[1]:
                bb[3] = earlier_bb[1] - bb[1] # cut bottom
            else:
                temp = bb[1] + bb[3]
                bb[1] = earlier_bb[1] + earlier_bb[3]  # cut top
                bb[3] = temp - bb[1]


            print(
                f"new width: {bb[2]}, new height: {bb[3]}, old width,height: {bb_width}, {bb_height}"
            )
            
            # Drop item if remaining width or height is smaller than 40% of original
            cutoff = 0.3
            if bb[2] < cutoff * bb_width or bb[3] < cutoff * bb_height:
                print(
                    f"Object {obj['class']} bounding box too small after overlap check, dropping"
                )
                obj["bounding_box"] = None

    for obj in visible_obj:
        if obj["bounding_box"] is None:
            print(f"Object {obj['class']} bounding box not visible")
        else:
            bb = obj["bounding_box"]
            dist = obj["distance"]
            bbox = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
            print(f"Object {obj['class']} bounding box at frame {i:03d}:", bbox)
            draw.rectangle(
                bbox,
                outline="red",
                width=2,
            )
            draw.text((bbox[2], bbox[3]), obj["class"], fill="red")
            annotations.append(
                {
                    "class": obj["class"],
                    "type": obj["type"],
                    "bounding_box": bbox,
                    "distance": dist.tolist() if isinstance(dist, np.ndarray) else dist,
                }
            )

    with open(os.path.join(output_dir, f"visible_objects_{i:03d}.json"), "w") as f:
        json.dump(annotations, f)

    # Save the annotated image
    annotated_img_path = os.path.join(output_dir, f"robot_ann{i:03d}.png")
    img.save(annotated_img_path)


print("Rendering complete. Frames saved to:", output_dir)
