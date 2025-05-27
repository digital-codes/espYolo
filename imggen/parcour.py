import os
import numpy as np

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
)


def color(rgb):
    return Pigment("color", rgb)


# Output directory for frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Oval path parameters
a, b = 0.25, 0.25  # 50 cm elliptical path

# Robot dims
rx, ry, rz = 0.02, 0.05, 0.06  # Robot dimensions (6 cm x 3 cm x 1 cm)

camera_fov = 43.6  # horizontal FOV ~ matching 2.32 mm lens on ~1 mm sensor

object_coords = [
    {"type":"box","pos0":[0.2, 0, 0.2],"pos1":[.25,.05,.25]},  # Box
    {"type":"cone","pos0":[0.1, 0, 0.1],"r0":.03,"pos1":[0.1, 0.08, 0.1],"r1":0},  # Box
    {"type":"sphere","pos0":[-0.2, 0.04, 0.25],"r0":.04,"pos1":[.25,.05,.25]},  # Box
]


objects = []
for obj in object_coords:
    if obj["type"] == "box":
        objects.append(Box(obj["pos0"], obj["pos1"], color([1, 0.6, 0.5])))
    elif obj["type"] == "cone":
        objects.append(Cone(obj["pos0"], obj["r0"], obj["pos1"], obj["r1"], color([1, 0.8, 0])))
    elif obj["type"] == "sphere":
        objects.append(Sphere(obj["pos0"], obj["r0"], color([0.4, 0.6, 1])))


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
    [.7, .4, .7],  # 1.5 m above the center
    "look_at",
    [0, 0, 0],  # look at the center of the arena
    "angle",
    60,  # wider FOV to capture full track
)


def robot_position(t, duration):
    angle = 2 * np.pi * (1.5 * t / duration)
    return [a * np.cos(angle), 0, b * np.sin(angle)]


def cam_pos(x,y,z,ry,rz):
    cx = 0
    cy = ry + .02
    cz = rz  
    return (
        [x - cx - 0.01, y + cy - 0.005, z + cz - 0.01],
        [x + cx + 0.01, y + cy + 0.005, z + cz + 0.01],
        [x + cx, y + cy, z + cz + .01]  # make sure camera is not inside box: front
    )




def robot_union(x, z, rx=0.03, ry=0.025, rz=0.05):
    return Union(
        # Body box
        Box([x - rx, 0, z - rz], [x + rx, ry, z + rz], color([0.4, 0.4, 0.4])),
        # Wheels (rear left and right)
        Cylinder(
            [x + rx - 0.005, 0.005, z - rz],
            [x + rx - 0.005, 0.005, z - rz - 0.02],
            0.01,
            color([0.1, 0.1, 0.1]),
        ),
        Cylinder(
            [x - rx + 0.005, 0.005, z - rz],
            [x - rx + 0.005, 0.005, z - rz - 0.02],
            0.01,
            color([0.1, 0.1, 0.1]),
        ),
        # Camera mount
        Box(
            [x - 0.015, ry, z + rz - .01],
            [x + 0.015, ry + 0.02, z + rz],
            color([1.2, 0.2, 0.2]),
        ),
        # Camera box
        Box(
            cam_pos(x, 0, z, ry, rz)[0],
            cam_pos(x, 0, z, ry, rz)[1],
            
            #[x - 0.01, ry + 0.02, z + rz - 0.015],
            #[x + 0.01, ry + 0.03, z + rz + 0.005],
            color([0.1, 1.1, 0.1]),
        )
    )


def oval_track_segments(radius=0.25, width=0.02, gap=0.005, height=0.001, segments=60):
    objects = []
    length = 2 * np.pi * radius / segments - gap  # Length of each segment
    for i in range(segments):
        theta1 = 2 * np.pi * i / segments
        # Outer segment
        x1, z1 = radius * np.cos(theta1), radius * np.sin(theta1)
        objects.append(
            Box(
                [x1 - width / 2, height, z1 - length / 2],
                [x1 + width / 2, height, z1 + length / 2],
                "rotate",
                [0, 0,  0],
                color([1, 1, 1]),
            )
        )
    return objects


def project_point(point, cam_pos, look_at, fov_deg, img_width, img_height):
    # Build camera coordinate system
    forward = np.array(look_at) - np.array(cam_pos)
    forward /= np.linalg.norm(forward)

    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    # Camera view matrix (rotation)
    R = np.stack([right, up, -forward], axis=1)
    p_cam = np.dot(R.T, np.array(point) - np.array(cam_pos))

    # Check if point is in front of the camera
    if p_cam[2] <= 0:
        return None, None  # behind camera

    # Perspective projection
    aspect = img_width / img_height
    fov_rad = np.radians(fov_deg)
    scale = np.tan(fov_rad / 2) * p_cam[2]
    x = (p_cam[0] / scale) * (img_width / 2) + img_width / 2
    y = (-p_cam[1] / (scale / aspect)) * (img_height / 2) + img_height / 2

    # Bounding box could be estimated based on object size & distance
    return (int(x), int(y)), p_cam  # screen coordinate + relative cam coord


def create_scene(t, duration, view="robot"):
    pos = robot_position(t, duration)
    look_at = [pos[0], 0, pos[2] + 5 * rz]
    # camera_pos = [pos[0], ry + 0.1, pos[2] + rz / 3]
    camera_pos = cam_pos(pos[0],0,pos[2], ry,rz)[2]
    #print("Camera position:", camera_pos)
    #def cam_pos(x,y,z,ry,rz):


    if view == "bird":
        camera = birds_eye_camera
    elif view == "robot":
        camera = Camera("location", camera_pos, "look_at", look_at, "angle", camera_fov)
    elif view == "side":
        camera = side_camera

    track = oval_track_segments()



    return Scene(
        camera,
        [
            LightSource([0, 10, 0], "color", [1.0, 1.0, 1.0], "shadowless"),
            LightSource([.7,.4,.7], 'color', [.5,.5,.5],"spotlight","radius",40,"point_at",[0, 0, 0]),

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
            robot_union(pos[0], pos[2], rx, ry, rz),
            ## body
            # Box([pos[0] - rx/2, 0, pos[2] - rz/2], [pos[0] + rx/2, ry, pos[2] + rz/2], color([0.2, 0.2, 0.2])),
            # camera mount
            # Box([pos[0] - rx/4, ry + .01, pos[2] - rz/4], [pos[0] + rx/4, ry + 0.05, pos[2] + rz/4], color([0.3, 0.3, 0.3])),
            ## wheels
            ##Cylinder([pos[0] + 0.025, 0.005, pos[2] - 0.05], [pos[0] + 0.025, 0.005, pos[2] - 0.07], 0.01, color([0.05, 0.05, 0.05])),
            ## Cylinder([pos[0] - 0.025, 0.005, pos[2] - 0.05], [pos[0] - 0.025, 0.005, pos[2] - 0.07], 0.01, color([0.05, 0.05, 0.05])),
            # Arena floor
            Box([-0.5, -0.01, -0.5], [0.5, 0, 0.5], color([0.9, 0.9, 0.0])),
            # Oval track (50 cm diameter, 2 cm width)
            *track,
            # objects
            *objects,
            #Box([0.2, 0, 0.2], [0.25, 0.05, 0.25], color([1, 0.6, 0.5])),
            #Cone([0.1, 0, 0.1], 0.03, [0.1, 0.08, 0.1], 0, color([1, 0.8, 0])),
            #Sphere([-0.2, 0.04, 0.25], 0.04, color([0.4, 0.6, 1])),
            # bg
            Background("color", [1, 10, 1]),
        ],
    )


# Animation parameters
duration = 4.0
fps = 15
frames = int(duration * fps)


for i in range(frames):
    t = i / fps
    scene_robot = create_scene(t, duration, "robot")
    scene_bird = create_scene(t, duration, "bird")
    scene_side = create_scene(t, duration, "side")

    scene_robot.render(
        os.path.join(output_dir, f"robot_{i:03d}.png"),
        width=600,
        height=450,
        antialiasing=0.01,
    )
    scene_bird.render(
        os.path.join(output_dir, f"bird_{i:03d}.png"),
        width=600,
        height=450,
        antialiasing=0.01,
    )
    scene_side.render(
        os.path.join(output_dir, f"side_{i:03d}.png"),
        width=600,
        height=450,
        antialiasing=0.01,
    )

    # def project_point(point, cam_pos, look_at, fov_deg, img_width, img_height):
    # bounding boxes 
    pos = robot_position(t, duration)
    look_at = [pos[0], 0, pos[2] + 5 * rz]
    # camera_pos = [pos[0], ry + 0.1, pos[2] + rz / 3]
    camera_pos = cam_pos(pos[0],0,pos[2], ry,rz)[2]
    
    for obj in object_coords:
        # Calculate the center position of the bounding box
        pnt = [obj["pos1"][i] - (obj["pos1"][i] - obj["pos0"][i]) / 2 for i in range(3)]
        screen_coords, rel_pos = project_point(pnt,camera_pos, look_at, camera_fov, 600, 450)
        if screen_coords:
            print(f"Object {obj['type']} at frame {i:03d} on screen at:", screen_coords)
        else:
            print(f"Object {obj['type']} at frame {i:03d} not visible")
    

print("Rendering complete. Frames saved to:", output_dir)
