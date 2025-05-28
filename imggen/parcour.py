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
)


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
    {"type":"box","pos0":[0.2, 0, 0.2],"pos1":[.24,.04,.24]},  # Box
    {"type":"cone","pos0":[0.1, 0, 0.1],"r0":.03,"pos1":[0.1, 0.08, 0.1],"r1":0},  # Cone
    {"type":"sphere","pos0":[-0.2, 0.04, 0.25],"r0":.04},  # Sphere
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
    return [a * np.cos(angle), 0, b * np.sin(angle),np.degrees(angle)]


def cam_pos(x,y,z,ry,rz,h=.02):
    cx = 0
    cy = ry #  + .02
    cz = rz  
    return (
        [x - cx - 0.01, y + cy + h/4, z + cz - 0.01],
        [x + cx + 0.01, y + cy + 3*h/4, z + cz + 0.01],
        [x + cx, y + cy + h/2, z + cz]  # make sure camera is not inside box: front
    )




def robot_union(x, z, rx=0.03, ry=0.025, rz=0.05, rot=0):
    print(x,z,rot)
    cam0 = cam_pos(0, 0, 0, ry, 0)[0]
    cam1 = cam_pos(0, 0, 0, ry, 0)[1]
    return Union(
        # Body box
        Box([- rx/2, 0, 0], [rx/2, ry, -rz], color([0.4, 0.4, 0.4])),
        # Wheels (rear left and right)
        Cylinder(
            [ -.01, 0.0, 0],
            [  .01, 0.0, 0],
            0.02,"rotate", [0, 0, 0],"translate", [-rx/2 -.015, 0.02, 0],
            color([1,1,1]),
        ),
        Cylinder(
            [ -.01, 0.0, 0],
            [  .01, 0.0, 0],
            0.02, "rotate", [0, 0, 0],"translate", [rx/2 + .015, 0.02, 0],
            color([1,1,1]),
        ),
        # Camera mount
        Box(
            [- 0.015, ry, - .01],
            [+ 0.015, ry + 0.02, 0],
            color([1.2, 0.2, 0.2]),
        ),
        # Camera box
        Box(
            [cam0[0],cam0[1],cam0[2] - .01],
            [cam1[0],cam1[1],cam1[2] - .015],
            #cam_pos(0, 0, 0, ry*.9, 0)[0],
            #cam_pos(0, 0, 0, ry*.9, 0)[1],
            color([0.1, 1.1, 0.1]),
        ),
        "rotate",[0, -rot, 0],
        "translate",[x, 0, z]
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
    """
    Projects a 3D world-space point onto 2D image space using pinhole projection.
    Returns 2D screen coordinates and camera-space coordinates (or None if not visible).
    """
    right, up, forward = get_camera_basis(cam_pos, look_at)
    print("Camera basis vectors:", right, up, forward)

    # Build rotation matrix: columns = right, up, -forward
    R = np.stack([right, up, forward], axis=1)

    # Transform point into camera space
    p_world = np.array(point) - np.array(cam_pos)
    p_cam = np.dot(R.T, p_world)

    print("Point in camera space:", p_cam)
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


def create_scene(t, duration, view="robot"):
    pos = robot_position(t, duration)
    angle = pos[3]  # angle in degrees
    camera_pos = cam_pos(pos[0],0,pos[2], ry,0)[2]
    cam_dz = np.cos(np.radians(-angle))
    cam_dx = np.sin(np.radians(-angle))
    x = pos[0] + cam_dx * 5*rz
    z = pos[2] + cam_dz * 5*rz
    look_at = [x, 0, z]  # Look at point in front of the robot    
    
    #print("Camera position:", camera_pos)
    #def cam_pos(x,y,z,ry,rz):


    if view == "bird":
        camera = birds_eye_camera
    elif view == "robot":
        camera = Camera("location", camera_pos, "look_at", look_at, "angle", camera_fov)
    elif view == "side":
        camera = side_camera

    track = oval_track_segments()

    pointer = Cylinder(
        [camera_pos[0], camera_pos[1] + 0.005, camera_pos[2]], # camera_pos,
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
            robot_union(pos[0], pos[2], rx, ry, rz,angle),
            pointer,
            antenna,
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

    # project_point(point, cam_pos, look_at, fov_deg, img_width, img_height)
    # get coordinates
    pos = robot_position(t, duration)
    angle = pos[3]  # angle in degrees
    camera_pos = cam_pos(pos[0],0,pos[2], ry,0)[2]
    cam_dz = np.cos(np.radians(-angle))
    cam_dx = np.sin(np.radians(-angle))
    x = pos[0] + cam_dx * 5*rz
    z = pos[2] + cam_dz * 5*rz
    look_at = [x, 0, z]  # Look at point in front of the robot    

    print("Robot position at frame", i, ":", pos)
    print("Camera position:", camera_pos)
    print("Look at position:", look_at)

    
    # bounding boxes 
    # Project objects onto the camera view
    
    visible_obj = []
    for obj in object_coords:
        # Calculate the center position of the bounding box
        pnt = [obj["pos1"][i] - (obj["pos1"][i] - obj["pos0"][i]) / 2 for i in range(3)]
        screen_coords, rel_pos = project_point(pnt,camera_pos, look_at, camera_fov, 600, 450)
        if screen_coords:
            print(f"Object {obj['type']}, {pnt} at frame {i:03d} on screen at:", screen_coords, rel_pos)
            visible_obj.append({"obj":obj["type"], "coords:":screen_coords})
        else:
            print(f"Object {obj['type']}, {pnt} at frame {i:03d} not visible")
    with open(os.path.join(output_dir, f"visible_objects_{i:03d}.txt"), "w") as f:
        json.dump(visible_obj,f)

    # Open the rendered image
    img_path = os.path.join(output_dir, f"robot_{i:03d}.png")
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_width, img_height = img.size

    # Draw bounding boxes on the image

    draw = ImageDraw.Draw(img)
    for obj in visible_obj:
        coords = obj["coords:"]
        x, y = coords
        box_size = 10  # Size of the bounding box
        draw.rectangle(
            [x - box_size, y - box_size, x + box_size, y + box_size],
            outline="red",
            width=2,
        )
        draw.text((x + box_size, y - box_size), obj["obj"], fill="red")

    # Save the annotated image
    annotated_img_path = os.path.join(output_dir, f"robot_ann{i:03d}.png")
    img.save(annotated_img_path)
        

print("Rendering complete. Frames saved to:", output_dir)
