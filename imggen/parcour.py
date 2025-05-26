import os
import numpy as np
from vapory import Scene, Camera, LightSource, Background, Box, Cone, Sphere, Cylinder, Union, Pigment

def color(rgb):
    return Pigment('color', rgb)


# Output directory for frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Oval path parameters
a, b = 0.25, 0.25  # 50 cm elliptical path

# Robot dims
rx, ry, rz = 0.02, 0.05, 0.06  # Robot dimensions (6 cm x 3 cm x 1 cm)

camera_fov = 43.6     # horizontal FOV ~ matching 2.32 mm lens on ~1 mm sensor


birds_eye_camera = Camera(
    'location', [0, 1.5, 0],   # 1.5 m above the center
    'look_at',  [0, 0, 0],     # look at the center of the arena
    'angle', 60                # wider FOV to capture full track
)



def robot_position(t, duration):
    angle = 2 * np.pi * (t / duration)
    return [a * np.cos(angle), 0, b * np.sin(angle)]

# robot cam looks away from wheels

def oval_track_segments(radius_outer=0.26, radius_inner=0.24, height=0.001, segments=60):
    objects = []
    for i in range(segments):
        theta1 = 2 * np.pi * i / segments
        theta2 = 2 * np.pi * (i + 1) / segments

        # Outer segment
        x1, z1 = 0.25 * np.cos(theta1), 0.25 * np.sin(theta1)
        x2, z2 = 0.25 * np.cos(theta2), 0.25 * np.sin(theta2)
        objects.append(
            Cylinder([x1, height, z1], [x2, height, z2], 0.01, color([1, 1, 1]))
        )
    return objects



def create_scene(t, duration,view="robot"):
    pos = robot_position(t, duration)
    look_at = [pos[0], 0, pos[2] + 10*rz]
    camera_pos = [pos[0], ry + .1, pos[2] + rz/3]
    torch_pos = [pos[0], ry + .1, pos[2] + rz/3]

    if view == "bird":
        camera = birds_eye_camera
    elif view == "robot":
        camera = Camera(
            'location', camera_pos,
            'look_at', look_at,
            'angle', camera_fov
        )

    track = oval_track_segments()
    
    return Scene(
        camera,
        [
            LightSource([0, 10, 0], 'color', [1.4, 0.4, 0.4], 'shadowless'),
            LightSource([2, 4, -3], 'color', [1.5, 1.5, 1.5]),
            LightSource(
                torch_pos,
                'color', [1, 1, 5],
                'spotlight',
                'point_at', look_at,
                'radius', 20,           # beam width
                'falloff', 30,          # soft edge
                'tightness', 10,        # intensity at center
                'shadowless'
            ),
            # robot
            ## body
            Box([pos[0] - rx/2, 0, pos[2] - rz/2], [pos[0] + rx/2, ry, pos[2] + rz/2], color([0.2, 0.2, 0.2])),
            # camera mount
            Box([pos[0] - rx/4, ry + .01, pos[2] - rz/4], [pos[0] + rx/4, ry + 0.05, pos[2] + rz/4], color([0.3, 0.3, 0.3])),
            ## wheels
            Cylinder([pos[0] + 0.025, 0.005, pos[2] - 0.05], [pos[0] + 0.025, 0.005, pos[2] - 0.07], 0.01, color([0.05, 0.05, 0.05])),
            Cylinder([pos[0] - 0.025, 0.005, pos[2] - 0.05], [pos[0] - 0.025, 0.005, pos[2] - 0.07], 0.01, color([0.05, 0.05, 0.05])),
            # Arena floor
            Box([-0.5, -0.01, -0.5], [0.5, 0, 0.5], color([0.9, 0.9, 0.0])),
            # Oval track (50 cm diameter, 2 cm width)
            *track,
            #Cylinder([-0.25, 0.001, 0], [0.25, 0.001, 0], 0.26, color([1, 1, 1])),
            #Cylinder([-0.25, 0.002, 0], [0.25, 0.002, 0], 0.24, color([0.9, 0.9, 0.9])),
            # objects
            Box([0.2, 0, 0.2], [0.25, 0.05, 0.25], color([1, 0.6, 0.5])),
            Cone([0.1, 0, 0.1], 0.03, [0.1, 0.08, 0.1], 0, color([1, 0.8, 0])),
            Sphere([-0.2, 0.04, 0.25], 0.04, color([0.4, 0.6, 1])),
            # bg
            Background('color', [1, 10, 1])
        ])

# Animation parameters
duration = 3.0
fps = 10
frames = int(duration * fps)


for i in range(frames):
    t = i / fps
    scene_robot = create_scene(t, duration, "robot")
    scene_bird  = create_scene(t, duration, "bird")

    scene_robot.render(os.path.join(output_dir, f"robot_{i:03d}.png"), width=400, height=300, antialiasing=0.01)
    scene_bird.render(os.path.join(output_dir, f"bird_{i:03d}.png"), width=400, height=300, antialiasing=0.01)



print("Rendering complete. Frames saved to:", output_dir)

