#!/usr/bin/env python3
import os

urdf_path = os.path.expanduser('~/ros2_ws/src/arena_viz/models/DDRena/DDRena.urdf')

with open(urdf_path, 'r') as f:
    content = f.read()

# This replaces the relative path with the global ROS package path
fixed_content = content.replace('filename="meshes/', 'filename="package://arena_viz/models/DDRena/meshes/')

with open(urdf_path, 'w') as f:
    f.write(fixed_content)

print("Successfully updated all mesh paths in DDRena.urdf")
