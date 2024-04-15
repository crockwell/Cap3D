# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: June 22, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

import bpy
import sys
import mathutils
from mathutils import Vector, Matrix, Euler
import argparse
import numpy as np
import math
import os
import time
import pickle
from PIL import Image
import random

## solve the division problem
from decimal import Decimal, getcontext
getcontext().prec = 28  # Set the precision for the decimal calculations.

parser = argparse.ArgumentParser()
parser.add_argument('--object_path_pkl', type = str, required = True)
parser.add_argument("--parent_dir", type = str, default='./example_material')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

uid_paths = pickle.load(open(args.object_path_pkl, 'rb'))
# random.shuffle(uids)


bpy.context.scene.render.engine = 'CYCLES'
# small samples for fast rendering
bpy.context.scene.cycles.samples = 16
# bpy.context.scene.cycles.samples = 128
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    if 'NVIDIA' in d['name']:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    else:
        d["use"] = 0 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

render_prefs = bpy.context.preferences.addons['cycles'].preferences
render_device_type = render_prefs.compute_device_type
compute_device_type = render_prefs.devices[0].type if len(render_prefs.devices) > 0 else None
# Check if the compute device type is GPU
if render_device_type == 'CUDA' and compute_device_type == 'CUDA':
    # GPU is being used for rendering
    print("Using GPU for rendering")
else:
    # GPU is not being used for rendering
    print("Not using GPU for rendering")


# if the object is too far away from the origin, pull it closer
def check_object_location(mesh_objects, max_distance):
    # Compute the maximum distance of any object from the origin
    max_obj_distance = max(obj.location.length for obj in mesh_objects)

    # If any object is too far from the origin, move all mesh_objects closer to the origin
    if max_obj_distance > max_distance:
        print("mesh_objects are too far from the origin. Centering objects...")
        bbox_center, _ = compute_bounding_box(mesh_objects)
        for obj in mesh_objects:
            obj.location -= bbox_center
        bpy.context.view_layer.update()
    else:
        print("Object initial locations are within range.")
        print('max_obj_distance:', max_obj_distance, 'max_distance:', max_distance)

    # Compute the maximum distance again and check if it's within range
    max_obj_distance = max(obj.location.length for obj in mesh_objects)
    if max_obj_distance > max_distance:
        print("Objects are still too far from the origin. Please adjust the object locations and try again.")
        return False
    else:
        print("Object locations are within range.")
        return True

# compute the bounding box of the mesh objects
def compute_bounding_box(mesh_objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in mesh_objects:
        matrix_world = obj.matrix_world
        mesh = obj.data

        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co

            min_coords = Vector((min(min_coords[i], global_coord[i]) for i in range(3)))
            max_coords = Vector((max(max_coords[i], global_coord[i]) for i in range(3)))

    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    return bbox_center, bbox_size

# normalize objects 
def normalize_and_center_objects(mesh_objects, normalization_range):

    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    # Check the location of the objects and move them closer to the origin if necessary
    check_object_location(mesh_objects, 1000)

    # Compute the bounding box of the objects again after making adjustments
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)

    # Normalize the objects within a certain range
    max_dimension = max(bbox_size.x, bbox_size.y, bbox_size.z)
    scaling_factor = normalization_range / max_dimension

    for obj in mesh_objects:
        mesh = obj.data
        matrix_world = obj.matrix_world
        inv_matrix_world = matrix_world.inverted()
        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co
            global_coord -= bbox_center
            global_coord *= scaling_factor
            vert.co = inv_matrix_world @ global_coord
        mesh.update()
        obj.data.update()

    bpy.context.view_layer.update()
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)
    print('final bbox_center: ', bbox_center)
    print('final bbox_size: ', bbox_size)

    return bbox_center, bbox_size

# check if rendered object will cross the boundary of the image
def project_points_to_camera_space(obj, camera):
    bpy.context.view_layer.update()
    # Get the 8 corners of the bounding box in local space
    bbox_local = [Vector(corner) for corner in obj.bound_box]
    # print(bbox_local)

    # Transform bounding box corners to world space
    bbox_world = [obj.matrix_world @ corner for corner in bbox_local]
    bbox_world = [np.array(corner) for corner in bbox_world]  # convert to numpy

    # Get the 4x4 transformation matrix of the camera
    RT = np.array(camera.matrix_world.inverted())
    RT = RT[:3, :4]  # Remove the last row to make it a 3x4 matrix

    # Get the intrinsic matrix K from the camera properties
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    f_x = width / 2.0 / np.tan(camera.data.angle / 2.0)
    f_y = height / 2.0 / np.tan(camera.data.angle / 2.0)
    c_x = width / 2.0
    c_y = height / 2.0

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    bbox_camera = []
    bbox_image = []

    for vertex in bbox_world:
        # Transform from world to camera space
        XYZ_camera = np.dot(RT, np.append(vertex, 1))  # Append 1 to make it a 4-element vector for multiplication with RT

        # Project from camera space to image space
        XYZ_image = np.dot(K, XYZ_camera)

        # Homogenize to get pixel coordinates
        XYZ_image /= XYZ_image[2]

        bbox_camera.append(XYZ_camera)
        bbox_image.append(XYZ_image[:2])  # Keep only x and y

    # Check if the coordinates are within the normalized device coordinates [-1, 1]
    is_within_ndc = all(np.all(np.abs(vertex[:2]) <= 1) for vertex in bbox_image)

    # print(is_within_ndc)
    return bbox_image

# prepare the scene
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Create lights

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()

def create_light(name, light_type, energy, location, rotation):
    bpy.ops.object.light_add(type=light_type, align='WORLD', location=location, scale=(1, 1, 1))
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.rotation_euler = rotation
    return light

def three_point_lighting():
    
    # Key ligh
    key_light = create_light(
        name="KeyLight",
        light_type='AREA',
        energy=1000,
        location=(4, -4, 4),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    key_light.data.size = 2

    # Fill light
    fill_light = create_light(
        name="FillLight",
        light_type='AREA',
        energy=300,
        location=(-4, -4, 2),
        rotation=(math.radians(45), 0, math.radians(135))
    )
    fill_light.data.size = 2

    # Rim/Back light
    rim_light = create_light(
        name="RimLight",
        light_type='AREA',
        energy=600,
        location=(0, 4, 0),
        rotation=(math.radians(45), 0, math.radians(225))
    )
    rim_light.data.size = 2

three_point_lighting()

for i in range(8):
    os.makedirs(os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d'%i), exist_ok=True)
    os.makedirs(os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d_CamMatrix'%i), exist_ok=True)
os.makedirs(os.path.join(args.parent_dir, 'Cap3D_captions'), exist_ok=True)

for uid_path in uid_paths:
    if not os.path.exists(uid_path):
        continue

    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    _, ext = os.path.splitext(uid_path)
    ext = ext.lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=uid_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=uid_path)

    print('begin*************')
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    # Compute the bounding box for the objects
    normalization_range = 1.0
    bbox_center, bbox_size = normalize_and_center_objects(mesh_objects, normalization_range)

    distance = max(bbox_size.x, bbox_size.y, bbox_size.z)
    ratio = 1.15
    elevation_factor = 0.2


    camera = bpy.context.scene.camera
    name = uid_path.split('/')[-1].split('.')[0]
    for camera_opt in range(-1, 8):
        # use transparent background to adjust camera distance
        if camera_opt == -1:
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.scene.render.film_transparent = True
            camera.location = Vector((distance * ratio, - distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 0:
            img_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view_bg', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
            img = Image.open(img_path)
            img_array = np.array(img)
            if np.sum(img_array<10) > 1020000:
                print(name, 'WARNING: rendered image may contain too much white space')

            # change to white background to render the final 8 views
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            bpy.context.scene.render.film_transparent = False
            camera.location = Vector((distance * ratio, - distance * ratio, distance * elevation_factor * ratio))

            # check if the object is within the image
            while True:
                flag_list = []
                for obj in mesh_objects:
                    bbox_image = project_points_to_camera_space(obj, camera)
                    if np.max(np.array(bbox_image) > 512) or np.min(np.array(bbox_image) < 0):
                        flag_list.append(0)
                        ratio += 0.1
                        camera.location = Vector((distance * ratio, - distance * ratio, distance * elevation_factor * ratio))
                if len(flag_list) == 0:
                    break
        elif camera_opt == 1:
            camera.location = Vector((- distance * ratio,  distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 2:
            elevation_factor = 0.5
            camera.location = Vector((distance * ratio,  -distance * ratio*0.5, distance * elevation_factor * ratio))
        elif camera_opt == 3:
            elevation_factor = 0.7
            camera.location = Vector((- distance * ratio *0.5,  distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 4:
            camera.location = Vector((distance * ratio,  distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 5:
            camera.location = Vector((-distance * ratio,  -distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 6:
            elevation_factor = 0.5
            camera.location = Vector((distance * ratio,  distance * ratio*0.5, -distance * elevation_factor * ratio))
        elif camera_opt == 7:
            elevation_factor = 0.7
            camera.location = Vector((- distance * ratio *0.5,  -distance * ratio, -distance * elevation_factor * ratio))

        # Make the camera point at the bounding box center
        direction = (bbox_center - camera.location).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()

        camera.data.clip_start = 0.1
        camera.data.clip_end = max(1000, distance * 2)

        print('bbox_center: ', bbox_center)
        print('bbox_size: ', bbox_size)
        print('distance: ', distance)
        print('camera.location: ', camera.location)

        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        if camera_opt == -1:
            file_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view_bg', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
            bpy.context.scene.render.filepath = file_path
            if os.path.exists(file_path):
               continue
        else:
            file_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d'%camera_opt, '%s_%d.png'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            bpy.context.scene.render.filepath = file_path
            if os.path.exists(file_path):
               continue

        bpy.ops.render.render(write_still=True)

        def get_3x4_RT_matrix_from_blender(cam):
            # Use matrix_world instead to account for all constraints
            location, rotation = cam.matrix_world.decompose()[0:2]
            R_world2bcam = rotation.to_matrix().transposed()

            # Use location from matrix_world to account for constraints:     
            T_world2bcam = -1*R_world2bcam @ location

            # put into 3x4 matrix
            RT = Matrix((
                R_world2bcam[0][:] + (T_world2bcam[0],),
                R_world2bcam[1][:] + (T_world2bcam[1],),
                R_world2bcam[2][:] + (T_world2bcam[2],)
                ))
            return RT

        if camera_opt>=0:
            RT = get_3x4_RT_matrix_from_blender(camera)
            
            RT_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d_CamMatrix'%camera_opt, '%s_%d.npy'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            if os.path.exists(RT_path):
                continue
            np.save(RT_path, RT)

bpy.ops.wm.quit_blender()

