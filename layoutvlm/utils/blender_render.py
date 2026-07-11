import argparse, sys, os, math, re
import bpy
import bmesh
from bpy_extras.image_utils import load_image
import math
import numpy as np
import os
import bpy
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import shift
import json
from mathutils import Vector
from utils.blender_utils import reset_scene, setup_background, create_wall_mesh, add_material, setup_camera, load_hdri, set_rendering_settings, apply_texture_to_object
import utils.transformations as tra
from utils.colors import get_categorical_colors
import trimesh
import cv2
from utils.blender_utils import get_pixel_coordinates, reset_blender
from utils.plot_utils import annotate_image_with_coordinates


def get_visual_marks(floor_vertices, scene, cam, interval=1):
    # The world coordinate we want to project
    visual_marks = dict()
    min_vertices = np.min(floor_vertices, axis=0)
    max_vertices = np.max(floor_vertices, axis=0)
    for min_x in range(math.floor(min_vertices[0]), math.ceil(max_vertices[0])+2, interval):
        for min_y in range(math.floor(min_vertices[1]), math.ceil(max_vertices[1])+2, interval):
            world_coord = Vector((min_x, min_y, 0))
            pixel_x, pixel_y = get_pixel_coordinates(scene, cam, world_coord)
            visual_marks[(min_x, min_y)] = (pixel_x, pixel_y)
    return visual_marks


def create_bounding_box(position, rotation, scale, color, transparency=0.5):
    """

    Example:
    create_bounding_box(position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 2, 0.5), color=(1, 0, 0), transparency=0.3)

    :param position: xyz coordinates, in meters
    :param rotation: euler angles, in radians
    :param scale: xyz dimensions, in meters
    :param color: rgb values in the range [0, 1]
    :param transparency: alpha value in the range [0, 1]
    :return:
    """

    # Add a cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=position, rotation=rotation)
    bounding_box = bpy.context.object

    # Scale the cube to the desired dimensions
    bounding_box.scale = scale

    # Create a new material with transparency
    mat = bpy.data.materials.new(name="BoundingBoxMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)  # Set the color
    bsdf.inputs["Alpha"].default_value = transparency
    mat.blend_method = 'BLEND'

    # Assign the material to the bounding box
    if bounding_box.data.materials:
        bounding_box.data.materials[0] = mat
    else:
        bounding_box.data.materials.append(mat)

    # Optional: Enable backface culling for better visual appearance
    mat.use_backface_culling = True

    return bounding_box


def get_wall_normal(corner1, corner2, add_radian=None):
    """corner1 and corner2 are the two points on the wall, given in counter-clockwise order"""
    vector = np.array([corner2[0] - corner1[0], corner2[1] - corner1[1]], dtype=np.float32)
    if add_radian:
        # Assuming add_radian is the angle to add in radians
        cos_theta = np.cos(add_radian)
        sin_theta = np.sin(add_radian)
        # Construct the rotation matrix
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]], dtype=np.float32)
        # Multiply the rotation vector by the rotation matrix
        vector = np.dot(rotation_matrix, vector)

    vector = vector / np.linalg.norm(vector)
    return vector


def get_obj_dimensions(obj, frame="object"):
    # Ensure the object is of type 'MESH'
    if obj.type != 'MESH':
        raise ValueError(f"The object '{obj.name}' is not a mesh.")

    # Get the bounding box coordinates in local space
    bbox = [Vector(corner) for corner in obj.bound_box]

    # Convert the local bounding box coordinates to world space
    if frame == "world":
        bbox = [obj.matrix_world @ corner for corner in bbox]

    # Calculate the minimum and maximum coordinates along each axis
    min_x = min(corner.x for corner in bbox)
    max_x = max(corner.x for corner in bbox)
    min_y = min(corner.y for corner in bbox)
    max_y = max(corner.y for corner in bbox)
    min_z = min(corner.z for corner in bbox)
    max_z = max(corner.z for corner in bbox)

    # Calculate the dimensions
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z

    return [width, depth, height]


# Function to create an arrow representing an axis
def create_arrow(start, end, radius=0.02, color=(1, 0, 0, 1), name="Arrow"):
    # Create a cylinder (for the shaft)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=(end - start).length, location=(start + end) / 2)
    shaft = bpy.context.object
    shaft.name = name + "_shaft"

    # Align the shaft to point towards the end
    direction = end - start
    rot_quat = direction.to_track_quat('Z', 'Y')
    shaft.rotation_euler = rot_quat.to_euler()

    # Create a cone (for the tip)
    tip_length = radius * 2
    bpy.ops.mesh.primitive_cone_add(radius1=radius * 2, depth=tip_length, location=end)
    tip = bpy.context.object
    tip.name = name + "_tip"

    # Align the tip to point towards the end
    tip.rotation_euler = rot_quat.to_euler()

    # Create a material for the arrow
    mat = bpy.data.materials.new(name + "_Material")
    mat.diffuse_color = color
    shaft.data.materials.append(mat)
    tip.data.materials.append(mat)

    # Combine shaft and tip into one object
    bpy.ops.object.select_all(action='DESELECT')
    shaft.select_set(True)
    tip.select_set(True)
    bpy.ops.object.join()


# Function to add a coordinate frame at a specific location
def add_coordinate_frame(location=Vector((0, 0, 0.)), scale=1.0):
    # Define the length and color of each axis
    axis_length = scale
    x_color = (1, 0, 0, 1)  # Red
    y_color = (0, 1, 0, 1)  # Green
    z_color = (0, 0, 1, 1)  # Blue

    # Create the X axis arrow
    create_arrow(location, location + Vector((axis_length, 0, 0)), color=x_color, name="X_Axis")

    # Create the Y axis arrow
    create_arrow(location, location + Vector((0, axis_length, 0)), color=y_color, name="Y_Axis")

    # Create the Z axis arrow
    create_arrow(location, location + Vector((0, 0, axis_length)), color=z_color, name="Z_Axis")


def show_current_render(save_dir):
    render_path = f"{save_dir}/tmp.png"
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    img = cv2.imread(render_path)
    cv2.imshow("render", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def render_existing_scene(placed_assets, task, save_dir, add_hdri=True, topdown_save_file=None, sideview_save_file=None, add_coordinate_mark=True,
                          annotate_object=True, annotate_wall=True, render_top_down=True, adjust_top_down_angle=None, high_res=False, rotate_90=True,
                          apply_3dfront_texture=False, recenter_mesh=True, fov_multiplier=1.1, default_font_size=None,
                          combine_obj_components=False, side_view_phi=45, side_view_indices=[3], save_blend=False,
                          add_object_bbox=False, ignore_asset_instance_idx=False, floor_material="Travertine008"):
    """
    :param placed_assets: the set of assets that have been placed in the scene / to be rendered
    :param task: just for getting the boundary
    :param save_dir:
    :param add_hdri:
    :param topdown_save_file:
    :param sideview_save_file:
    :param add_coordinate_mark:
    :param annotate_object:
    :param annotate_wall:
    :param render_top_down:
    :param adjust_top_down_angle:
    :param high_res:
    :param rotate_90:
    :param apply_3dfront_texture: this is needed because loading material directly from .obj will fail
    :param recenter_mesh: whether to re-center the mesh to centroid.
    :param fov_multiplier: a factor for increasing the fov of the camera.
    :param combine_obj_components: whether to combine the components of the obj file into one object.
                                   Recommend to set to True if loading 3d front .obj files.
                                   If set to False, one example failure case is 0003d406-5f27-4bbf-94cd-1cff7c310ba1_LivingRoom-54780
    :param apply_3dfront_orientation_correction: 3d front .obj faces y axis by default, if we want to make it face x axis, set to True.
    :param ignore_asset_instance_idx: useful for generating object renderings for each group of objects
    :return:
    """

    if add_object_bbox:
        assert annotate_object, "add_object_bbox can only be True when annotate_object is True"

    reset_blender()
    setup_background()
    # compute scene boundary
    floor_vertices = np.array(task["boundary"]["floor_vertices"])
    floor_x_values = [point[0] for point in floor_vertices]
    floor_y_values = [point[1] for point in floor_vertices]
    floor_center_x = (max(floor_x_values) + min(floor_x_values)) / 2
    floor_center_y = (max(floor_y_values) + min(floor_y_values)) / 2
    floor_width = max(max(floor_x_values) - min(floor_x_values), max(floor_y_values) - min(floor_y_values))

    wall_height = np.array(task["boundary"]["wall_height"]) if "wall_height" in task["boundary"] else 1
    # Clear existing mesh objects and lights in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    # Create a new empty scene
    # Set the new empty scene as the active scene
    bpy.context.window.scene = bpy.context.scene
    # Update the user interface
    bpy.context.view_layer.update()
    # build floor and walls
    # floor_obj = create_wall_mesh("floor", floor_vertices)
    floor_obj = create_wall_mesh("floor", floor_vertices)
    if adjust_top_down_angle is not None:
        # asset centric rendreing
        floor_material = "Travertine008"
    add_material(floor_obj, os.path.join("/viscam/projects/SceneAug/ambientcg", floor_material))

    bpy.ops.object.select_all(action='DESELECT')
    floor_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # Unwrap using Smart UV Project
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

    if annotate_object:
        candidate_colors = get_categorical_colors(20, colormap_name='tab20', color_range="0-1", color_format="rgb")
        asset_count = 0
        asset_dict = {}

    for instance_id, asset in task["assets"].items():
        if instance_id not in placed_assets.keys():
            continue
        objects_before_import = set(bpy.context.scene.objects)
        file_path = asset["path"]
        if ".gltf" in file_path or ".glb" in file_path:
            bpy.ops.import_scene.gltf(filepath=file_path)
        elif ".obj" in file_path:
            bpy.ops.wm.obj_import(filepath=file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        if not combine_obj_components:
            loaded = bpy.context.view_layer.objects.active
        else:
            objects_after_import = set(bpy.context.scene.objects)
            new_objects = objects_after_import - objects_before_import
            bpy.ops.object.select_all(action='DESELECT')
            for obj in new_objects:
                obj.select_set(True)
            try:
                bpy.ops.object.join()
            except Exception as e:
                print(f"Error joining objects: {e}, ignoring the join operation")
            loaded = bpy.context.view_layer.objects.active

        # add texture for front 3d objects
        if apply_3dfront_texture:
            texture_path = os.path.join(os.path.split(file_path)[0], "texture.png")
            if os.path.exists(texture_path):
                apply_texture_to_object(loaded, texture_path)

        bpy.ops.object.select_all(action='DESELECT')
        loaded.select_set(True)
        if recenter_mesh:
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
        # rotate the object according to pose["rotation"] (0, 90, 90)
        # after preprocessing, the object faces -y axis by default
        # make the object face +x by default (Blender rotates the object clockwise!)
        if rotate_90:
            bpy.ops.transform.rotate(value=-math.radians(90), orient_axis='Z')
        # for i in range(3): assert loaded.rotation_euler[i] ==0
        bpy.context.object.rotation_mode = "XYZ"
        if "scale" in placed_assets[instance_id]:
            loaded.scale = placed_assets[instance_id]["scale"]
        #elif "scale" in task["assets"][instance_id]:
        #    loaded.scale = placed_assets[instance_id]["scale"]
        # TODO (Weiyu): Why are we reading both from the `placed_assets` dict and the `task["assets"]` dict? Can we remove one?
        if isinstance(placed_assets[instance_id]["rotation"], float):
            loaded.rotation_euler[-1] += np.deg2rad(placed_assets[instance_id]["rotation"])
        else:
            for i in range(3):
                loaded.rotation_euler[i] += np.deg2rad(placed_assets[instance_id]["rotation"][i])
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        for i in range(3): assert loaded.rotation_euler[i] ==0 
        #bpy.ops.transform.rotate(value=math.radians(pose['rotation']['y']), orient_axis='Z')
        # dim = get_dimensions_with_hierarchy(loaded)
        # holo_dim = data_holodeck[obj['assetId']]['assetMetadata']['boundingBox']
        xyz_location = placed_assets[instance_id]["position"]
        # default place the object on the floor
        if len(xyz_location) == 2:
            xyz_location = [xyz_location[0], xyz_location[1], placed_assets[instance_id]["scale"] * placed_assets[instance_id]["assetMetadata"]["boundingBox"]["z"]/2]
        loaded.location = xyz_location

        if annotate_object:
            # create obb for the object
            # NOTE: for 3d front assets, the center is at the bottom, but the center of the bbox is at the center
            #asset_location = loaded.location.copy()
            #asset_location[2] = asset_location[2] + asset_scale[2] / 2  # add z
            # extract location, rotation, scale from asset
            asset_position = placed_assets[instance_id]["position"]
            asset_rotation = np.deg2rad(placed_assets[instance_id]["rotation"]) #loaded.rotation_euler
            try:
                asset_scale = get_obj_dimensions(loaded, frame="object")
            except Exception as e:
                print(f"Error getting object dimensions: {e}, using the bounding box instead")
                import pdb;pdb.set_trace()
                continue

            # asset_scale_2 = trimesh.load(file_path).bounding_box.extents
            # asset_tf = tra.euler_matrix(asset_rotation[0], asset_rotation[1], asset_rotation[2])
            # asset_tf = asset_tf @ tra.euler_matrix(0, 0, np.pi/2)
            # asset_rotation = tra.euler_from_matrix(asset_tf)
            if add_object_bbox:
                bbox_rotation = loaded.rotation_euler
                create_bounding_box(asset_position, bbox_rotation, scale=asset_scale, color=candidate_colors[asset_count], transparency=0.3)

            if ignore_asset_instance_idx:
                asset_name = asset["asset_var_name"]
            else:
                asset_name = f"{asset['asset_var_name']}[{asset['instance_idx']}]"

            asset_dict[asset_count] = {
                "position": asset_position, 
                "rotation": asset_rotation,
                "size": asset_scale,
                "name": asset_name,
                "path": file_path, # "texture_path": texture_path,
                "category": asset["category"]
            }
            asset_count += 1

    if add_coordinate_mark:
        # asset-centric mode, add coordinate frame at the bottom left corner of the room
        if adjust_top_down_angle is not None:
            add_coordinate_frame(Vector((floor_center_x-floor_width/2, floor_center_y-floor_width/2, 0)))
        # scene-centric mode, add coordinate frame at the origin
        else:
            add_coordinate_frame()

    ### add light
    if add_hdri:
        load_hdri()

    output_images = []
    set_rendering_settings(high_res=high_res)
    visual_marks = dict()
    ### render top down images
    if render_top_down:
        ### setup camera
        cam, cam_constraint = setup_camera(
            floor_center_x, floor_center_y, floor_width, wall_height,
            fov_multiplier=fov_multiplier, use_damped_track=(adjust_top_down_angle is not None)
        )
        interval = 2
        if adjust_top_down_angle is not None:
            # _phi = math.radians(20)
            cam.rotation_euler = (0, 0, 0)
            original_z = cam.location.z # * 1.5
            theta = 0
            phi = math.radians(adjust_top_down_angle)
            point = (
                floor_center_x + original_z * math.sin(phi) * math.cos(theta),
                floor_center_y + original_z * math.sin(phi) * math.sin(theta),
                original_z * math.cos(phi),
            )
            cam.location = point
            interval = 1
            
        render_path = topdown_save_file if topdown_save_file else f"{save_dir}/top_down_rendering.png" 
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        if add_coordinate_mark:
            _visual_marks = get_visual_marks(floor_vertices, bpy.context.scene, cam, interval=interval)
            annotate_image_with_coordinates(image_path=render_path, visual_marks=_visual_marks, output_path=render_path, format="coordinate")

        _visual_marks = []
        if annotate_object:
            for asset_name in asset_dict:
                asset_data = asset_dict[asset_name]
                asset_rotation = asset_data["rotation"]
                pixel_x, pixel_y = get_pixel_coordinates(bpy.context.scene, cam, asset_data["position"])
                end_arrow_pixel_x, end_arrow_pixel_y = get_pixel_coordinates(
                    bpy.context.scene, cam, 
                    [
                        asset_data["position"][0] + 0.75 * math.cos(asset_rotation[-1]),
                        asset_data["position"][1] + 0.75 * math.sin(asset_rotation[-1]),
                        asset_data["position"][2]
                    ]
                )
                _visual_marks.append({ "text": asset_data["name"], "pixel": (pixel_x, pixel_y), "end_arrow_pixel": (end_arrow_pixel_x, end_arrow_pixel_y)})
                print(f"Asset {asset_data['name']} is at pixel ({pixel_x}, {pixel_y})")

        if annotate_wall:
            # Add walls to _visual_marks
            for i, vertex in enumerate(floor_vertices):
                next_vertex = floor_vertices[(i + 1) % len(floor_vertices)]
                # if wall is less than 2 meters, don't annotate
                if np.linalg.norm(np.array(vertex) - np.array(next_vertex)) < 2:
                    continue
                wall_center = [(vertex[0] + next_vertex[0]) / 2, (vertex[1] + next_vertex[1]) / 2, 0]
                # Move the room center 0.5 meters away from [center_x, center_y]
                # direction_vector should be the wall normal direction
                # direction_vector = np.array([floor_center_x, floor_center_y]) - np.array(wall_center[:2])
                # direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
                direction_vector = get_wall_normal(vertex, next_vertex, add_radian=np.pi/2)
                # get the wall normal
                new_wall_center = np.array(wall_center[:2]) - 0.2 * direction_vector
                wall_center = [new_wall_center[0], new_wall_center[1], 0]
                pixel_x, pixel_y = get_pixel_coordinates(bpy.context.scene, cam, wall_center)
                end_arrow_pixel_x, end_arrow_pixel_y = get_pixel_coordinates(
                    bpy.context.scene, cam, 
                    [
                        wall_center[0] + 0.75 * direction_vector[0],
                        wall_center[1] + 0.75 * direction_vector[1],
                        0
                    ]
                )
                _visual_marks.append({"text": f"walls[{i}]", "pixel": (pixel_x, pixel_y), "end_arrow_pixel": (end_arrow_pixel_x, end_arrow_pixel_y), "color": "white"})

        if adjust_top_down_angle is not None:
            # asset centric mode
            annotate_image_with_coordinates(image_path=render_path, visual_marks=_visual_marks, output_path=render_path, format="text", default_font_size=24)
        else:
            # scene centric mode
            annotate_image_with_coordinates(image_path=render_path, visual_marks=_visual_marks, output_path=render_path, format="text", default_font_size=24 if default_font_size is None else default_font_size)
        output_images.append(render_path)

    ### render side images
    cam, cam_constraint = setup_camera(
        floor_center_x, floor_center_y, floor_width, wall_height,
        fov_multiplier=fov_multiplier, use_damped_track=False
    )
    original_z = cam.location.z
    # remove all cam constraints

    for side_view_index in side_view_indices:
        # set the camera position
        theta = (side_view_index / 4) * math.pi * 2
        _phi = math.radians(side_view_phi)
        point = (
            floor_center_x + original_z * math.sin(_phi) * math.cos(theta),
            floor_center_y + original_z * math.sin(_phi) * math.sin(theta),
            original_z * math.cos(_phi),
        )
        cam.location = point
        # render the image
        render_path = sideview_save_file if sideview_save_file else f"{save_dir}/side_rendering_{side_view_phi}_{side_view_index}.png"
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        if add_coordinate_mark:
            visual_marks = get_visual_marks(floor_vertices, bpy.context.scene, cam, interval=2)
            annotate_image_with_coordinates(image_path=render_path, visual_marks=visual_marks, output_path=render_path)
        output_images.append(render_path)

    if save_blend:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=f"{save_dir}/scene.blend")

    return output_images, visual_marks
