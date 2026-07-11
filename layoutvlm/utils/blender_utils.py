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


def setup_background():
    import bpy
    # Set up a new world or modify the existing one
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    # Clear existing nodes
    nodes = world.node_tree.nodes
    nodes.clear()
    # Create the nodes for the world shader
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = (200, 0)
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.location = (0, 0)
    # Set background node to emit white light
    background_node.inputs['Color'].default_value = (1, 1, 1, 1)  # Background appears white
    background_node.inputs['Strength'].default_value = 1.0  # Uniform light strength
    # Connect the nodes
    world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    print("White background set up.")

def reset_blender():
    bpy.ops.wm.read_factory_settings()
    #for scene in bpy.data.scenes:
    #    for obj in scene.objects:
    #        scene.objects.unlink(obj)
    #for block in bpy.data.orphaned_data:
    #    bpy.data.orphaned_data.remove(block)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    for scene in bpy.data.scenes:
        if scene.rigidbody_world:
            scene.rigidbody_world.point_cache.frame_start = 1
            bpy.ops.ptcache.free_bake_all()

def clear_render_results():
    for scene in bpy.data.scenes:
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR' and space.image:
                        space.image = None
    if bpy.data.images.get("Render Result"):
        bpy.data.images.remove(bpy.data.images["Render Result"], do_unlink=True)

def reset_scene():
    # Clear render results first
    #clear_render_results()
    
    # Unlink all objects first
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Check for users and remove materials
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material, do_unlink=True)

    # Check for users and remove textures
    for texture in bpy.data.textures:
        if texture.users == 0:
            bpy.data.textures.remove(texture, do_unlink=True)

    # Check for users and remove images
    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image, do_unlink=True)
    

def import_glb(file_path, location=(0, 0, 0), rotation=(0, 0, 0), scale=(0.01, 0.01, 0.01),centering=True):
    if not os.path.exists(file_path):
        return None
    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=file_path)

    # Get the imported object
    imported_object = bpy.context.selected_objects[0]

    # Set the location, rotation, and scale
    # imported_object.location = location
    imported_object.rotation_euler = rotation
    imported_object.scale = scale

    # offset = -imported_object.location
    if centering:
        bpy.context.view_layer.objects.active = imported_object
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    # Apply the offset to keep the specified location
    # imported_object.location += offset
    imported_object.location = location
    return imported_object


def create_wall_mesh(name, vertices):
    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # Link the object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # Make the new object the active object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Enter Edit mode to create the wall geometry
    bpy.ops.object.mode_set(mode='EDIT')

    # Create a BMesh
    bm = bmesh.new()

    # Create the vertices
    for v in vertices:
        bm.verts.new(v)

    # Ensure the lookup table is updated
    bm.verts.ensure_lookup_table()


    # Create the edges between consecutive vertices
    for i in range(len(vertices)-1):
        bm.edges.new([bm.verts[i], bm.verts[i+1]])

    # Create the face (assuming a closed loop)
    bm.faces.new(bm.verts)

    bpy.ops.object.mode_set(mode='OBJECT')


    # Update the mesh with the BMesh data
    bm.to_mesh(mesh)
    bm.free()
    return obj 


def create_cube(name, min_xyz, max_xyz,location,rotate=False):
    # Calculate dimensions of the cube
    print(min_xyz)
    print(max_xyz)
    dimensions = [max_xyz[i] - min_xyz[i] for i in range(3)]

    # Calculate location of the cube
    # location = [(max_xyz[i] + min_xyz[i]) / 2 for i in range(3)]

    # Create the cube object
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    cube = bpy.context.active_object

    # Resize the cube to the specified dimensions
    cube.dimensions = dimensions

    if rotate:
        bpy.context.view_layer.objects.active = cube
        bpy.ops.transform.rotate(value=math.radians(90), orient_axis='Z')


    # cube.location.x += assetPosition[0]
    # cube.location.y += assetPosition[2]
    # cube.location.z += assetPosition[1]
    # Set the cube name
    cube.name = name
    return cube


def is_image_loaded(image_filepath):
    for image in bpy.data.images:
        if image.filepath == image_filepath:
            return image
    return None


def apply_texture_to_object(obj, texture_path):
    # Create a new material
    mat = bpy.data.materials.new(name="TextureMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    # Load the texture image
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)

    # Connect the texture to the BSDF
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def add_material(obj, path_texture, add_uv=False, material_pos=-1, texture_scale=(1.8, 1.8), existing_material_name=None):
    material_id = os.path.basename(path_texture)
    if add_uv:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj

        obj.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        # Unwrap using Smart UV Project
        bpy.ops.uv.smart_project()

        # Switch back to Object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.view_layer.objects.active = obj

    # Check if the material already exists
    existing_material = bpy.data.materials.get(material_id)
    if existing_material:
        new_material = existing_material
    else:
        # Create a new material
        new_material = bpy.data.materials.new(name=material_id)
        new_material.use_nodes = True
        node_tree = new_material.node_tree

        # Clear default nodes
        for node in node_tree.nodes:
            node_tree.nodes.remove(node)

        principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
# Create an image texture node
        image_texture_node = node_tree.nodes.new(type='ShaderNodeTexImage')
        
        resolutions = ["1K", "2K", "12K"]
        for resolution in resolutions:
            if os.path.exists(f"{path_texture}/{material_id}_{resolution}-JPG_Color.jpg"):
                break
        else:
            print("No texture found for the object.")
            return

        image = is_image_loaded(f"{path_texture}/{material_id}_{resolution}-JPG_Color.jpg")
        if image is None:
            image = load_image(f"{path_texture}/{material_id}_{resolution}-JPG_Color.jpg", new_material)
        image_texture_node.image = image

        # Add Texture Coordinate and Mapping nodes for texture scaling
        tex_coord_node = node_tree.nodes.new(type='ShaderNodeTexCoord')
        mapping_node = node_tree.nodes.new(type='ShaderNodeMapping')
        mapping_node.inputs['Scale'].default_value[0] = texture_scale[0]
        mapping_node.inputs['Scale'].default_value[1] = texture_scale[1]

        # Link nodes
        node_tree.links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        node_tree.links.new(mapping_node.outputs['Vector'], image_texture_node.inputs['Vector'])

        # normal
        img_normal = is_image_loaded(f"{path_texture}/{material_id}_{resolution}-JPG_NormalGL.jpg")
        if img_normal is None:
            img_normal = load_image(f"{path_texture}/{material_id}_{resolution}-JPG_NormalGL.jpg", new_material)
        image_texture_node_normal = node_tree.nodes.new(type='ShaderNodeTexImage')
        image_texture_node_normal.image = img_normal    
        image_texture_node_normal.image.colorspace_settings.name = 'Non-Color'

        normal_map_node = node_tree.nodes.new(type='ShaderNodeNormalMap')

        node_tree.links.new(normal_map_node.outputs["Normal"], principled_node.inputs["Normal"])
        node_tree.links.new(image_texture_node_normal.outputs["Color"], normal_map_node.inputs["Color"])

        # rough
        if os.path.exists(f"{path_texture}/{material_id}_{resolution}-JPG_Roughness.jpg"):
            img_rough = is_image_loaded(f"{path_texture}/{material_id}_{resolution}-JPG_Roughness.jpg")
            if img_rough is None:
                img_rough = load_image(f"{path_texture}/{material_id}_{resolution}-JPG_Roughness.jpg", new_material)

            image_texture_node_rough = node_tree.nodes.new(type='ShaderNodeTexImage')
            image_texture_node_rough.image = img_rough    
            image_texture_node_rough.image.colorspace_settings.name = 'Non-Color'

            node_tree.links.new(image_texture_node_rough.outputs["Color"], principled_node.inputs["Roughness"])

        # metal
        if os.path.exists(f"{path_texture}/{material_id}_{resolution}-JPG_Metalness.jpg"):
            img_metal = is_image_loaded(f"{path_texture}/{material_id}_{resolution}-JPG_Metalness.jpg")
            if img_metal is None:
                img_metal = load_image(f"{path_texture}/{material_id}_{resolution}-JPG_Metalness.jpg", new_material)

            image_texture_node_metal = node_tree.nodes.new(type='ShaderNodeTexImage')
            image_texture_node_metal.image = img_metal    
            image_texture_node_metal.image.colorspace_settings.name = 'Non-Color'

            node_tree.links.new(image_texture_node_metal.outputs["Color"], principled_node.inputs["Metallic"])

        # connecting
        node_tree.links.new(image_texture_node.outputs["Color"], principled_node.inputs["Base Color"])
        
        material_output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        node_tree.links.new(principled_node.outputs["BSDF"], material_output_node.inputs["Surface"])

    if existing_material_name is not None:
        # link the part of the object that connects to existing material with the new material
        for slot in obj.material_slots:
            ### case insensitive comparison
            if slot.material and existing_material_name.lower() in slot.material.name.lower():
                # Create a new material slot and assign the new material to it
                obj.data.materials.append(new_material)
                new_slot_index = len(obj.material_slots) - 1
                # Assign the new material slot to the faces that used the old material
                for polygon in obj.data.polygons:
                    if polygon.material_index == obj.material_slots.find(slot.name):
                        polygon.material_index = new_slot_index
    else:
        # Link the material to the object
        if material_pos == -1 or len(obj.data.materials) == 0:
            obj.data.materials.clear()
            obj.data.materials.append(new_material)
        else:
            obj.data.materials[material_pos] = new_material


def load_hdri(hdri_path='./data/HDRIs/studio_small_08_4k.exr', hdri_strength=1, hide=True):  # Replace with the correct path
    # Path to the HDRI file
    # Ensure this path is correct and points to the 'meadow_4k.exr' file on your system

    if not os.path.exists(hdri_path):
        current_file = os.path.abspath(__file__)
        hdri_path = os.path.join(os.path.dirname(os.path.dirname(current_file)), hdri_path)

    # Check if the file exists
    if not os.path.exists(hdri_path):
        print("HDRI file not found:", hdri_path)
    else:
        # Get the world
        world = bpy.data.worlds['World']  # Replace 'World' with your world name if different

        # Enable nodes for the world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear existing nodes (optional, be careful with this)
        # for node in nodes:
        #     nodes.remove(node)

        # Create a new Environment Texture node
        env_texture = nodes.new(type='ShaderNodeTexEnvironment')

        texture_coord = nodes.new(type="ShaderNodeTexCoord")
        # create a new Mapping node
        mapping_node = nodes.new(type='ShaderNodeMapping')

        # Load the HDRI image
        env_texture.image = bpy.data.images.load(hdri_path)

        # Create a Background node
        background = nodes.new(type='ShaderNodeBackground')
        background.location = (-100, 0)

        # Create a World Output node if it doesn't exist
        if 'World Output' not in nodes:
            world_output = nodes.new(type='ShaderNodeOutputWorld')
            world_output.location = (100, 0)
        else:
            world_output = nodes['World Output']

        if hide:
            # Prevent HDRI from showing in the background
            # Add a Light Path node and mix shader to control the visibility
            light_path = nodes.new(type='ShaderNodeLightPath')
            mix_shader = nodes.new(type='ShaderNodeMixShader')
            bg_transparent = nodes.new(type='ShaderNodeBackground')
            bg_transparent.inputs['Color'].default_value = (0, 0, 0, 1)  # Black, fully transparent
            # Link the nodes to use Light Path for mixing
            links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
            links.new(background.outputs['Background'], mix_shader.inputs[1])
            links.new(bg_transparent.outputs['Background'], mix_shader.inputs[2])
            links.new(mix_shader.outputs['Shader'], world_output.inputs['Surface'])
        else:
            links.new(background.outputs['Background'], world_output.inputs['Surface'])

        # Link the nodes
        links.new(texture_coord.outputs['Generated'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], env_texture.inputs['Vector'])
        links.new(env_texture.outputs['Color'], background.inputs['Color'])
        background.inputs['Strength'].default_value = hdri_strength
        print("HDRI background set successfully.")


def setup_camera(center_x, center_y, width, wall_height=.1, wide_lens=False, fov_multiplier=1.1, use_damped_track=False):
    # Check if the camera exists, if not, create one
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
    else:
        cam = bpy.data.objects["Camera"]
    bpy.context.scene.camera = cam

    if wide_lens:
        cam.data.lens /= 2 #35/2
    #cam.data.type = 'ORTHO'

    target_width = abs(fov_multiplier * width)  # Target width to cover
    fov = 2 * math.atan((cam.data.sensor_width / (2 * cam.data.lens)))  # Horizontal FoV calculation
    cam.location.x = center_x
    cam.location.y = center_y
    cam.location.z = wall_height + (target_width / 2) / math.tan(fov / 2)  # Z position calculation

    # clear camera constraints
    cam.constraints.clear()
    if use_damped_track:
        cam_constraint = cam.constraints.new(type="DAMPED_TRACK")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    else:
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"

    empty = bpy.data.objects.new("Empty", None)
    empty.location = (center_x, center_y, 0)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty
    # cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def world_to_camera_view(scene, camera, coord):
    """Convert world coordinates to camera view coordinates"""
    co_local = camera.matrix_world.normalized().inverted() @ coord
    z = -co_local.z

    camera_data = camera.data
    frame = [-v for v in camera_data.view_frame(scene=scene)[:3]]
    if camera_data.type != 'ORTHO':
        frame = [(v / (v.z / z)) for v in frame]

    min_x, max_x = frame[1].x, frame[2].x
    min_y, max_y = frame[0].y, frame[1].y
    x = (co_local.x - min_x) / (max_x - min_x)
    y = (co_local.y - min_y) / (max_y - min_y)
    return Vector((x, y, z))


def get_pixel_coordinates(scene, camera, world_coord):
    """Get pixel coordinates for a given world coordinate"""
    if isinstance(world_coord, np.ndarray) or isinstance(world_coord, list):
        world_coord = Vector(world_coord)
    coord_2d = world_to_camera_view(scene, camera, world_coord)
    return (coord_2d.x, 1 - coord_2d.y)
    #import pdb;pdb.set_trace()
    #render = scene.render
    #return (
    #    round(coord_2d.x * render.resolution_x),
    #    round(coord_2d.y * render.resolution_y)
    #)


def set_rendering_settings(panorama=False, high_res=False) -> None:
    render = bpy.context.scene.render
    render.engine = 'BLENDER_EEVEE'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"

    # Enable CUDA and select all CUDA GPUs
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type == 'CUDA':
                device.use = True
    except:
        print('no CUDA devices found')

    render.resolution_x = 512
    render.resolution_y = 512
    #    bpy.context.scene.cycles.samples = 32
    #    bpy.context.scene.cycles.diffuse_bounces = 1
    #    bpy.context.scene.cycles.glossy_bounces = 1
    #    bpy.context.scene.cycles.transparent_max_bounces = 2
    #    bpy.context.scene.cycles.transmission_bounces = 2

    render.resolution_percentage = 50
    if high_res:
        render.resolution_x = 1080 #1920
        render.resolution_y = 1080
        render.resolution_percentage = 100
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 128

    bpy.context.scene.cycles.diffuse_bounces = 3
    bpy.context.scene.cycles.glossy_bounces = 3
    bpy.context.scene.cycles.transparent_max_bounces = 5
    bpy.context.scene.cycles.transmission_bounces = 5
    bpy.context.scene.cycles.filter_width = 0.01

    bpy.context.scene.cycles.use_denoising = True
    indoor_camera = False
    if indoor_camera:
        bpy.context.scene.render.film_transparent = False
    else:
        bpy.context.scene.render.film_transparent = True


def display_vertex_color():
    # Ensure we are in object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get the active object
    obj = bpy.context.active_object

    # Create a new material
    material = bpy.data.materials.new(name="ColorAttributeMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Clear all nodes to start clean
    for node in nodes:
        nodes.remove(node)

    # Create Principled BSDF and Material Output nodes
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    output = nodes.new(type='ShaderNodeOutputMaterial')

    # Link BSDF to Material Output
    material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Set the location of the nodes
    bsdf.location = (0, 0)
    output.location = (200, 0)

    # Check if the object has a color attribute
    if "Col" in obj.data.attributes:
        # Create an Attribute node and set it to the color attribute
        attr = nodes.new(type='ShaderNodeAttribute')
        attr.attribute_name = "Col"  # Replace 'Col' with the name of your color attribute

        # Link Attribute node to BSDF
        material.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        print("Object has no color attribute.")

    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')

    parser.add_argument(
        '--output', 
        type=str, 
        default='/viscam/u/sgu33/GenLayout/temp_rendering/',
        help='The path the output will be dumped to.'
    )
    parser.add_argument(
        '--json',
        help='path to the json to load from holodeck'
    )

    parser.add_argument(
        '--content',
        help='path to content for loading windows and doors'
    )

    parser.add_argument(
        '--asset_source',
        help='default should be objaverse'
    )

    parser.add_argument(
        '--asset_dir',
        help='path to the asset respository'
    )

    #parser.add_argument(
    #    '--objaverse_path',
    #    help='objaverse path'
    #)

    # output = '/viscam/u/sgu33/GenLayout/temp_rendering/'
    # data_path = '/viscam/projects/GenLayout/3dfront_processed/3dfront/b59e2e3f-d611-4e62-a72a-b3d7ff4bbd08/Bedroom-15614/scene_revised.json'

    # with open(data_path, 'r') as file:
    #     data = json.load(file)
    # task = data["task"]
    # group = data["group"]
    # render_scene(group, task, output, render_top_down=True, render_pano=True, high_res=False, num_side_images=0, phi=45, save_blend=False)
