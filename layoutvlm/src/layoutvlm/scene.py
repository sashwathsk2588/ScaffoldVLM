"""
This scene definition is used to extract constraints from an existing 3D scene.
"""
import math
from shapely import Polygon
import numpy as np
import json
import torch
import torch.nn.functional as F
from .constraints import ALL_CONSTRAINTS
import re 
from collections import defaultdict
from .constraints import Constraint
from .device_utils import get_device_with_index, to_device


def parse_constraint(constraint):
    # Use a regex to match the function name and its arguments
    pattern = re.compile(r"solver\.(\w+)\((.+)\)")
    match = pattern.match(constraint.strip())
    
    if match:
        function_name = match.group(1)
        args_str = match.group(2)
        
        # Split the arguments, keeping commas inside parentheses intact
        args = re.findall(r'\w+\[\d+\]|\([^)]+\)|[\w\.]+(?:=[\d\.]+)?', args_str)
        
        return function_name, args
    return None, None

def generate_loops(constraints):

    constraints = list(set(constraints))
    grouped_by_type = defaultdict(list)
    new_group = defaultdict(list)

    # Group constraints by type and base arguments
    for constraint in constraints:
        constraint_type, args = parse_constraint(constraint)
        
        if constraint_type and args:
            key = tuple(arg.split('[')[0] for arg in args)
            grouped_by_type[(constraint_type, key)].append(args)

    output_constraints = []
    
    sorting_group = defaultdict(list)
    for (constraint_type, key), args_list in grouped_by_type.items():
        # Handle `locate_grid` without any modification
        if constraint_type == "locate_grid":
            for args in args_list:
                output_constraints.append(f"solver.{constraint_type}({', '.join(args)}))")
            continue

        if len(args_list) > 1:
            # Check if we can loop over the first asset
            first_asset_indices = [int(re.search(r'\[(\d+)\]', args[0]).group(1)) for args in args_list]
            if len(set(first_asset_indices)) == len(first_asset_indices) and len(first_asset_indices) == max(first_asset_indices) - min(first_asset_indices) + 1:
                if all(args[1] == args_list[0][1] for args in args_list):
                    additional_args = args_list[0][2:] if len(args_list[0]) > 2 else []
                    loop_text = f"for i in range({min(first_asset_indices)}, {max(first_asset_indices) + 1}):\n\t"
                    loop_text += f"solver.{constraint_type}({key[0]}[i], {args_list[0][1]}"
                    if additional_args:
                        loop_text += f", {', '.join(additional_args)}"
                    loop_text += ")"
                    output_constraints.append(loop_text)
                    new_group[key[0].split('[')[0]].append(loop_text) 
                    continue
            
            # Check if we can loop over the second asset
            second_asset_indices = [int(re.search(r'\[(\d+)\]', args[1]).group(1)) for args in args_list if re.search(r'\[(\d+)\]', args[1])]
            if second_asset_indices and len(set(second_asset_indices)) == len(second_asset_indices) and len(second_asset_indices) == max(second_asset_indices) - min(second_asset_indices) + 1:
                if all(args[0] == args_list[0][0] for args in args_list):
                    additional_args = args_list[0][2:] if len(args_list[0]) > 2 else []
                    loop_text = f"for i in range({min(second_asset_indices)}, {max(second_asset_indices) + 1}):\n\t"
                    loop_text += f"solver.{constraint_type}({args_list[0][0]}, {key[1]}[i]"
                    if additional_args:
                        loop_text += f", {', '.join(additional_args)}"
                    loop_text += ")"
                    output_constraints.append(loop_text)
                    new_group[args_list[0][0].split('[')[0]].append(loop_text) 
                    continue
        
        # If no loop can be generated, add the original constraints
        for args in args_list:
            # Fix any extra brackets and missing parentheses
            cleaned_args = []
            for arg in args:
                # Remove extra closing brackets
                arg = arg.replace(']]', ']')
                cleaned_args.append(arg)
            output_constraints.append(f"solver.{constraint_type}({', '.join(cleaned_args)})")
            new_group[cleaned_args[0].split('[')[0]].append(f"solver.{constraint_type}({', '.join(cleaned_args)})")
    
    final_constraints = []
    for object_name, constraints in new_group.items():
        constraints = sorted(constraints)
        for cons in constraints:
            final_constraints.append(cons)
    

    return final_constraints
    
#class Grid:
#    def __init__(self, grid_id, pos):
#        self.id = f"{grid_id}"
#        #self.pos = pos
#        self.position = torch.tensor([pos[0], pos[1]], requires_grad=True, dtype=torch.float32) #, requires_grad=False
       

class Wall:
    def __init__(self, wall_id, vertices, device=None):
        self.id = wall_id
        self.device = get_device_with_index() if device is None else device
        # Ensure device is available
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.device = 'cpu'
        self.corner1 = vertices[0]
        self.corner2 = vertices[1]
        self.optimize = 0
        self.position = torch.tensor(
            [(self.corner1[0] + self.corner2[0])/2, (self.corner1[1] + self.corner2[1])/2, 0],
            dtype=torch.float32, requires_grad=False, device=self.device
        )
        self.rotation = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False, device=self.device)
        # this is only for distance constraint?
        self.size = None
        self.dimension = torch.tensor(
            [abs(self.corner1[0] - self.corner2[0]), abs(self.corner1[1] - self.corner2[1]), 1],
            dtype=torch.float32, requires_grad=False, device=self.device
        )

    def __str__(self):
        return f"Wall({self.id}, c1={list(self.corner1)}, c2={list(self.corner2)} )"
    
    def get_2dvector(self, add_radian=0):
        vector = torch.tensor([self.corner2[0] - self.corner1[0], self.corner2[1] - self.corner1[1]], dtype=torch.float32).to(self.device)
        _vector = F.normalize(vector, p=2, dim=-1)
        if add_radian:
            # Assuming add_degree is the angle to add in degrees
            theta = torch.tensor(add_radian, dtype=torch.float32)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            # Construct the rotation matrix
            rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                            [sin_theta, cos_theta]], dtype=torch.float32).to(self.device)
            # Multiply the rotation vector by the rotation matrix
            _vector = torch.matmul(rotation_matrix, _vector)
        return _vector


class AssetInstance:
    def __init__(self, id, position, rotation, size, onCeiling=False, optimize=1, device=None):
        self.id = id
        self.size = size
        self.onCeiling = onCeiling
        self.device = get_device_with_index() if device is None else device
        # Ensure device is available
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.device = 'cpu'
        self.optimize = optimize
        if len(position) == 2:
            position = [position[0], position[1], 0]
        if position is None: position = [0, 0, 0]
        if rotation is None: rotation = [0, 0, 0]
        ### the folloiwng two will be updated during optimization 
        self.position = torch.nn.Parameter(
            torch.tensor(list(position), dtype=torch.float32, device=self.device),
            requires_grad=(self.optimize > 0)
        )
        # rotation is given as radians
        # add epsilon to avoid vanishing gradient
        rotation[-1] += math.pi/1000
        self.rotation = torch.nn.Parameter(
            torch.tensor([math.cos(rotation[-1]), math.sin(rotation[-1])], dtype=torch.float32, device=self.device),
            requires_grad=(self.optimize > 0)
        )
        self.raw_z_rotation = rotation[-1]
        assert self.position.is_leaf
        assert self.rotation.is_leaf
        self.dimensions  = torch.tensor(size, dtype=torch.float32, requires_grad=False).to(self.device)

    def __str__(self):
        return f"AssetInstance({self.id}, xyz={self.position.detach().cpu().numpy()}, rot={self.rotation.detach().cpu().numpy()}, dim={self.dimensions.detach().cpu().numpy()}, onCeiling={self.onCeiling})"
    
    def get_theta(self, use_degree=False):
        radian = math.atan2(self.rotation[1].item(), self.rotation[0].item())
        if use_degree:
            return math.degrees(radian)
        return radian

    def get_2dvector(self, add_radian=0):
        _rotation = F.normalize(self.rotation, p=2, dim=-1)
        if add_radian:
            # Assuming add_degree is the angle to add in degrees
            theta = torch.tensor(add_radian, dtype=torch.float32) #* (torch.pi / 180)  # Convert degrees to radians
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            # Construct the rotation matrix
            rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                            [sin_theta, cos_theta]], dtype=torch.float32).to(self.device)
            # Multiply the rotation vector by the rotation matrix
            _rotation = torch.matmul(rotation_matrix, _rotation)
        return _rotation

    def get_2dpolygon(self):
        _rotation = F.normalize(self.rotation, p=2, dim=-1)
       
        cos_theta = _rotation[0] #torch.cos(self.rotation[-1])
        sin_theta = _rotation[1] #torch.sin(self.rotation[-1])
        # Use torch.stack to maintain gradient flow
       

        rotation_matrix = torch.stack([
            torch.stack([cos_theta, sin_theta], dim=-1),
            torch.stack([-sin_theta, cos_theta], dim=-1)
        ])
        
        # Define the local corners of the bounding box
        local_corners = torch.tensor([
            [-self.dimensions[0]/2, -self.dimensions[1]/2],
            [-self.dimensions[0]/2,  self.dimensions[1]/2],
            [ self.dimensions[0]/2,  self.dimensions[1]/2],
            [ self.dimensions[0]/2, -self.dimensions[1]/2]
        ], dtype=rotation_matrix.dtype, requires_grad=False).to(self.device)
        
        # Rotate and translate the corners to get the global coordinates
        #global_corners = torch.matmul(rotation_matrix, local_corners) + self.position[:2]
        global_corners = torch.matmul(local_corners, rotation_matrix) + self.position[:2]
        return global_corners


class Scene:
    def __init__(self, boundary, assets, full_assets, scene_id):
        self.boundary = boundary
        self.assets = assets
        self.full_assets = full_assets
        self.scene_id = scene_id
        self.wall_assets = []

        num_walls = len(boundary)
        for idx in range(num_walls):
            vertices = np.array([
                boundary[idx],
                boundary[(idx+1)%num_walls]
            ]).astype(np.float32)

            self.wall_assets.append(
                Wall(
                    f"walls[{idx}]",
                    vertices=[vertices[0], vertices[1]],
                )
            )


    def extract_constraints(self, dump_path=None, debug=True):
        # point_towards
        program_str = ""
        position_str = ""
        WALL_EPSILON = 0.1
        EPSILON = 0.01
        ON_TOP_OF_EPSILON = -0.1
        assets = self.assets
        num_walls = len(self.wall_assets)
        num_assets = len(self.assets)
        orientation_dict = {}

        with torch.no_grad():
            # if dump_path is not None:
            #     file = open(dump_path + "/constraints_output.txt", "w")
            
            ### wall-related constraints
            for i in range(num_walls):
                wall_asset = self.wall_assets[i]
                for j in range(num_assets):
                    asset = self.assets[j]
                    constraint = ALL_CONSTRAINTS["against_wall"]
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    constraint_value = constraint.evaluate([asset, wall_asset], device=device).item()
                    if debug:
                        print("Wall constraint value between asset {} and wall {}: {}".format(asset.id, wall_asset.id, constraint_value))
                    if constraint.evaluate([asset, wall_asset]).item() < WALL_EPSILON:
                        #self.constraints.append(("against_wall", [i, j]))
                        asset_name = '-'.join(asset.id.split('-')[:-1])
                        asset_idx = asset.id.split('-')[-1]
                        program_str += f"solver.against_wall({asset_name}.placements[{asset_idx}], {wall_asset.id})\n"
                        orientation_dict[j] = True
                        # if dump_path is not None:
                        #     file.write(f"against_wall({asset.id}, {wall_asset.wall_id})\n")

            ### constraints between assets
            for i in range(num_assets):
                
                for j in range(i+1, num_assets):
                    point_constraint = ALL_CONSTRAINTS["point_towards"]
                    asset_name_i = '-'.join(assets[i].id.split('-')[:-1])
                    asset_idx_i = assets[i].id.split('-')[-1]
                    asset_name_j = '-'.join(assets[j].id.split('-')[:-1])
                    asset_idx_j = assets[j].id.split('-')[-1]
                    if point_constraint.evaluate([assets[i], assets[j]]).item() < EPSILON:
                        
                        program_str += f"solver.point_towards({asset_name_i}.placements[{asset_idx_i}], {asset_name_j}.placements[{asset_idx_j}])\n"
                        orientation_dict[i] = True
                        orientation_dict[j] = True

                    if point_constraint.evaluate([assets[j], assets[i]]).item() < EPSILON:
                       
                        program_str += f"solver.point_towards({asset_name_j}.placements[{asset_idx_j}], {asset_name_i}.placements[{asset_idx_i}])\n"
                        orientation_dict[i] = True
                        orientation_dict[j] = True
                        
                    close_to_constraint = ALL_CONSTRAINTS["close_to"]
                    moderate_distance_away_constraint = ALL_CONSTRAINTS["moderate_distance"]
                    if close_to_constraint.evaluate([assets[i], assets[j]]).item() < EPSILON:
                        program_str += f"solver.distance_constraint({asset_name_i}.placements[{asset_idx_i}], {asset_name_j}.placements[{asset_idx_j}], min_distance=0.0, max_distance=1.0)\n"
                    elif moderate_distance_away_constraint.evaluate([assets[i], assets[j]]).item() < EPSILON:
                        program_str += f"solver.distance_constraint({asset_name_i}.placements[{asset_idx_i}], {asset_name_j}.placements[{asset_idx_j}], min_distance=1.0, max_distance=3.0)\n"
                 
                    
                    aligned_constraint = ALL_CONSTRAINTS["align_with"]
                    aligned_with_value = aligned_constraint.evaluate([assets[i], assets[j]]).item()
                    if debug:
                        print("Aligned with constraint value between asset {} and asset {}: {}".format(assets[i].id, assets[j].id, aligned_with_value))
                    if aligned_with_value < EPSILON:
                        program_str += f"solver.align_with({asset_name_i}.placements[{asset_idx_i}], {asset_name_j}.placements[{asset_idx_j}])\n"
                        orientation_dict[i] = True
                        orientation_dict[j] = True

                if i not in orientation_dict.keys():
                    asset_name_i = '-'.join(assets[i].id.split('-')[:-1])
                    asset_idx_i = assets[i].id.split('-')[-1]
                    aligned_constraint = ALL_CONSTRAINTS["align_with"]
                    for w in range(num_walls):
                        wall_asset = self.wall_assets[w]
                        #print(aligned_constraint.evaluate([assets[i], wall_asset]).item())
                        if aligned_constraint.evaluate([assets[i], wall_asset]).item() == 0.0:
                            program_str += f"solver.align_with({asset_name_i}.placements[{asset_idx_i}], {wall_asset.id}])\n"
                            break
                    for w in range(len(self.full_assets)):
                        asset_name_w = '-'.join(self.full_assets[w].id.split('-')[:-1])
                        asset_idx_w = self.full_assets[w].id.split('-')[-1]
                        #print(aligned_constraint.evaluate([assets[i], wall_asset]).item())
                        if aligned_constraint.evaluate([assets[i], self.full_assets[w]]).item() == 0.0 and asset_name_i != asset_name_w:
                            program_str += f"solver.align_with({asset_name_i}.placements[{asset_idx_i}], {asset_name_w}.placements[{asset_idx_w}])\n"
                            #break
                    for w in range(len(self.full_assets)):
                        point_constraint = ALL_CONSTRAINTS["point_towards"]
                        asset_name_w = '-'.join(self.full_assets[w].id.split('-')[:-1])
                        asset_idx_w = self.full_assets[w].id.split('-')[-1]
                        #print(aligned_constraint.evaluate([assets[i], wall_asset]).item())
                        if point_constraint.evaluate([assets[i], self.full_assets[w]]).item() < EPSILON and asset_name_i != asset_name_w:
                            program_str += f"solver.point_towards({asset_name_i}.placements[{asset_idx_i}],  {asset_name_w}.placements[{asset_idx_w}])\n"
                            #break
                        if point_constraint.evaluate([self.full_assets[w], assets[i]]).item() < EPSILON and asset_name_i != asset_name_w:
                            program_str += f"solver.point_towards({asset_name_w}.placements[{asset_idx_w}],  {asset_name_i}.placements[{asset_idx_i}])\n"
                            #break

            for i in range(num_assets):
                
                for j in range(len(self.full_assets)):
                    on_constraint = ALL_CONSTRAINTS["on_top_of"]
                    asset_i_id = assets[i].id
                    asset_j_id = self.full_assets[j].id
                    asset_name_i = '-'.join(assets[i].id.split('-')[:-1])
                    asset_idx_i = assets[i].id.split('-')[-1]
                    asset_name_j = '-'.join(self.full_assets[j].id.split('-')[:-1])
                    asset_idx_j = self.full_assets[j].id.split('-')[-1]

                    i_on_top_of_j = on_constraint.evaluate([assets[i], self.full_assets[j]]).item()
                    j_on_top_of_i = on_constraint.evaluate([self.full_assets[j], assets[i]]).item()
                    if debug:
                        print("On top of constraint value between asset {} and asset {}: {}".format(assets[i].id, self.full_assets[j].id, i_on_top_of_j))
                        print("On top of constraint value between asset {} and asset {}: {}".format(self.full_assets[j].id, assets[i].id, j_on_top_of_i))

                    if i_on_top_of_j < ON_TOP_OF_EPSILON and asset_i_id != asset_j_id:
                        program_str += f"solver.on_top_of({asset_name_i}.placements[{asset_idx_i}], {asset_name_j}.placements[{asset_idx_j}])\n"
                    
                    if j_on_top_of_i < ON_TOP_OF_EPSILON and asset_i_id != asset_j_id:
                        program_str += f"solver.on_top_of({asset_name_j}.placements[{asset_idx_j}], {asset_name_i}.placements[{asset_idx_i}])\n"
                   
            ### constraints between floor grid and assets
            for i in range(num_assets):
                #grid = find_nearest_grid(assets[i], self.grids_dict)
                asset_name_i = '-'.join(assets[i].id.split('-')[:-1])
                asset_idx_i = assets[i].id.split('-')[-1]
                if abs(assets[i].position[2] - assets[i].size[2] / 2) < 0.1 :
                    position_str += f"{asset_name_i}.placements[{asset_idx_i}].position = [{assets[i].position[0]:.1f}, {assets[i].position[1]:.1f}, {asset_name_i}.size[2]/2]\n"
                else:
                    position_str += f"{asset_name_i}.placements[{asset_idx_i}].position = [{assets[i].position[0]:.1f}, {assets[i].position[1]:.1f}, {assets[i].position[2]:.1f}]\n"
                    

        # print("----original-----")
        # print(position_str + program_str)
        constraints = program_str.strip().splitlines()
    
        sorted_constraints = sorted(constraints)
        
        sorted_program_str = "\n".join(sorted_constraints)
        
        optimized_constraints = generate_loops(sorted_program_str.strip().split("\n"))
        for_loop = "\n".join(optimized_constraints)
        final_str = position_str + for_loop + '\n'
        final_str = final_str.replace('\n\n', '\n')
        # print("----for loop-----")
        # print(final_str)
        return final_str





    def extract_constraints_v2(self, visual_mark_instances, debug=False, use_rotation_for_optimization=False):
        """
        We will extract constraints given groundtruth layout

        We will extract the following constraints: against_wall, distance_constraint, on_top_of, align_with, point_towards

        Definition:
        against_wall: between an asset and a wall
        distance_constraint: between two assets, or between an asset and a visual mark. This is deterministic. We use close (0-1m) and moderate (1-3m) distance constraints
        on_top_of: between two assets. We will the non-differentiable version of this constraint, which check z values.
        align_with: between two assets
        point_towards: between two assets, or between an asset and a visual mark

        :param visual_mark_instances:
        :param debug:
        :return:
        """


        WALL_EPSILON = 0.1
        EPSILON = 0.01
        ON_TOP_OF_EPSILON = -0.1
        assets = self.assets
        num_walls = len(self.wall_assets)
        num_assets = len(self.assets)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        orientation_dict = {}

        program_str = ""
        position_str = ""
        wall_constraint = ALL_CONSTRAINTS["against_wall"]
        point_constraint = ALL_CONSTRAINTS["point_towards"]
        close_to_constraint = ALL_CONSTRAINTS["close_to_deterministic"]
        moderate_distance_away_constraint = ALL_CONSTRAINTS["moderate_distance_deterministic"]
        aligned_constraint = ALL_CONSTRAINTS["align_with"]
        on_constraint = ALL_CONSTRAINTS["on_top_of_deterministic"]
        current_id_list = []
        with torch.no_grad():
            # if dump_path is not None:
            #     file = open(dump_path + "/constraints_output.txt", "w")

            for i in range(num_assets):

                asset_i = self.assets[i]
                asset_name_i = asset_i.id.split("[")[0]
                asset_idx_i = asset_i.id.split("[")[1].split("]")[0]
                current_id_list.append(asset_i.id)

                # against wall
                for j in range(num_walls):
                    wall_asset = self.wall_assets[j]
                    
                    constraint_value = wall_constraint.evaluate([asset_i, wall_asset]).item()
                    if debug:
                        print("Wall constraint value between asset {} and wall {}: {}".format(asset_i.id, wall_asset.id, constraint_value))
                    if constraint_value < WALL_EPSILON:
                        program_str += f"solver.against_wall({asset_name_i}[{asset_idx_i}], {wall_asset.id})\n"
                        orientation_dict[i] = True

                # between an asset and a visual mark
                if visual_mark_instances!= []:
                    visual_mark_to_distance = {}
                    for visual_mark_instance in visual_mark_instances:
                        # distance
                        close_to_constraint = ALL_CONSTRAINTS["close_to_deterministic"]
                        constraint_value, distance = close_to_constraint.evaluate([asset_i, visual_mark_instance])
                        if constraint_value.item() < EPSILON:
                            visual_mark_to_distance[visual_mark_instance] = distance

                        # point towards
                        point_constraint = ALL_CONSTRAINTS["point_towards"]
                        if point_constraint.evaluate([assets[i], visual_mark_instance]).item() < EPSILON:
                            program_str += f"solver.point_towards({asset_name_i}[{asset_idx_i}], {visual_mark_instance.id}])\n"

                    # find the closest visual mark
                    if visual_mark_to_distance:
                        visual_mark_instance = min(visual_mark_to_distance, key=visual_mark_to_distance.get)
                        program_str += f"solver.distance_constraint({asset_name_i}[{asset_idx_i}], {visual_mark_instance.id}, min_distance=0.0, max_distance=1.0)\n"

                # between assets
                for j in range(num_assets):

                    if i == j:
                        continue

                    asset_name_j = assets[j].id.split("[")[0]
                    asset_idx_j = assets[j].id.split("[")[1].split("]")[0]

                    # point towards
                    if point_constraint.evaluate([assets[i], assets[j]]).item() < EPSILON:
                        print(point_constraint.evaluate([assets[i], assets[j]]).item())
                        program_str += f"solver.point_towards({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        orientation_dict[i] = True
                        orientation_dict[j] = True

                    # distance
                    if close_to_constraint.evaluate([assets[i], assets[j]])[0].item() < EPSILON:
                        program_str += f"solver.distance_constraint({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}], min_distance=0.0, max_distance=1.0)\n"
                    elif moderate_distance_away_constraint.evaluate([assets[i], assets[j]])[0].item() < EPSILON:
                        program_str += f"solver.distance_constraint({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}], min_distance=1.0, max_distance=3.0)\n"

                    # align with
                    aligned_with_value = aligned_constraint.evaluate([assets[i], assets[j]]).item()
                    if debug:
                        print("Aligned with constraint value between asset {} and asset {}: {}".format(assets[i].id,
                                                                                                       assets[j].id,
                                                                                                       aligned_with_value))
                    if aligned_with_value < EPSILON:
                        program_str += f"solver.align_with({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        orientation_dict[i] = True
                        orientation_dict[j] = True

                    # on top of
                    # if i == 2 and j == 0:
                    #     import pdb; pdb.set_trace()
                    on_top_of_value = on_constraint.evaluate([assets[i], assets[j]]).item()
                    if debug:
                        print("On top of constraint value between asset {} and asset {}: {}".format(assets[i].id,
                                                                                                    assets[j].id,
                                                                                                    on_top_of_value))
                    if on_top_of_value < ON_TOP_OF_EPSILON:
                        program_str += f"solver.on_top_of({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"

            for i in range(num_assets):
                asset_i = self.assets[i]
                asset_name_i = asset_i.id.split("[")[0]
                asset_idx_i = asset_i.id.split("[")[1].split("]")[0]
                for k in range(len(self.full_assets)):
                    asset_k = self.full_assets[k]
                    asset_name_k = asset_k.id.split("[")[0]
                    asset_idx_k = asset_k.id.split("[")[1].split("]")[0]
                    # add orientation constraints for assets that are not rotationally constrained to assets within its group
                    if i not in orientation_dict and asset_k.id not in current_id_list:
                        #print(asset_k)
                        aligned_with_value = aligned_constraint.evaluate([asset_i, asset_k]).item()
                        if aligned_with_value < EPSILON:
                            program_str += f"solver.align_with({asset_name_i}[{asset_idx_i}], {asset_name_k}[{asset_idx_k}])\n"
                        if point_constraint.evaluate([asset_i, asset_k]).item() < EPSILON:
                            program_str += f"solver.point_towards({asset_name_i}[{asset_idx_i}], {asset_name_k}[{asset_idx_k}])\n"
                    # add on top of constraints between assets in different groups
                    elif asset_k.id not in current_id_list:
                        #print(asset_k)
                        if on_constraint.evaluate([asset_i, asset_k]).item() < ON_TOP_OF_EPSILON:
                            program_str += f"solver.on_top_of({asset_name_i}[{asset_idx_i}], {asset_name_k}[{asset_idx_k}])\n"
                        elif on_constraint.evaluate([asset_k, asset_i]).item() < ON_TOP_OF_EPSILON:
                            program_str += f"solver.on_top_of({asset_name_k}[{asset_idx_k}], {asset_name_i}[{asset_idx_i}])\n"
                    
            # exact position and orientation
            for i in range(num_assets):
                asset_i = self.assets[i]
                # grid = find_nearest_grid(assets[i], self.grids_dict)
                asset_name_i = asset_i.id.split("[")[0]
                asset_idx_i = asset_i.id.split("[")[1].split("]")[0]
                
                position_str += f"{asset_name_i}[{asset_idx_i}].position = [{assets[i].position[0]:.1f}, {assets[i].position[1]:.1f}, {assets[i].position[2]:.1f}]\n"
                if use_rotation_for_optimization:
                    position_str += f"{asset_name_i}[{asset_idx_i}].rotation = [0, 0, {assets[i].raw_z_rotation}]\n"
                else:
                    position_str += f"{asset_name_i}[{asset_idx_i}].rotation = [{round(np.rad2deg(assets[i].raw_z_rotation))}]\n"


        # print("----original-----")
        # # print(position_str + program_str)
        # print(program_str)
        
        constraints = program_str.strip().splitlines()
        sorted_program_str = "\n".join(constraints)
        optimized_constraints = generate_loops(sorted_program_str.strip().split("\n"))
        #print(optimized_constraints)
        for_loop = "\n".join(optimized_constraints)
        final_str = position_str + for_loop + '\n'
        final_str = final_str.replace('\n\n', '\n')
        print("----for loop-----")
        print(final_str)
        print("----for loop end-----")
        return final_str




    def extract_simple_constraints(self, visual_mark_instances, debug=False):

        WALL_EPSILON = 0.1
        EPSILON = 0.01
        ON_TOP_OF_EPSILON = -0.1
        assets = self.assets
        num_walls = len(self.wall_assets)
        num_assets = len(self.assets)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        program_str = ""
        position_str = ""
        wall_constraint = ALL_CONSTRAINTS["against_wall"]
        point_constraint = ALL_CONSTRAINTS["point_towards"]
        close_to_constraint = ALL_CONSTRAINTS["close_to_deterministic"]
        moderate_distance_away_constraint = ALL_CONSTRAINTS["moderate_distance_deterministic"]
        aligned_constraint = ALL_CONSTRAINTS["align_with"]
        on_constraint = ALL_CONSTRAINTS["on_top_of_deterministic"]
        full_constraints = {}
        with torch.no_grad():
     
            for i in range(num_assets):

                asset_i = self.assets[i]
                asset_name_i = asset_i.id.split("[")[0]
                asset_idx_i = asset_i.id.split("[")[1].split("]")[0]

                
            

                # against wall
                for j in range(num_walls):
                    if (asset_i.id,"wall") not in full_constraints:
                        full_constraints[(asset_i,"wall")] = {}
                    wall_asset = self.wall_assets[j]
                    
                    constraint_value = wall_constraint.evaluate([asset_i, wall_asset]).item()
                    if debug:
                        print("Wall constraint value between asset {} and wall {}: {}".format(asset_i.id, wall_asset.id, constraint_value))
                    if constraint_value < WALL_EPSILON:
                        full_constraints[(i,"wall")]["against_wall"] = f"solver.against_wall({asset_name_i}[{asset_idx_i}], {wall_asset.id})\n"
                        program_str += f"solver.against_wall({asset_name_i}[{asset_idx_i}], {wall_asset.id})\n"
                        
                # between assets
                for j in range(num_assets):

                    if i == j:
                        continue

                    if (i,j) not in full_constraints:
                        full_constraints[(i,j)] = {}

                    asset_name_j = assets[j].id.split("[")[0]
                    asset_idx_j = assets[j].id.split("[")[1].split("]")[0]

                    # point towards
                    if point_constraint.evaluate([assets[i], assets[j]]).item() < EPSILON:
                        #print(point_constraint.evaluate([assets[i], assets[j]]).item())
                        full_constraints[(i,j)]["point_towards"] = f"solver.point_towards({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        program_str += f"solver.point_towards({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        continue
                        
                    aligned_with_value = aligned_constraint.evaluate([assets[i], assets[j]]).item()
                    if debug:
                        print("Aligned with constraint value between asset {} and asset {}: {}".format(assets[i].id,
                                                                                                       assets[j].id,
                                                                                                       aligned_with_value))
                    if aligned_with_value < EPSILON:
                        
                        full_constraints[(i,j)]["align_with"] = f"solver.align_with({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        program_str += f"solver.align_with({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        continue
                       

                    # on top of
                    on_top_of_value = on_constraint.evaluate([assets[i], assets[j]]).item()
                    if debug:
                        print("On top of constraint value between asset {} and asset {}: {}".format(assets[i].id,
                                                                                                    assets[j].id,
                                                                                                    on_top_of_value))
                    if on_top_of_value < ON_TOP_OF_EPSILON:
                        full_constraints[(i,j)]["on_top_of"] = f"solver.on_top_of({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"
                        program_str += f"solver.on_top_of({asset_name_i}[{asset_idx_i}], {asset_name_j}[{asset_idx_j}])\n"


            new_constraints = {}
            for idx, constraints in full_constraints.items():
                for constraint, constraint_str in constraints.items():
                    i, j = idx
                    
                    
                    if j == "wall":
                        asset_i = self.assets[i]
                        if (asset_i.id,"wall") not in new_constraints:
                            new_constraints[(asset_i.id, "wall")] = {}
                        position_str = f"{asset_i.id}.position = [{asset_i.position[0]:.1f}, {asset_i.position[1]:.1f}, {asset_i.position[2]:.1f}]\n"
                        position_str += f"{asset_i.id}.rotation = [{round(np.degrees(np.arccos(asset_i.rotation[0].cpu().numpy())))}]\n"
                        new_constraints[(asset_i.id, "wall")][constraint] = position_str + constraint_str
                    else:
                        asset_i = self.assets[i]
                        asset_j = self.assets[j]
                        if (asset_i.id, asset_j.id) not in new_constraints:
                            new_constraints[(asset_i.id,asset_j.id)] = {}
                        position_str = f"{asset_i.id}.position = [{asset_i.position[0]:.1f}, {asset_i.position[1]:.1f}, {asset_i.position[2]:.1f}]\n"
                        position_str += f"{asset_i.id}.rotation = [{round(np.degrees(np.arccos(asset_i.rotation[0].cpu().numpy())))}]\n"
                        position_str += f"{asset_j.id}.position = [{asset_j.position[0]:.1f}, {asset_j.position[1]:.1f}, {asset_j.position[2]:.1f}]\n"
                        position_str += f"{asset_j.id}.rotation = [{round(np.degrees(np.arccos(asset_j.rotation[0].cpu().numpy())))}]\n"
                        
                        new_constraints[(asset_i.id, asset_j.id)][constraint] = position_str + constraint_str
            # # ### constraints between floor grid and assets
            # for i in range(num_assets):
            #     asset_i = self.assets[i]
            #     # grid = find_nearest_grid(assets[i], self.grids_dict)
            #     asset_name_i = asset_i.id.split("[")[0]
            #     asset_idx_i = asset_i.id.split("[")[1].split("]")[0]
                
            #     position_str += f"{asset_name_i}[{asset_idx_i}].position = [{assets[i].position[0]:.1f}, {assets[i].position[1]:.1f}, {assets[i].position[2]:.1f}]\n"
                
            #     position_str += f"{asset_name_i}[{asset_idx_i}].rotation = [{round(np.degrees(np.arccos(assets[i].rotation[0].cpu().numpy())))}]\n"


       
        #print(new_constraints)
        
    
        return new_constraints

        
def main():
    # Define the boundary of the scene
    device = "cuda" if torch.cuda.is_available() else "cpu"

    boundary = [
        [18.55171012878418, -11.003060340881348, 0.0],
        [22.67064094543457, -11.003060340881348, 0.0],
        [22.67064094543457, -2.5631699562072754, 0.0],
        [18.55171012878418, -2.5631699562072754, 0.0]
    ]

    # Example assets with positions, rotations, and sizes
    asset1 = AssetInstance(
        id="dining_table_A[0]",
        position=[20.0, -7.5, 0.8],
        rotation=[0, 0, 0],
        size=[2.0, 2.0, 1.0],
        optimize=1,
        device=device
    )
    
    asset2 = AssetInstance(
        id="dining_chair[0]",
        position=[19, -7, 0.4],
        rotation=[0, 0, 0],
        size=[1.0, 1.0, 1.0],
        optimize=1,
        device=device
    )

    asset3 = AssetInstance(
        id="dining_chair[1]",
        position=[19, -8, 0.4],
        rotation=[0, 0, 0],
        size=[1.0, 1.0, 1.0],
        optimize=1,
        device=device
    )

    asset4 = AssetInstance(
        id="dining_chair[2]",
        position=[21, -7, 0.4],
        rotation=[0, 0, np.pi],
        size=[1.0, 1.0, 1.0],
        optimize=1,
        device=device
    )

    asset5 = AssetInstance(
        id="dining_chair[3]",
        position=[21, -8, 0.4],
        rotation=[0, 0, np.pi],
        size=[1.0, 1.0, 1.0],
        optimize=1,
        device=device
    )


    asset6 = AssetInstance(
        id="table[0]",
        position=[19.0, -7.0, 0.8],
        rotation=[0, 0, 0],
        size=[0.5, 0.5, 1.0],
        optimize=1,
        device=device
    )

    asset7 = AssetInstance(
        id="lamp[0]",
        position=[19.0, -7.0, 2.4],
        rotation=[0, 0, 0],
        size=[2.0, 1.0, 1.0],
        optimize=1,
        device=device
    )

    # Add assets to a list
    assets = [asset1, asset2, asset3, asset4, asset5]
    
    full_assets = [asset1, asset2, asset3, asset4, asset5, asset6, asset7]

    # Create a scene with the boundary and assets
    scene = Scene(boundary, assets, full_assets, "example_scene")
    visual_mark_instances = []

    # Extract constraints and print them
    constraint_program = scene.extract_simple_constraints(visual_mark_instances)
    print(constraint_program)

if __name__ == "__main__":
    main()
