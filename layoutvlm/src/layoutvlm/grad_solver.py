import json
import collections
import random
from shapely import Polygon
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from utils.placement_utils import get_random_placement
from utils.plot_utils import visualize_grid
import imageio.v2 as imageio
import numpy as np
import os
import torch
import torch.nn.functional as F
from .scene import AssetInstance, Wall
from .constraints import bbox_overlap_loss, Constraint, against_wall, align_with, point_towards, distance_constraint
import re
from scipy.optimize import NonlinearConstraint
from .device_utils import get_device_with_index, to_device



class GradSolver:
    def __init__(self, boundary, solver_type="default"):
        """
        grad solver only maintains the following states:
        - self.boundary: the boundary of the room
        - self.on_top_of_assets: a list of asset pairs that are on top of each other

        Note: boudaries should be given in counterclockwise order
        """
        self.device = get_device_with_index()
        # Ensure device is available
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            self.device = 'cpu'
        self.boundary = boundary
        self.solver_type = solver_type
        self.on_top_of_assets = []
    
    def is_fixture(self, instance_id):
        return instance_id.startswith('walls') or instance_id.startswith('fixed_point') or instance_id == "room_0"


    def project_back_to_polygon(self, existing_constraints):
        MAX_ATTEMPTS = 3
        EPSILON = 0.01
        ### project all assets back to the boundary
        with torch.no_grad():
            polygon_shapely = Polygon(self.boundary)
            assert polygon_shapely.is_valid
            for instance_id, asset in self.solver_assets.items():
                if self.is_fixture(instance_id):
                    continue

                corners = asset.get_2dpolygon().cpu().detach().numpy()
                # check if there ia any nan in the corners
                if np.isnan(corners).any():
                    raise ValueError(f"NaN found in corners: {instance_id}, {corners}")
                # assert there is no nan in the corners
                attempt = 0
                while not all([polygon_shapely.buffer(EPSILON).contains(Point(corners[i])) for i in range(corners.shape[0])]):
                    for i in range(corners.shape[0]):
                        point = Point(corners[i])
                        if not polygon_shapely.buffer(EPSILON).contains(point):
                            projected_point = polygon_shapely.exterior.interpolate(polygon_shapely.exterior.project(point))
                            assert polygon_shapely.buffer(EPSILON).contains(projected_point)
                            translation = [projected_point.x - point.x, projected_point.y - point.y]
                            asset.position[:2] += torch.tensor(translation, dtype=torch.float32).to(self.device)
                            break
                    corners = asset.get_2dpolygon().cpu().detach().numpy()
                    attempt += 1
                    if attempt > MAX_ATTEMPTS:
                        break
            ### objects default on the ground
            for _, asset in self.solver_assets.items():
                if asset.optimize:
                    if asset.onCeiling:
                        asset.position[2] = torch.tensor(3, dtype=torch.float32).to(self.device)
                    else:
                        asset.position[-1] = asset.size[2]/2
            ### project a list of 2d coordinates to a polygon
            def _project(polygon, points):
                polygon_shapely = Polygon(polygon)
                points = [Point(point) for point in points]
                for i in range(len(points)):
                    point = points[i]
                    if not polygon_shapely.buffer(EPSILON).contains(point):
                        points[i] = polygon_shapely.exterior.interpolate(polygon_shapely.exterior.project(point))
                assert all([polygon_shapely.buffer(EPSILON).contains(point) for point in points])
                return [[point.x, point.y] for point in points]

            ### project objects to be "on_top_of" and make sure object center on top of the other object
            for constraint, instance_ids in existing_constraints:
                if constraint.constraint_name == "on_top_of":
                    # TODO: ideally this would project the asset to be on top of the other asset based on the mesh
                    asset_above = self.solver_assets[instance_ids[0]]
                    asset_below = self.solver_assets[instance_ids[1]]
                    #floor_z = self.boundary[0][-1]
                    #if asset_below.optimize and asset_below.position[2]-asset_below.size[2]/2 < floor_z:
                    #    asset_below.position[-1] = floor_z + asset_below.size[2]/2
                    if asset_above.optimize:
                        asset_above.position[-1] = asset_below.position[2] + asset_below.size[2]/2 + asset_above.size[2]/2
                        new_point = _project(asset_below.get_2dpolygon().cpu().numpy(), [asset_above.position[:2].cpu().numpy()])
                        asset_above.position[:2] = torch.tensor(new_point[0], dtype=torch.float32).to(self.device)



    def calc_loss(self, existing_constraints, new_constraints):
        ### default no overlap loss
        # all the assets that are not in self.on_top_of_assets
        import time

        def profile_time(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                # print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
                return result
            return wrapper

        @profile_time
        def calculate_overlap_loss():
            nonoverlap_assets = [asset for asset in self.solver_assets.values() if not self.is_fixture(asset.id)]
            _, iou = bbox_overlap_loss(nonoverlap_assets, only_consider_overlapping_assets=True, skipped_asset_pairs=self.on_top_of_assets, device=self.device)
            return (iou * 1000).to(self.device)

        @profile_time
        def evaluate_existing_constraints(existing_constraints):
            total_loss = 0
            for constraint, instance_ids in existing_constraints:
                assets = [self.solver_assets[instance_id] for instance_id in instance_ids]
                for asset in assets:
                    if isinstance(asset, torch.Tensor) and not asset.requires_grad:
                        raise RuntimeError(f"Asset tensor {asset} does not require gradients.")
                if not constraint.evaluate(assets).requires_grad:
                    raise RuntimeError("asset tensor does not require gradients.")
                total_loss += 100 * constraint.evaluate(assets)
            return total_loss

        @profile_time
        def evaluate_new_constraints(new_constraints):
            total_loss = 0
            for constraint, instance_ids in new_constraints:
                assets = [self.solver_assets[instance_id] for instance_id in instance_ids]
                for asset in assets:
                    if isinstance(asset, torch.Tensor) and not asset.requires_grad:
                        raise RuntimeError(f"Asset tensor {asset} does not require gradients.")
                if not constraint.evaluate(assets).requires_grad:
                    raise RuntimeError(f"Constraint.evaluate {constraint.constraint_name} does not require gradients.")
                try:
                    total_loss += constraint.evaluate(assets, device=self.device)
                except Exception as e:
                    print(f"Error evaluating constraint {constraint.constraint_name} with instance_ids {instance_ids}: {e}")
                    import pdb;pdb.set_trace()
            return total_loss

        overlap_loss = calculate_overlap_loss()
        existing_constraint_loss = evaluate_existing_constraints(existing_constraints)
        new_constraint_loss = evaluate_new_constraints(new_constraints)

        return overlap_loss, existing_constraint_loss, new_constraint_loss
    
    def annealing_objective_function(self, params, bounds, existing_constraints, new_constraints, params_dict):
        for instance_id, param_dict in params_dict.items():
            position_idx = param_dict["position"]
            rotation_idx = param_dict["rotation"]
            self.solver_assets[instance_id].position.data = torch.tensor(params[position_idx:position_idx+3], dtype=torch.float32, device=self.device)
            self.solver_assets[instance_id].rotation.data = torch.tensor(params[rotation_idx:rotation_idx+2], dtype=torch.float32, device=self.device)
        loss = self.calc_loss(existing_constraints, new_constraints)
        # print(f"Loss: {loss.item()}")
        return loss.item()

    def scipy_minimize_objective_function(self, params, bounds, existing_constraints, new_constraints, params_dict):
        for instance_id, param_dict in params_dict.items():
            position_idx = param_dict["position"]
            rotation_idx = param_dict["rotation"]
            self.solver_assets[instance_id].position.data = torch.tensor(params[position_idx:position_idx+3], dtype=torch.float32, device=self.device)
            self.solver_assets[instance_id].rotation.data = torch.tensor(params[rotation_idx:rotation_idx+2], dtype=torch.float32, device=self.device)
        loss = self.calc_loss(existing_constraints, new_constraints)
        # print(f"Loss: {loss.item()}")
        return loss.item()

    def optimize(self, assets, existing_constraints, new_constraints, iterations=400, learning_rate=0.01, temp_dir=None, output_gif_path=None):
        if len(assets) == 0:
            return {}
        if temp_dir: 
            os.makedirs(temp_dir, exist_ok=True)

        ### setup parameters
        self.solver_assets = assets
        all_constraints = existing_constraints + new_constraints
        self.project_back_to_polygon(all_constraints)
        # Check for NaN values in position and rotation
        
        saved_intermediate_states = {
            "boundary": self.boundary,
            "solver_assets": []
        }
        progress_bar = tqdm(total=iterations, desc="optimization progress")
        visualize_grid(self.boundary, self.solver_assets, os.path.join(temp_dir, f"frame_{progress_bar.n:04d}.png"))
        progress_bar.update(1)

        if self.solver_type == "minimize":
            from scipy.optimize import minimize

            x_values = [point[0] for point in self.boundary]
            y_values = [point[1] for point in self.boundary]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            bounds = []
            initial_params = []
            constraints = []
            params_dict = collections.defaultdict(dict)
            for instance_id, asset in self.solver_assets.items():
                if asset.optimize and asset.position.requires_grad:
                    params_dict[instance_id]["position"] = len(initial_params)
                    initial_params.extend(asset.position.cpu().detach().numpy().tolist())
                    sz = min(asset.size[:3])
                    bounds.extend([(x_min+sz/2, x_max-sz/2), (y_min+sz/2, y_max-sz/2), (0+asset.size[2]/2, 10)])
                    #num = len(initial_params)-3
                    #constraints.append(NonlinearConstraint(
                    #    lambda x: x[num],
                    #    lb=0,
                    #    ub=0
                    #))

                if asset.optimize and asset.rotation.requires_grad:
                    params_dict[instance_id]["rotation"] = len(initial_params)
                    initial_params.extend(asset.rotation.cpu().detach().numpy().tolist())
                    bounds.extend([(-1, 1), (-1, 1)])
                    rotation_idx = len(initial_params)-2
                    constraints.append(NonlinearConstraint(
                        lambda x, idx=rotation_idx: x[idx]**2 + x[idx+1]**2,  # sin²(θ) + cos²(θ)
                        lb=1.0,  # Must equal 1
                        ub=1.0
                    ))

            assert len(initial_params) == len(bounds)
            # Callback function to update progress bar
            # Callback function to update progress bar and print the loss
            def progress_callback(xk, state):
                # restore self.solver_assets from x using params_dict
                for instance_id, param_dict in params_dict.items():
                    position_idx = param_dict["position"]
                    rotation_idx = param_dict["rotation"]
                    self.solver_assets[instance_id].position.data = torch.tensor(xk[position_idx:position_idx+3], dtype=torch.float32, device=self.device)
                    self.solver_assets[instance_id].rotation.data = torch.tensor(xk[rotation_idx:rotation_idx+2], dtype=torch.float32, device=self.device)
                progress_bar.update(1)
                visualize_grid(self.boundary, self.solver_assets, os.path.join(temp_dir, f"frame_{progress_bar.n:04d}.png"))
                print(f"Step {progress_bar.n}, Loss: {state.fun}")  # Output the current loss (f)
                return False

            result = minimize(
                self.scipy_minimize_objective_function,
                x0=initial_params,
                bounds=bounds,
                constraints=constraints,
                args=(bounds, existing_constraints, new_constraints, params_dict),
                callback=progress_callback,
                method='trust-constr',
                options = {
                    'maxiter': iterations,
                    'disp': True,
                }
            )
            # restore self.solver_assets from result.x
            for instance_id, param_dict in params_dict.items():
                position_idx = param_dict["position"]
                rotation_idx = param_dict["rotation"]
                self.solver_assets[instance_id].position.data = torch.tensor(result.x[position_idx:position_idx+3], dtype=torch.float32, device=self.device)
                self.solver_assets[instance_id].rotation.data = torch.tensor(result.x[rotation_idx:rotation_idx+2], dtype=torch.float32, device=self.device)
                print(f"Instance {instance_id}, position: {self.solver_assets[instance_id].position}, rotation: {self.solver_assets[instance_id].rotation}")
            visualize_grid(self.boundary, self.solver_assets, os.path.join(temp_dir, f"frame_final.png"))

        elif self.solver_type == "dual_annealing":
            ###############################################################################################################################################
            ### dual annealing
            ###############################################################################################################################################
            from scipy.optimize import dual_annealing

            x_values = [point[0] for point in self.boundary]
            y_values = [point[1] for point in self.boundary]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            bounds = []
            initial_params = []
            params_dict = collections.defaultdict(dict)
            for instance_id, asset in self.solver_assets.items():
                if asset.optimize and asset.position.requires_grad:
                    params_dict[instance_id]["position"] = len(initial_params)
                    initial_params.extend(asset.position.cpu().detach().numpy().tolist())
                    bounds.extend([(x_min, x_max), (y_min, y_max), (0, 10)])

                if asset.optimize and asset.rotation.requires_grad:
                    params_dict[instance_id]["rotation"] = len(initial_params)
                    initial_params.extend(asset.rotation.cpu().detach().numpy().tolist())
                    bounds.extend([(-1, 1), (-1, 1)])

            assert len(initial_params) == len(bounds)
            # Callback function to update progress bar
            max_iterations = iterations
            progress_bar = tqdm(total=max_iterations, desc="Dual Annealing Progress")
            # Callback function to update progress bar and print the loss
            def progress_callback(x, f, context):
                # restore self.solver_assets from x using params_dict
                for instance_id, param_dict in params_dict.items():
                    position_idx = param_dict["position"]
                    rotation_idx = param_dict["rotation"]
                    self.solver_assets[instance_id].position.data = torch.tensor(x[position_idx:position_idx+3], dtype=torch.float32, device=self.device)
                    self.solver_assets[instance_id].rotation.data = torch.tensor(x[rotation_idx:rotation_idx+2], dtype=torch.float32, device=self.device)
                visualize_grid(self.boundary, self.solver_assets, os.path.join(temp_dir, f"frame_{progress_bar.n:04d}.png"))
                progress_bar.update(1)
                print(f"Step {progress_bar.n}, Loss: {f}")  # Output the current loss (f)
                if context == 2:  # Dual annealing complete
                    progress_bar.close()
                    return True
                return False

            result = dual_annealing(
                self.annealing_objective_function,
                bounds=bounds,
                args=(bounds, existing_constraints, new_constraints, params_dict),
                maxiter=max_iterations,
                seed=42,
                x0=initial_params,
                callback=progress_callback
            )
            # restore self.solver_assets from result.x
            for instance_id, param_dict in params_dict.items():
                position_idx = param_dict["position"]
                rotation_idx = param_dict["rotation"]
                self.solver_assets[instance_id].position.data = torch.tensor(result.x[position_idx:position_idx+3], dtype=torch.float32, device=self.device)
                self.solver_assets[instance_id].rotation.data = torch.tensor(result.x[rotation_idx:rotation_idx+2], dtype=torch.float32, device=self.device)

        else:
            ###############################################################################################################################################
            ### the original gradient descent solver
            ###############################################################################################################################################
            target_instance_ids = set()
            parameters = []
            for instance_id, asset in self.solver_assets.items():
                if asset.optimize and asset.position.requires_grad:
                    parameters.append(asset.position)
                    target_instance_ids.add(instance_id)
                if asset.optimize and asset.rotation.requires_grad:
                    parameters.append(asset.rotation)
                    target_instance_ids.add(instance_id)
            if len(parameters) == 0:
                return {}
            for parameter in parameters:
                assert not torch.isnan(parameter).any()

            ### setup optimizer
            #optimizer = torch.optim.Adam([asset.rotation for asset in self.solver_assets], lr=learning_rate)
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

            frame_paths = []
            prev_loss = 1e6
        
            def is_better_solution(losses, best_losses):
                best_overlap_loss, best_existing_constraint_loss, best_new_constraint_loss = [float(loss.item() if isinstance(loss, torch.Tensor) else loss) for loss in best_losses]
                overlap_loss, existing_constraint_loss, new_constraint_loss = [float(loss.item() if isinstance(loss, torch.Tensor) else loss) for loss in losses]
                if best_overlap_loss < 1e-3 and overlap_loss < 1e-3:
                    return (existing_constraint_loss + new_constraint_loss) < (best_existing_constraint_loss + best_new_constraint_loss)
                return (overlap_loss + existing_constraint_loss + new_constraint_loss) < (best_overlap_loss + best_existing_constraint_loss + best_new_constraint_loss)

            best_losses = (1e6, 1e6, 1e6)
            best_solution = collections.defaultdict(dict)
            for i in tqdm(range(iterations)):
                ### visualization
                if (i == iterations-1 or i % 10 == 0) and temp_dir:
                    frame_output_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    # visualize only the assets that are not walls
                    # Export self.solver_assets to a JSON file
                    def asset_to_dict(asset):
                        return {
                            "position": asset.position.cpu().detach().numpy().tolist(),
                            "rotation": asset.get_theta(),
                            "instance_id": asset.id,
                        }
                    assets_dict = {asset_id: asset_to_dict(asset) for asset_id, asset in self.solver_assets.items() if not asset_id.startswith('walls')}
                    saved_intermediate_states["solver_assets"].append(assets_dict)
                    visualize_grid(self.boundary, self.solver_assets, frame_output_path) 
                    frame_paths.append(frame_output_path)
                    image = imageio.imread(frame_output_path)
                    if i == iterations-1:
                        break

                optimizer.zero_grad()

                overlap_loss, existing_constraint_loss, new_constraint_loss = self.calc_loss(existing_constraints, new_constraints)
                # if overlap loss is smaller than the best 
                loss = overlap_loss + existing_constraint_loss + new_constraint_loss
                # print(f"Iteration {i}, Total Loss: {loss.item()}, Overlap Loss: {overlap_loss.item() if isinstance(overlap_loss, torch.Tensor) else overlap_loss}, Existing Constraint Loss: {existing_constraint_loss.item() if isinstance(existing_constraint_loss, torch.Tensor) else existing_constraint_loss}, New Constraint Loss: {new_constraint_loss.item() if isinstance(new_constraint_loss, torch.Tensor) else new_constraint_loss}")
                loss *= 0.01
                if not loss.requires_grad:
                    raise RuntimeError("Loss does not require gradients.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                # Apply gradients
                optimizer.step()

                if i%10 == 0 and is_better_solution((overlap_loss, existing_constraint_loss, new_constraint_loss), best_losses):
                    best_idx = i
                    best_losses = [
                        float(overlap_loss.item() if isinstance(overlap_loss, torch.Tensor) else overlap_loss),
                        float(existing_constraint_loss.item() if isinstance(existing_constraint_loss, torch.Tensor) else existing_constraint_loss),
                        float(new_constraint_loss.item() if isinstance(new_constraint_loss, torch.Tensor) else new_constraint_loss)
                    ]
                    for instance_id in target_instance_ids:
                        asset = self.solver_assets[instance_id]
                        best_solution[instance_id] = {
                            'position': asset.position.data.clone(),
                            'rotation': asset.rotation.data.clone()
                        }

                # Check for NaN values in position and rotation
                for asset_id, asset in self.solver_assets.items():
                    if torch.isnan(asset.position).any():
                        print(f"Warning: NaN detected in position for asset {asset_id}")
                        import pdb;pdb.set_trace()
                        # Replace NaN values with zeros
                        asset.position = torch.where(torch.isnan(asset.position), torch.zeros_like(asset.position), asset.position)
                    #if torch.isnan(asset.rotation).any():
                    #    print(f"Warning: NaN detected in rotation for asset {asset_id}")
                    #    # Replace NaN values with zeros
                    #    asset.rotation = torch.where(torch.isnan(asset.rotation), torch.zeros_like(asset.rotation), asset.rotation)

                # for k in self.solver_assets.keys():
                #     print(self.solver_assets[k].position)
                if i % 100 == 0 or i == iterations-2:
                    # if loss.item() - prev_loss >= 10:
                    #    break
                    prev_loss = loss.item()
                    self.project_back_to_polygon(all_constraints)
                    scheduler.step()
                    print(f"Iteration {i}, Total Loss: {loss.item()}, Overlap Loss: {overlap_loss.item() if isinstance(overlap_loss, torch.Tensor) else overlap_loss}, Existing Constraint Loss: {existing_constraint_loss.item() if isinstance(existing_constraint_loss, torch.Tensor) else existing_constraint_loss}, New Constraint Loss: {new_constraint_loss.item() if isinstance(new_constraint_loss, torch.Tensor) else new_constraint_loss}")

            if best_solution is not None:
                print(f"Found better solution at iteration {best_idx}")
                for instance_id in target_instance_ids:
                    self.solver_assets[instance_id].position.data = best_solution[instance_id]['position']
                    self.solver_assets[instance_id].rotation.data = best_solution[instance_id]['rotation']

            if output_gif_path:
                with imageio.get_writer(output_gif_path, mode='I', duration=0.5) as writer:
                    for frame_path in frame_paths:
                        image = imageio.imread(frame_path)
                        image = image[:, :, :3]
                        writer.append_data(image)
                # remove all frames
                #for frame_path in frame_paths:
                #    os.remove(frame_path)

        self.project_back_to_polygon(all_constraints)
        ## save the intermediate states
        json_output_path = os.path.join(temp_dir, f"saved_intermediate_states.json")
        with open(json_output_path, 'w') as f:
            json.dump(saved_intermediate_states, f, indent=2)

        results = {}
        for instance_id, asset in self.solver_assets.items():
            if self.is_fixture(instance_id):
                continue
            results[instance_id] = {
                "position": asset.position.cpu().detach().numpy().tolist(),
                "rotation": [0, 0, asset.get_theta()]
            }
            assert results[instance_id]["position"] is not None
            assert results[instance_id]["rotation"] is not None
        return results



if __name__ == "__main__":
    import pdb;pdb.set_trace()
