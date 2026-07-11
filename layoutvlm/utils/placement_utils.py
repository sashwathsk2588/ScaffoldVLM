import numpy as np
import random
import math
from shapely.geometry import Polygon, Point
from shapely.geometry import Point, LineString, Polygon
import re
from PIL import Image
import math


def IOU(polygon1, polygon2):
    """Calculates the IoU of two polygons.

    Args:
        polygon1: A Shapely Polygon object.
        polygon2: A Shapely Polygon object.

    Returns:
        The IoU of the two polygons, as a float.
    """

    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)
    iou = intersection.area / union.area
    return iou


def half_vector_intersects_polygon(start_point, direction, polygon):
    """
    Check if a half vector (ray) intersects with a polygon.

    Parameters:
    - start_point: tuple of (x, y), the starting point of the ray.
    - direction: tuple of (dx, dy), the direction vector of the ray.
    - polygon: shapely.geometry.Polygon, the polygon to check for intersection.

    Returns:
    - bool: True if the ray intersects with the polygon, False otherwise.
    """
    # Create the starting point
    start = Point(start_point)
    
    # Create an end point far away in the direction of the ray
    # Adjust the multiplier to ensure the line segment is long enough
    multiplier = 1e6
    end_point = (start_point[0] + direction[0] * multiplier, start_point[1] + direction[1] * multiplier)
    
    # Create the line representing the ray
    ray = LineString([start_point, end_point])
    
    # Check for intersection
    return ray.intersects(polygon)


def triangle_area(A, B, C):
    """Calculate the area of a triangle given by points A, B, and C"""
    return abs((A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1])) / 2.0)

def random_point_in_triangle(A, B, C):
    """Generate a random point within the triangle defined by points A, B, and C"""
    r1, r2 = random.random(), random.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    x = (1 - r1 - r2) * A[0] + r1 * B[0] + r2 * C[0]
    y = (1 - r1 - r2) * A[1] + r1 * B[1] + r2 * C[1]
    return (x, y)

def get_random_placement(floor_vertices, add_z=False):
    if len(floor_vertices[0]) == 2:
        polygon = Polygon([[x,y] for x,y in floor_vertices])
    else:
        polygon = Polygon([[x,y] for x,y,z in floor_vertices])
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        random_point = [random.uniform(minx, maxx), random.uniform(miny, maxy)]
        if polygon.contains(Point(random_point)):
            if add_z:
                random_point = [random_point[0], random_point[1], 0]
            return random_point

### TODO: check if this rotation is implemented correctly?
def get_bbox_corners(position, rotation, bbox_size):
    x,y,z = position
    rotation_deg = rotation[2]

    # Calculate the rotation matrix
    theta = np.radians(-rotation_deg)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    dim_x, dim_y, dim_z = bbox_size

    epsilon = 0.05
    dim_x = max(0, dim_x - epsilon)
    dim_y = max(0, dim_y - epsilon)
    dim_z = max(0, dim_z - epsilon)

    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Define the corners of the box before rotation
    half_dim_x, half_dim_y, half_dim_z = dim_x / 2, dim_y / 2, dim_z / 2
    corners = np.array([
        [-half_dim_x, -half_dim_y, -half_dim_z],
        [half_dim_x, -half_dim_y, -half_dim_z],
        [half_dim_x, half_dim_y, -half_dim_z],
        [-half_dim_x, half_dim_y, -half_dim_z],
        [-half_dim_x, -half_dim_y, half_dim_z],
        [half_dim_x, -half_dim_y, half_dim_z],
        [half_dim_x, half_dim_y, half_dim_z],
        [-half_dim_x, half_dim_y, half_dim_z]
    ])

    # Rotate the corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate the corners to the position (x, y, z)
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    rotated_corners[:, 2] += z
    
    return rotated_corners

def fill_result_with_random_placements(task, layout_result, MAX_RETRIES=10):
    unplaced_assets = []
    for asset_id in task["assets"].keys():
        if asset_id not in layout_result.keys():
            unplaced_assets.append(asset_id)
    print("randomly placed assets: ", unplaced_assets)
    if len(unplaced_assets) == 0:
        return
    boundary = Polygon([[x, y] for x, y, z in task["boundary"]["floor_vertices"]]) 
    for asset_id in unplaced_assets:
        for _ in range(MAX_RETRIES):
            x, y = get_random_placement(task["boundary"]["floor_vertices"])
            z = task["assets"][asset_id]["assetMetadata"]["boundingBox"]["z"]/2
            rotation =  [0, 0, np.random.randint(0, 360)]
            layout_result[asset_id] =  {
                "position": [x, y, z],
                "rotation": rotation
            }
            bbox_size = [
                task["assets"][asset_id]["assetMetadata"]["boundingBox"]["x"],
                task["assets"][asset_id]["assetMetadata"]["boundingBox"]["y"],
                task["assets"][asset_id]["assetMetadata"]["boundingBox"]["z"],
            ]
            bbox_corners = get_bbox_corners([x,y,z], rotation, bbox_size)
            if all(boundary.contains(Point(x,y)) for x,y,z in bbox_corners):
                break


def extract_numbers(input_string):
    # Regular expression to match floating point numbers
    float_pattern = r'[-+]?\d*\.\d+|\d+'
    # Find all matches of floating point numbers in the input string
    matches = re.findall(float_pattern, input_string)
    # Convert matches to floats
    float_numbers = [float(num) for num in matches]
    return float_numbers


def convert_json_format(data, for_gpt4o=False):
    conversation = []
    
    for conv in data['conversations']:
        if conv['from'] == 'human':
            user_content = [
                {
                    "type": "text",
                    "text": conv['value'].replace("<image>", "")
                }
            ]
            if for_gpt4o:
                for i in range(conv['value'].count("<image>")):
                    encoded_image = encode_image(data['image'][i])
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                        }
                    )
            else:
                user_content.extend([{"type": "image"} for _ in range(conv['value'].count("<image>"))])
            
            conversation.append({
                "role": "user",
                "content": user_content
            })
    image_list = [Image.open(image_path).convert("RGB") for image_path in data["image"]]
    return conversation, image_list

def extract_asset_info(code):
    position = []
    rotation = []
    constraints = []
    
    # Regex to find positions and rotations
    position_pattern = re.compile(r"\w+\[\d\]\.position\s*=\s*\[([\d.,\s-]+)\]")
    rotation_pattern = re.compile(r"\w+\[\d\]\.rotation\s*=\s*\[([\d.,\s-]+)\]")
    
    # Regex to find constraints
    constraint_pattern = re.compile(r"(solver\.\w+\([^)]*\))")
    
    # Find all positions
    for match in position_pattern.findall(code):
        pos = [float(x) for x in match.split(',')]
        position.append(pos)

    # Find all rotations
    for match in rotation_pattern.findall(code):
        rot = [float(x) for x in match.split(',')]
        # If only one value is provided, assume it's for the Z axis and add [0, 0, value]
        if len(rot) == 1:
            rotation.append([0.0, 0.0, rot[0]])
        else:
            rotation.append(rot)
    
    # Find all constraints
    constraints = constraint_pattern.findall(code)
    
    # Join constraints into a formatted string
    constraint_str = "\n".join(constraints)
    
    return position, rotation, constraint_str

def extract_initialization_from_string(data_str, debug=True):
    # Regular expression pattern to match object, position, and rotation
    position_pattern = re.compile(r"(\w+\[\d+\])\.position = (\[[-\d., ]+\])")
    rotation_pattern = re.compile(r"(\w+\[\d+\])\.rotation = (\[[-\d., ]+\])")
    
    # Find all positions and rotations
    positions = position_pattern.findall(data_str)
    rotations = rotation_pattern.findall(data_str)

    # Create a dictionary to store the results
    objects = {}

    # Populate the dictionary with positions
    for obj, pos in positions:
        pos_list = eval(pos)  # Convert string to list
        if obj not in objects:
            objects[obj] = {'position': pos_list, 'rotation': None}

    # Add rotations to the corresponding objects
    for obj, rot in rotations:
        rot_list = eval(rot)  # Convert string to list
        if obj in objects:
            objects[obj]['rotation'] = rot_list
    
    return objects


def replace_z_rot_degree_to_rpy_radians(code):
    # Regex pattern to find 'xxx.rotation = [some_value]'
    pattern = r'(\w+\[\d+\])\.rotation = \[(-?\d+\.?\d*)\]'

    # Function to replace degrees with radians
    def replace_with_radians(match):
        obj = match.group(1)
        degrees = float(match.group(2))  # Convert the degree value to float
        radians = math.radians(degrees)  # Convert degrees to radians
        return f'{obj}.rotation = [0, 0, {radians}]'

    # Replace all occurrences in the code
    modified_code = re.sub(pattern, replace_with_radians, code)
    return modified_code


if __name__ == '__main__':
    # Example usage
    floor_vertices = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    print(get_random_placement(floor_vertices))
