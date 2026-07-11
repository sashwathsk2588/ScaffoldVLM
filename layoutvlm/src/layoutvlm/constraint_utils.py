import torch
import numpy as np
from shapely.geometry import Polygon, Point
from utils.placement_utils import half_vector_intersects_polygon
import torch.nn.functional as F

    
def point_to_segment_batch_loss(points, segments):
    """
    Calculate the shortest distance between a batch of points and a batch of line segments in a differentiable manner using PyTorch.

    Parameters:
    points (Tensor): A batch of 2D points of shape [N, 2].
    segments (Tensor): A batch of line segments of shape [M, 4].

    Returns:
    Tensor: A tensor of shape [N, M] containing the shortest distances from each point to each line segment.
    """
    px, py = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1)
    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]

    # Reshape for broadcasting
    x1, y1, x2, y2 = x1.unsqueeze(0), y1.unsqueeze(0), x2.unsqueeze(0), y2.unsqueeze(0)

    # Vector from the first endpoint to the points
    dpx = px - x1
    dpy = py - y1

    # Vector from the first endpoint to the second endpoint
    dx = x2 - x1
    dy = y2 - y1

    # Dot product of the above vectors
    dot_product = dpx * dx + dpy * dy

    # Length squared of the segment vector
    len_sq = dx * dx + dy * dy

    # Projection factor normalized to [0, 1]
    projection = dot_product / (len_sq + 1e-8)
    projection = torch.clamp(projection, 0, 1)

    # Closest points on the segments
    closest_x = x1 + projection * dx
    closest_y = y1 + projection * dy

    #closest = torch.concat([closest_x, closest_y], dim=-1)
    #pp = torch.concat([px, py], dim=-1)

    # Distance from the points to the closest points on the segments
    #distance = torch.sqrt((closest_x - px) ** 2 + (closest_y - py) ** 2)
    distance = (closest_x - px) ** 2 + (closest_y - py) ** 2
    return distance


def cosine_distance_loss(vector1, vector2, epsilon=1e-5, beta=10):
    """
    Calculate the loss based on the cosine distance between two vectors.

    Args:
        vector1 (torch.Tensor): First vector tensor.
        vector2 (torch.Tensor): Second vector tensor.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Normalize the vectors
    vector1_norm = F.normalize(vector1, p=2, dim=-1)
    # apply small perturbation?
    # vector1_norm = vector1_norm + epsilon * torch.sign(vector1_norm)
    vector2_norm = F.normalize(vector2, p=2, dim=-1)

    # Calculate cosine similarity
    cosine_similarity = torch.sum(vector1_norm * vector2_norm, dim=-1)
    return 1 - cosine_similarity.mean()
    #cosine_similarity = cosine_similarity.clamp(-1 + epsilon, 1 - epsilon)
    #return F.softplus(-beta * cosine_similarity).mean()
    # Convert cosine similarity to cosine distance (1 - cosine similarity)
    #return (-cosine_similarity).mean()



# def distance_loss(coord1, coord2, min_distance=1.0, max_distance=3.0):
#     """
#     Calculate the loss based on the distance between two coordinates being within a specific range.

#     Args:
#         coord1 (torch.Tensor): First coordinate tensor.
#         coord2 (torch.Tensor): Second coordinate tensor.
#         min_distance (float): The minimum distance threshold. Default is 1.0.
#         max_distance (float): The maximum distance threshold. Default is 3.0.

#     Returns:
#         torch.Tensor: The computed loss.
#     """
#     # Calculate the squared Euclidean distance between the two coordinates
#     distance = torch.sqrt(torch.sum((coord1 - coord2) ** 2))

#     # Use differentiable operations to calculate loss based on the distance range
#     below_min_loss = F.relu(min_distance - distance) ** 2
#     above_max_loss = F.relu(distance - max_distance) ** 2
#     within_range_loss = torch.tensor(0.00, dtype=distance.dtype, device=distance.device)

#     # Select the appropriate loss
#     loss = torch.where(
#         distance < min_distance,
#         below_min_loss,
#         torch.where(distance > max_distance, above_max_loss, within_range_loss)
#     )

#     return loss

def distance_loss(coord1, coord2, min_distance=1.0, max_distance=3.0):
    """
    Calculate the loss based on the distance between two coordinates being within a specific range.

    Args:
        coord1 (torch.Tensor): First coordinate tensor.
        coord2 (torch.Tensor): Second coordinate tensor.
        min_distance (float): The minimum distance threshold. Default is 1.0.
        max_distance (float): The maximum distance threshold. Default is 3.0.

    Returns:
        torch.Tensor: The computed loss.
    """
    min_distance = 0 if min_distance is None else min_distance
    max_distance = 1e6 if max_distance is None else max_distance

    # Calculate the squared Euclidean distance between the two coordinates
    squared_distance = torch.sum((coord1 - coord2) ** 2)
    
    # Compute the loss based on the distance range using smooth approximations
    below_min_loss = F.relu(min_distance**2 - squared_distance)
    above_max_loss = F.relu(squared_distance - max_distance**2)
    # if 
    
    loss = below_min_loss + above_max_loss
    return loss




def is_point_on_line_segment(point, line_start, line_end):
    """Check if a point lies on a line segment."""
    return (np.cross(line_end - line_start, point - line_start) == 0 and 
            np.dot(line_end - line_start, point - line_start) >= 0 and 
            np.dot(line_start - line_end, point - line_end) >= 0)

def ray_intersects_segment(origin, direction, v1, v2):
    """Check if a ray intersects with a line segment."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Check if the ray's origin is on the line segment
    if is_point_on_line_segment(origin, v1, v2):
        return True
    
    # Calculate the intersection point
    v = v2 - v1
    cross_product = np.cross(direction, v)
    
    # Check if the ray is parallel to the line segment
    if abs(cross_product) < 1e-8:
        return False
    
    t = np.cross(v1 - origin, v) / cross_product
    u = np.cross(direction, origin - v1) / cross_product
    
    # Check if the intersection point is on the line segment and in the direction of the ray
    return t >= 0 and 0 <= u <= 1

def ray_intersects_polygon(origin, direction, polygon):
    """Check if a ray intersects with a polygon."""
    origin = np.array(origin)
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)  # Normalize direction
    
    for i in range(len(polygon)):
        if ray_intersects_segment(origin, direction, polygon[i], polygon[(i + 1) % len(polygon)]):
            return True
    
    return False
