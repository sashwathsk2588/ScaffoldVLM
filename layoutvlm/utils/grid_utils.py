from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import cv2


def normalize_and_scale_vertices(vertices, image_size):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    scale = (image_size[0] - 30) / (max_coords - min_coords)
    scale = np.min(scale)  # Ensure uniform scaling
    scaled_vertices = (vertices - min_coords) * scale
    return scaled_vertices

def translate_to_center(vertices, image_size):
    # Calculate the bounding box of the vertices
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    bbox_center = (min_coords + max_coords) / 2
    
    # Calculate the center of the image
    image_center = np.array(image_size) / 2
    
    # Calculate the translation vector
    translation_vector = image_center - bbox_center
    
    # Apply the translation
    translated_vertices = vertices + translation_vector
    return translated_vertices

# Function to create a mask for the floor mesh
def create_floor_mask(xyz, image_size=(224, 224)):
    # Extract the x and y coordinates (assuming z is constant for the floor)
    vertices = []
    for i in range(0, len(xyz), 3):
        x = xyz[i+2]
        y = xyz[i]
        z = xyz[i+1]
        vertices.append([x, y])
    vertices_np = np.array(vertices)
    
    # Normalize and scale the coordinates
    scaled_vertices = normalize_and_scale_vertices(vertices_np, image_size)
    
    # Translate vertices to the center of the image
    centered_vertices = translate_to_center(scaled_vertices, image_size)
    
    min_coords = np.min(centered_vertices, axis=0)
    max_coords = np.max(centered_vertices, axis=0)
    bounding_box = (min_coords, max_coords)

    # Create a convex hull from the centered vertices
    hull = ConvexHull(centered_vertices)
    hull_vertices = centered_vertices[hull.vertices]
    
    # Create a blank mask
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # Fill the mask based on the hull
    points = [[x, y] for x, y in hull_vertices]
    mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=255)
    
    return (min_coords, max_coords), mask

def create_grid(min_coords, max_coords, image_mask, output_path):
    image_height, image_width = image_mask.shape

    # Define the dimensions of the grid
    grid_rows, grid_cols = 4, 4

    x_min, y_min = min_coords
    x_max, y_max = max_coords
    #print(min_coords, max_coords)

    # Calculate the width and height of the bounding box
    w = x_max - x_min
    h = y_max - y_min

    # Calculate the size of each grid cell for the white area
    cell_width_cropped = w / grid_cols
    cell_height_cropped = h / grid_rows

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Display the mask image
    ax.imshow(image_mask, cmap='gray')

    # Overlay the 4x4 grid only on the white area
    for i in range(grid_cols + 1):
        plt.plot([x_min + i * cell_width_cropped, x_min + i * cell_width_cropped], [y_min, y_min + h], color="red")
    for j in range(grid_rows + 1):
        plt.plot([x_min, x_min + w], [y_min + j * cell_height_cropped, y_min + j * cell_height_cropped], color="red")

    # Add the numbers centered in each grid cell in the white area
    for i in range(grid_cols):
        for j in range(grid_rows):
            ax.text(i, j, str(j * grid_cols + i + 1), color="red", ha='center', va='center', fontsize=8)

    plt.axis('off')
    fig.savefig(os.path.join(output_path, 'floor_mask_grid.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def find_bounding_box_from_vertices(vertices):
    # Extract x and y coordinates
    vertices = np.array(vertices)[:, :2]  # Use only x and y coordinates
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    return min_coords, max_coords

def find_bounding_box(mask):
    # Find non-zero points (white pixels)
    non_zero_points = np.transpose(np.nonzero(mask))

    # Calculate the minimum and maximum coordinates
    min_coords = np.min(non_zero_points, axis=0)
    max_coords = np.max(non_zero_points, axis=0)

    return min_coords, max_coords

# Function to determine which grid cell a point belongs to
def get_grid_number(point, min_coords_meters, max_coords_meters, min_coords_pixels, max_coords_pixels, grid_rows=4, grid_cols=4):
    x_min_m, y_min_m = min_coords_meters
    x_max_m, y_max_m = max_coords_meters

    x_min_p, y_min_p = min_coords_pixels
    x_max_p, y_max_p = max_coords_pixels

    # Calculate the width and height of the bounding box in meters
    w_m = x_max_m - x_min_m
    h_m = y_max_m - y_min_m

    # Calculate the width and height of the bounding box in pixels
    w_p = x_max_p - x_min_p
    h_p = y_max_p - y_min_p

    # Extract x and y coordinates of the point
    x, y = point

    # Map the point from meters to pixel values
    scaled_x = ((x - x_min_m) / w_m) * w_p + x_min_p
    scaled_y = ((y - y_min_m) / h_m) * h_p + y_min_p

    # Calculate the size of each grid cell in pixels
    cell_width = w_p / grid_cols
    cell_height = h_p / grid_rows

    # Determine the grid cell
    col = int((scaled_x - x_min_p) / cell_width)
    row = int((scaled_y - y_min_p) / cell_height)

    # Calculate the grid number
    grid_number = row * grid_cols + col + 1

    return grid_number

def find_grid_number(point, boundary, mask_image_path):
    # Assuming the key for the boundary is 'boundary'
    point = (point[0], point[1])
    # Find the bounding box from the boundary vertices in meters
    min_coords_meters, max_coords_meters = find_bounding_box_from_vertices(boundary)

    # Load the mask image
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask is read correctly
    if mask is None:
        raise ValueError("Mask image could not be read. Check the path.")

    # Find the bounding box of the floor mask in pixels
    min_coords_pixels, max_coords_pixels = find_bounding_box(mask)

    # Determine the grid number for the given point
    grid_number = get_grid_number(point, min_coords_meters, max_coords_meters, min_coords_pixels, max_coords_pixels)
    return grid_number

def split_into_grids(boundary, num_grids=4):
    """
    Split the bounding box into a grid.
    min_coords: (min_x, min_y)
    max_coords: (max_x, max_y)
    num_grids: Number of grids along one axis (e.g., 4 for a 4x4 grid)
    Returns: List of grid boundaries as tuples [(grid_number, (min_x, min_y, max_x, max_y)), ...]
    """

    min_coords, max_coords = find_bounding_box_from_vertices(boundary)
    #print(min_coords, max_coords)
    min_x, min_y = min_coords
    max_x, max_y = max_coords
    grid_width = (max_x - min_x) / num_grids
    grid_height = (max_y - min_y) / num_grids
    
    grids = {}
    grid_number = 1
    
    for row in range(num_grids):
        for col in range(num_grids):
            grid_min_x = min_x + col * grid_width
            grid_min_y = max_y - (row + 1) * grid_height
            grid_max_x = grid_min_x + grid_width
            grid_max_y = grid_min_y + grid_height
            grids[grid_number] = (grid_min_x, grid_min_y, grid_max_x, grid_max_y)
            grid_number += 1
    
    return grids
    
