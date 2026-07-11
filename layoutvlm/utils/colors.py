import numpy as np
import matplotlib.pyplot as plt


def convert_color_range(color, from_range='0-1', to_range='0-255'):
    """
    Convert the color range from [0, 1] to [0, 255] or vice versa
    :param color: list or tuple of RGB values
    :param from_range: '0-1' or '0-255'
    :param to_range: '0-1' or '0-255'
    :return: list of RGB values
    """
    if from_range == '0-1' and to_range == '0-255':
        return [int(c * 255) for c in color]
    elif from_range == '0-255' and to_range == '0-1':
        return [c / 255 for c in color]
    elif from_range == to_range:
        return color
    else:
        raise ValueError(f"Conversion from {from_range} to {to_range} not supported")


def convert_color_format(color, from_format="rgba", to_format="rgb", alpha_value=1.0):
    if from_format == "rgba" and to_format == "rgb":
        return color[:3]
    elif from_format == "rgb" and to_format == "rgba":
        color = list(color)
        color.append(alpha_value)
        return color
    elif from_format == "rgb" and to_format == "bgr":
        return color[::-1]
    elif from_format == "bgr" and to_format == "rgb":
        return color[::-1]
    elif from_format == to_format:
        return color
    else:
        raise ValueError(f"Conversion from {from_format} to {to_format} not supported")


def get_categorical_colors(num_categories: int, colormap_name='tab10', color_range='0-255', color_format='rgb'):

    # Get the colormap
    if colormap_name == 'tab10':
        colormap = plt.cm.tab10
    elif colormap_name == 'tab20':
        colormap = plt.cm.tab20
    elif colormap_name == "viridis":
        colormap = plt.cm.viridis
    elif colormap_name == "jet":
        colormap = plt.cm.jet
    else:
        raise ValueError(f"colormap {colormap_name} not supported")

    # Normalize values to the range [0, 1] for colormap
    norm = plt.Normalize(0, num_categories - 1)

    # Assign colors to each value
    colors = [colormap(norm(value)) for value in range(num_categories)]

    # Convert colors to the desired range
    colors = [convert_color_range(color, from_range='0-1', to_range=color_range) for color in colors]
    colors = [convert_color_format(color, from_format='rgba', to_format=color_format) for color in colors]

    if colormap_name == "tab20" and num_categories == 20:
        colors = colors[::2] + colors[1::2]

    return colors
