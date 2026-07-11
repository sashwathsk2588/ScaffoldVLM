import math
import matplotlib
from shapely.geometry import Polygon
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageDraw, ImageFont
from io import BytesIO


def annotate_image_with_coordinates(image_path, visual_marks, output_path, format="coordinate", default_font_size=18):
    script_path = os.path.dirname(os.path.realpath(__file__))
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    wall_font_size = 18 * img_width / 1000
    wall_font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), wall_font_size)

    to_draw_list = []
    if format == "coordinate":
        assert type(visual_marks) == dict
        for (x, y), (pixel_x, pixel_y) in visual_marks.items():
            to_draw_list.append({"pixel": (pixel_x, pixel_y), "text": f"({x},{y})"})

        # Define font (you may need to adjust the path to a font file)
        # adjust the font size based on the image size
        default_font_size = 18 * img_width / 1000
        font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), default_font_size)

    elif format == "text":
        assert type(visual_marks) == list
        for visual_mark in visual_marks:
            assert type(visual_mark) == dict
            assert "pixel" in visual_mark and "text" in visual_mark
        to_draw_list = visual_marks

        default_font_size = default_font_size * img_width / 1000
        font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), default_font_size)

    else:
        raise ValueError("Invalid format. Choose 'coordinate' or 'text'.")

    # Draw marks and coordinates
    for to_draw_dict in to_draw_list:

        pixel_x, pixel_y = to_draw_dict["pixel"]
        text = to_draw_dict["text"]

        pixel_x = pixel_x * img_width
        pixel_y = pixel_y * img_height

        # Draw an arrow from pixel_x, pixel_y to end_pixel_x, end_pixel_y
        if "end_arrow_pixel" in to_draw_dict:
            end_pixel_x, end_pixel_y = to_draw_dict["end_arrow_pixel"]
            end_pixel_x = end_pixel_x * img_width
            end_pixel_y = end_pixel_y * img_height
            # Calculate arrow properties
            arrow_length = ((end_pixel_x - pixel_x)**2 + (end_pixel_y - pixel_y)**2)**0.5
            angle = math.atan2(end_pixel_y - pixel_y, end_pixel_x - pixel_x)
            # Draw the arrow shaft
            draw.line([pixel_x, pixel_y, end_pixel_x, end_pixel_y], fill="black", width=5)
            # Draw the arrow head
            arrow_head_length = min(15, arrow_length / 3)  # Adjust size as needed
            arrow_head_width = arrow_head_length
            x1 = end_pixel_x - arrow_head_length * math.cos(angle) + arrow_head_width * math.sin(angle)
            y1 = end_pixel_y - arrow_head_length * math.sin(angle) - arrow_head_width * math.cos(angle)
            x2 = end_pixel_x - arrow_head_length * math.cos(angle) - arrow_head_width * math.sin(angle)
            y2 = end_pixel_y - arrow_head_length * math.sin(angle) + arrow_head_width * math.cos(angle)
            draw.polygon([end_pixel_x, end_pixel_y, x1, y1, x2, y2], fill="black")

        # Draw a small red dot
        dot_radius = 3
        draw.ellipse([pixel_x - dot_radius, pixel_y - dot_radius, 
                      pixel_x + dot_radius, pixel_y + dot_radius], 
                     fill="red", outline="red")
        # Draw coordinate text
        text_bbox = draw.textbbox((pixel_x, pixel_y), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # coordinate?
        if text.startswith("("):
            draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2), text, font=font, fill="red")
        else:
            # draw wall variable names
            #if text.startswith("walls["):
            #    draw.text((pixel_x - text_w/2, pixel_y + dot_radius + 2), 
            #            text, font=wall_font, fill="red" if text.startswith("(") else "white")
            # draw asset variable names
            font_color = "black"
            if "end_arrow_pixel" in to_draw_dict:
                if end_pixel_y >= pixel_y:
                    if default_font_size > 80:
                        # that means the arrow is pointing downwards, adjust the text position to be above the dot
                        draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2 - text_h*1.3), text, font=font, fill=font_color)
                    else:
                        draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2 - text_h*1.1), text, font=font, fill=font_color)
                else:
                    draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2), text, font=font, fill=font_color)
            else:
                draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2), text, font=font, fill=font_color)
    # Save the annotated image
    img.save(output_path)
    print(f"Annotated image saved to {output_path}")


def load_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Image not loaded correctly.")
    return image
    #except Exception as e:
    #    print(f"Error loading image: {e}")
    #    return None


def overlay_bounding_box(image_path, output_path, visual_mark_path="./data/visual_mark_num.png"):
    image1 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(visual_mark_path, cv2.IMREAD_UNCHANGED)
    #print(image1.shape, image2.shape)
    #assert image1.shape[-1] == 4 and image2.shape[-1] == 4

    # Extract the alpha channel from the second image
    # make this semi-transparent
    #image1[..., -1] = image1[..., -1] * 0.1
    # cover image2 over image2
    #mask2 = image2[..., -1]
    #image1[mask2 > 0] = image2[mask2 > 0]

    # Extract the alpha channel from the second image
    mask1 = image1[..., -1] / 255.0
    mask2 = image2[..., -1] / 255.0
    # 
    #image1[..., :3] = image1[..., :3] = 0.5 * image1[..., :3]
    image1[mask1 > 0, -1] = image1[mask1 > 0, -1] * 0.5
    for c in range(0, 3): image1[..., c] = (1-mask2) * image1[..., c] + mask2 * image2[..., c]
    #image1[..., -1] = np.clip(image1[..., -1] + image2[..., -1], 0, 255).astype(np.uint8)
    # Save or display the result
    has_content = (mask1 + mask2) > 0
    image1[..., -1] = has_content * 255 + (1 - has_content) * 0
    cv2.imwrite(output_path, image1)

    #import pdb;pdb.set_trace()
    #cv2.imshow('Composite Image', composite)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    ### Carrie's version
    #try:
    #    # Check if the image has an alpha channel (transparency)
    #    if image.shape[2] == 4:
    #        alpha_channel = image[:, :, 3]
    #        _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    #    else:
    #        print("The image does not have an alpha channel.")
    #        binary_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #        _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)

    #    # Find contours in the binary mask
    #    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #    # Get the bounding box of the largest contour
    #    if contours:
    #        largest_contour = max(contours, key=cv2.contourArea)
    #        x, y, w, h = cv2.boundingRect(largest_contour)

    #        # Draw the bounding box on the original image
    #        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255, 255), 2)  # Blue color with full opacity

    #        # Define corner points and their labels
    #        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    #        labels = [ '0', '1', '2', '3']
    #        offsets = [(-10, -10), (10, -10), (10, 20), (-10, 20)]  # Offsets to place text outside bounding box

    #        # Overlay the numbers at the four corners
    #        for (corner, label, offset) in zip(corners, labels, offsets):
    #            text_position = (corner[0] + offset[0], corner[1] + offset[1])
    #            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 2)  # Larger font size

    #        cv2.imwrite(output_path, image)
    #        # Convert BGR to RGB for displaying with matplotlib
    #    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    #        
    #    #     # Display the result
    #    #     plt.imshow(image_rgb)
    #    #     plt.title('Image with Bounding Box and Corner Labels')
    #    #     plt.axis('off')
    #    #     plt.show()
    #    else:
    #        print("No contours found in the image.")
    #except Exception as e:
    #    print(f"Error processing image: {e}")


def overlay_text(image_paths, texts=None, output_path=None, text_color=(255, 255, 255)):
    if texts is not None:
        if len(image_paths) != len(texts):
            texts = None
    # Open an image file
    images = [Image.open(image_path) for image_path in image_paths]
    img_width, img_height = images[0].size
    positions = [(0, 0), (img_width, 0), (0, img_height), (img_width, img_height)]

    if len(image_paths) == 4:
        grid_image = Image.new('RGB', (img_width * 2, img_height * 2), (255, 255, 255))
    else:
        grid_image = Image.new('RGB', (img_width * 2, img_height), (255, 255, 255))
    # Paste each image into the grid, with check for alpha channel
    for index, image in enumerate(images):
        # If the image has an alpha channel, create a white background and composite the image onto it
        if image.mode == 'RGBA':
            white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
            white_bg.paste(image, (0, 0), image)
            image = white_bg.convert('RGB')
        
        # Calculate the position for pasting in the grid
        x = (index % 2) * img_width
        y = (index // 2) * img_height
        if texts is not None:
            # Paste the image
            # grid_image.paste(image, (x, y))
            # Prepare the draw object
            draw = ImageDraw.Draw(image)
            # Define the position for the text (upper-left corner)
            position = (10, 10)
            # Define the text color
            text_color = (0,0,0)  # White
            # Define the font (default font with size 40)
            font = ImageFont.truetype("utils/Arial.ttf", 50)
            # Text to be overlaid
            # Overlay the text on the image
            draw.text(position, texts[index], fill=text_color, font=font)
            # Paste the image
        grid_image.paste(image, (x, y))

    # Draw black lines to create grid boundaries
    draw = ImageDraw.Draw(grid_image)
    line_width = 3
    for i in range(1, 2):
        # Vertical lines
        draw.line((img_width * i + line_width * i, 0, img_width * i + line_width * i, grid_image.height), fill='black', width=line_width)
        # Horizontal lines
        draw.line((0, img_height * i + line_width * i, grid_image.width, img_height * i + line_width * i), fill='black', width=line_width)

    grid_image.save(output_path)
    return output_path


def visualize_grid(room_poly, assets: dict, output_path: str, grid_points=None, randomize_color=False, output_size=(1200,1200), dpi=300):
    ### assets: list of Assets
    room_poly = Polygon(room_poly)
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    # Pre-determined color palette
    color_palette = [
        'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'
    ]

    # Calculate figure size in inches
    fig_size = (output_size[0] / dpi, output_size[1] / dpi)
    # Create a new figure with fixed size and DPI
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    fig.set_dpi(dpi)

    # draw the room
    x, y = room_poly.exterior.xy
    ax.plot(x, y, '-', label='Room', color='black', linewidth=2)

    # draw the solutions
    idx = 0
    for instance_id, asset in assets.items():
        if instance_id.startswith('walls'):
            continue
        center_x, center_y, center_z = asset.position.cpu().detach().numpy()
        # asset.rotation[-1].cpu().detach().item()
        rotation = asset.get_theta()
        rotated_corners = asset.get_2dpolygon().cpu().detach().numpy()

          # Choose color based on randomize_color flag
        if randomize_color:
            color = (random.random(), random.random(), random.random())
        else:
            color = color_palette[idx % len(color_palette)]

        # create a polygon for the solution
        obj_poly = Polygon(rotated_corners)
        x, y = obj_poly.exterior.xy
        ax.plot(x, y, '-', linewidth=2, color=color, clip_on=False)
        ax.text(rotated_corners[0][0]-0.3, rotated_corners[0][1]-0.3, asset.id, fontsize=8, ha='center', color=color, clip_on=False)

        # ax.text(center_x, center_y, object_name, fontsize=18, ha='center')

        # the object points towards the +y axis
        # get the direction of the object after rotation
        #degree = np.radians(-rotation+90)
        # set arrow direction based on rotation
        ax.arrow(center_x, center_y, np.cos(rotation)/2, np.sin(rotation)/2, head_width=0.1, fc=color, clip_on=False, color=color)
        idx += 1

    # Fix the x-y axis based on the room polygon's bounding box
    min_x, min_y, max_x, max_y = room_poly.bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # axis off
    #ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal', 'box')  # to keep the ratios equal along x and y axis
    #plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, pad_inches=0.1, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    #vertices = [
    #    [0, 0, -1.6],
    #    [5, 0, -1.6], 
    #    [5, 5, -1.6],
    #    [0, 5, -1.6],
    #]
    vertices = [
        [-1, -1, -1.6],
        [-1, 5, -1.6], 
        [5, 5, -1.6],
        [5, -1, -1.6],
    ]
    assets = [
        Asset(
            id=1,
            category="chair",
            bounding_box=[[1, 1, 1], [2, 2, 2]],
            rotation=[0, 90, 0]
        ),
        Asset(
            id=1,
            category="chair",
            bounding_box=[[0, 0, 0], [1, 1, 1]],
            rotation=[0, 30, 0]
        ),
        Asset(
            id=2,
            category="table",
            bounding_box=[[1, 1, 0.8], [2, 3, 1]],
            rotation=[0, 145, 0]
        ),
    ]
    create_time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
    #output_path = f"tmp/{create_time}.pdf"
    output_path = f"tmp/test.pdf"
    os.makedirs("tmp", exist_ok=True)
    visualize_grid(vertices, assets, output_path)

    ### overlay text
    image_path = "/Users/sunfanyun/Downloads/3D_scene_generation/GenLayout/data/sceneVerse/preprocessed/ProcThor/0_shelf/render_{}.png"
    image_paths = [image_path.format(degree) for degree in [0, 90, 180, 270]]
    texts = [f"{degree}°" for degree in [0, 90, 180, 270]]
    output_path = "tmp.png"
    overlay_text(image_paths, texts, output_path)

