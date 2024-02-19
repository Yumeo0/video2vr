import cv2
import numpy as np


def add_border(image, paddingH, paddingV):
    # Define padding values for top, bottom, left and right
    top_padding = paddingV
    bottom_padding = paddingV
    left_padding = paddingH
    right_padding = paddingH

    # Choose the border type
    # cv2.BORDER_CONSTANT - Adds a constant colored border. The color is chosen with the next parameter.
    # cv2.BORDER_REFLECT - Border will be mirror reflection of the border elements.
    # cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - Same as above, but with a slight change.
    # cv2.BORDER_REPLICATE - Last element is replicated throughout.
    # cv2.BORDER_WRAP - Not sure how to explain, it's weird.

    border_type = cv2.BORDER_CONSTANT

    border_color = [0, 0, 0]

    # Add padding to the image
    padded_image = cv2.copyMakeBorder(
        image,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        border_type,
        value=border_color,
    )

    return padded_image


def fisheye_transform(image: np.ndarray):
    # Assuming image is a single channel or grayscale for simplicity
    height, width = image.shape[:2]

    # New camera matrix based on width and height of the image
    K = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])

    # Distortion coefficients - values to experiment with
    D = np.array([-0.25, 0.25, 0, 0])

    # Generate new projection matrix
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
    )

    # Apply fisheye
    fisheye_img = cv2.remap(
        image,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return fisheye_img


# Function for shifting the perspective based on depth map
def render_new_perspective(shift, depth_map, frame):
    h, w = frame.shape[:2]

    # Calculate the map for remapping
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    # Displace map_x based on depth, you might need to normalize the depth values
    map_x = map_x + shift * depth_map

    # Ensure map_x and map_y stay within image bounds
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    # Apply the remapping to the frame
    new_image = cv2.remap(
        frame,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    )

    return new_image
