import os
from pathlib import Path


def load_depth_map_paths(depth_map_folder: str) -> list[str]:
    frame_number = 0  # Starting frame number
    depth_maps = []  # List to hold the depth map arrays
    while True:
        depth_map_filename = f"depth_frame_{frame_number:04d}.png"
        depth_map_path = os.path.join(depth_map_folder, depth_map_filename)

        # Check if the depth map file exists
        if not os.path.isfile(depth_map_path):
            break  # Exit loop if the file doesn't exist

        depth_maps.append(depth_map_path)

        # Increment the frame_number for the next iteration
        frame_number += 1

    # At this point, 'depth_maps' contains all the loaded depth map arrays
    print(f"Found {len(depth_maps)} depth maps.")

    return depth_maps


def get_file_name(full_file_path):
    return Path(full_file_path).name
