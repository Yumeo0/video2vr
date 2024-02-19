import cv2
import torch
import os
import shutil
import numpy as np
from PIL import Image
from transformers import pipeline
from typing import Callable
from libs.cuda import current_device
from libs.file_utils import *
from libs.image_utils import *
from libs.ffmpeg import run_ffmpeg_command
from libs.settings import settings_manager


def process_frames_in_batch(
    pipe,
    frames_batch: list[Image.Image],
    frame_numbers: list[int],
    depth_maps_dir: str,
    callback: Callable[[np.ndarray, np.ndarray], any],
):
    # Process the batch of frames through the pipeline
    batch_results = pipe(images=frames_batch)

    # Iterate over the results and the frames
    for depth_map_data, frame, frame_idx in zip(
        batch_results, frames_batch, frame_numbers
    ):
        depth_map = np.array(depth_map_data["depth"])

        # ... (process the depth map as before)
        # Convert depth map to 8-bit for saving
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8 = depth_map_normalized.astype("uint8")

        # Save the depth map to a file
        depth_output_path = os.path.join(
            depth_maps_dir, f"depth_frame_{frame_idx:04d}.png"
        )
        cv2.imwrite(depth_output_path, depth_map_uint8)
        print(f"[INFO] Saved {depth_output_path}")

        # If a callback function is provided, call it with the frame data
        if callback:
            callback(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR), depth_map_uint8)


def generate_depth_maps_with_batch(
    video_input_path: str, callback: Callable[[np.ndarray, np.ndarray], any]
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Better speed, moderately less quality: Intel/dpt-swinv2-large-384
    # Best quality: Intel/dpt-beit-large-512
    # Idk: LiheYoung/depth-anything-small-hf
    pipe = pipeline(
        task="depth-estimation",
        model=settings_manager.settings.ai_model,
        device=device,
    )

    depth_maps_dir = "depth_maps"
    os.makedirs(depth_maps_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_input_path)
    frame_number = 0

    frames_batch = []
    frame_numbers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        frames_batch.append(pil_img)
        frame_numbers.append(frame_number)

        # When the batch is full, process it
        if len(frames_batch) == settings_manager.settings.batch_size:
            process_frames_in_batch(
                pipe, frames_batch, frame_numbers, depth_maps_dir, callback
            )
            # Reset the batch
            frames_batch = []
            frame_numbers = []

        print("[INFO] Read frame " + str(frame_number))

        # Increase frame number
        frame_number += 1

    # Process the last batch if it's not empty
    if frames_batch:
        process_frames_in_batch(
            pipe, frames_batch, frame_numbers, depth_maps_dir, callback
        )

    cap.release()
    cv2.destroyAllWindows()


def generate_depth_maps(
    video_input_path: str, callback: Callable[[np.ndarray, np.ndarray], any]
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {current_device()}")

    pipe = pipeline(
        task="depth-estimation", model="Intel/dpt-beit-large-512", device=device
    )

    # Create a directory for saving depth maps if it doesn't exist
    depth_maps_dir = "depth_maps"
    if not os.path.exists(depth_maps_dir):
        os.makedirs(depth_maps_dir)

    # Hook into the video file
    cap = cv2.VideoCapture(video_input_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Transform input for MiDaS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict and resize as before
        with torch.no_grad():
            # prediction = midas(imgbatch)
            pil_img = Image.fromarray(img)
            result = pipe(images=pil_img)

            depth_map = np.array(result["depth"])

            # Convert depth map to 8-bit for saving
            depth_map_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX
            )
            depth_map_uint8 = depth_map_normalized.astype("uint8")

            # Save the depth map
            depth_output_path = os.path.join(
                depth_maps_dir, f"depth_frame_{frame_number:04d}.png"
            )
            cv2.imwrite(depth_output_path, depth_map_uint8)
            print(f"[INFO] Saved {depth_output_path}")

            # Increase frame number
            frame_number += 1

        # If you have a callback function passed, call it with the frame data
        if callback:
            callback(frame, depth_map_uint8)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


# Placeholder function for generating 3D points from depth (assuming orthographic projection for simplicity)
def depth_to_3d(depth_map):
    h, w = depth_map.shape
    i, j = np.indices((h, w))

    x = j - w // 2  # These coordinates would actually be dependent on real-world units
    y = i - h // 2
    z = depth_map.astype(float)  # Assuming depth map is normalized
    return x, y, z


def optimize_vr_video(video_input_path: str, input_filepath: str, output_filepath: str):
    """Optimizes a VR video using FFmpeg.

    Args:
        input_filepath (str): The file path of the input video.
        output_filepath (str): The file path of the optimized output video.
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        input_filepath,
        "-i",
        video_input_path,
        "-c:v",
        "libx265",
        "-preset",
        "medium",  # You can change this to 'fast' if speed is a concern
        "-crf",
        "22",
        #'-vf', 'scale=1920x540', # Scale down resolution
        "-c:a",
        "aac",
        "-map",
        "0:v:0",  # Take the video from the first input file
        "-map",
        "1:a:0",  # Take the audio from the second input file
        "-y",  # Override existing file
        output_filepath,
    ]

    print("Optimizing video")

    # Run the FFmpeg command using the 'run_ffmpeg_command' function
    success = run_ffmpeg_command(ffmpeg_cmd)

    if success:
        print("Video optimization completed successfully.")
        return True
    else:
        print("Video optimization failed.")
        return False


def generate_vr_video(
    video_input_path: str, callback: Callable[[np.ndarray, np.ndarray], any]
):
    generate_depth_maps_with_batch(video_input_path, callback)

    # Video parameters
    depth_map_paths = load_depth_map_paths("depth_maps")

    # Video capture and properties
    video = cv2.VideoCapture(video_input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer for the output VR video
    os.makedirs("output", exist_ok=True)
    temp_filename = "output/temp_VR_SBS_" + get_file_name(video_input_path)
    filename = "output/VR_SBS_" + get_file_name(video_input_path)
    out = cv2.VideoWriter(
        temp_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width * 2, frame_height),
    )

    # Process each frame and save to output
    frame_index = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break  # Break if we reach the end of the video

        # Read the corresponding depth map
        depth_map_path = depth_map_paths[frame_index]

        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"Couldn't load depth map from {depth_map_path}. Stopping.")
            break

        # Normalize depth map from 0 to 1 if necessary
        depth_map_normalized = cv2.normalize(
            depth_map,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Render the new perspectives
        offset = settings_manager.settings.depth_offset
        left_image = render_new_perspective(-offset, depth_map_normalized, frame)
        right_image = render_new_perspective(offset, depth_map_normalized, frame)

        if callback:
            callback(left_image, right_image)

        # Combine images side by side
        combined_frame = np.hstack((left_image, right_image))
        out.write(combined_frame)

        frame_index += 1

    # Release everything when done
    video.release()
    out.release()

    if optimize_vr_video(video_input_path, temp_filename, filename):
        # Delete temporary files
        try:
            os.remove(temp_filename)
            print(f"The file {temp_filename} has been deleted.")
        except FileNotFoundError:
            print(f"The file {temp_filename} was not found and could not be deleted.")
        except PermissionError:
            print(f"Permission denied to delete the file {temp_filename}.")
        except Exception as e:
            print(f"An error occurred: {e}.")
        try:
            shutil.rmtree("depth_maps")
            print(f'The directory "depth_maps" has been deleted successfully')
        except OSError as error:
            print(f'Error: {error.strerror}. Directory "depth_maps" not deleted.')
    else:
        print("Didn't delete anything because an error occured.")
