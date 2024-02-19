import subprocess


def run_ffmpeg_command(cmd):
    """Runs an ffmpeg command using subprocess.

    Args:
        cmd (list): The ffmpeg command arguments as a list.

    Returns:
        bool: True if the command executed successfully, False otherwise.
    """
    try:
        # Execute the command
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            encoding="utf-8",
        )
        print("FFmpeg Output:", process.stdout)
        print("FFmpeg Error Output:", process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        # Handle errors in the called ffmpeg process
        print(f"An error occurred: {e.stderr}")
        return False
    except Exception as e:
        # Handle other exceptions, such as FileNotFoundError
        print(f"An exception occurred: {e}")
        return False
