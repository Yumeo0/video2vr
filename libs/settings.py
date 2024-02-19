import json
import os
from typing import NamedTuple


class AppSettings(NamedTuple):
    batch_size: int = 50
    ai_model: str = "LiheYoung/depth-anything-large-hf"
    depth_offset: int = 15


class SettingsManager:
    def __init__(self, filename="settings.json"):
        self.filename = filename
        self.settings = AppSettings()  # Initialize with default settings
        self.load_settings()

    def load_settings(self):
        if not os.path.isfile(self.filename):
            self.save_settings()
            return

        with open(self.filename, "r") as f:
            try:
                loaded_settings = json.load(f)
                self.settings = self.settings._replace(**loaded_settings)
            except json.JSONDecodeError:
                self.save_settings()

    def save_settings(self):
        with open(self.filename, "w") as f:
            # Only save back the settings that are in the AppSettings NamedTuple
            json.dump(self.settings._asdict(), f, indent=4)

    def set(self, key, value):
        if hasattr(self.settings, key):
            # Use **{} syntax to unpack the key-value into the _replace method
            self.settings = self.settings._replace(**{key: value})
            self.save_settings()
            print("Saved " + key)


settings_manager = SettingsManager()
