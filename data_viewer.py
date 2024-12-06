import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import glob

class DataViewer:
    def __init__(self, folder_camera, folder_depth_camera, folder_distance_sensor):
        self.folder_camera = folder_camera
        self.folder_depth_camera = folder_depth_camera
        self.folder_distance_sensor = folder_distance_sensor

        self.files_camera = self.get_files(self.folder_camera, 'c_')
        self.files_depth_camera = self.get_files(self.folder_depth_camera, 'd_')
        self.files_distance_sensor = self.get_files(self.folder_distance_sensor, 'ds_')

        common_timestamps = set(self.files_camera.keys()) & set(self.files_depth_camera.keys()) & set(self.files_distance_sensor.keys())
        self.timestamps = sorted(common_timestamps)

        if not self.timestamps:
            print("No matching files with identical timestamps found.")
            exit()

        self.current_index = 0
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.create_buttons()
        self.display_current()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_files(self, folder, prefix):
        extension = '.txt' if 'distance_sensor' in folder else '.png'
        pattern = os.path.join(folder, f"{prefix}*{extension}")
        files = glob.glob(pattern)
        file_dict = {}
        for file in files:
            basename = os.path.basename(file)
            parts = basename.split('_', 1)
            if len(parts) != 2:
                continue
            timestamp = os.path.splitext(parts[1])[0]
            file_dict[timestamp] = file
        return file_dict

    def display_current(self):
        timestamp = self.timestamps[self.current_index]
        camera_path = self.files_camera[timestamp]
        depth_camera_path = self.files_depth_camera[timestamp]
        distance_sensor_path = self.files_distance_sensor[timestamp]

        try:
            img_camera = Image.open(camera_path)
            img_depth = Image.open(depth_camera_path)
        except Exception as e:
            print(f"Error loading images for timestamp {timestamp}: {e}")
            return

        self.axs[0, 0].clear()
        self.axs[0, 0].imshow(img_camera)
        self.axs[0, 0].set_title(f"Camera Image ({os.path.basename(camera_path)})")
        self.axs[0, 0].axis('off')

        self.axs[0, 1].clear()
        self.axs[0, 1].imshow(img_depth, cmap='gray')
        self.axs[0, 1].set_title(f"Depth Image ({os.path.basename(depth_camera_path)})")
        self.axs[0, 1].axis('off')

        try:
            with open(distance_sensor_path, 'r') as f:
                sensor_data = f.read().strip()
        except Exception as e:
            print(f"Error reading distance sensor data for timestamp {timestamp}: {e}")
            sensor_data = "N/A"

        self.axs[1, 0].clear()
        self.axs[1, 0].text(0.5, 0.5, sensor_data, fontsize=12, ha='center', va='center')
        self.axs[1, 0].set_title("Distance Sensor Data")
        self.axs[1, 0].axis('off')

        self.axs[1, 1].clear()
        self.axs[1, 1].text(
            0.5, 0.5,
            f"Set {self.current_index + 1} of {len(self.timestamps)}\nTimestamp: {timestamp}",
            fontsize=12, ha='center', va='center'
        )
        self.axs[1, 1].axis('off')

        self.fig.canvas.draw_idle()

    def create_buttons(self):
        axprev = plt.axes([0.1, 0.02, 0.1, 0.05])
        axnext = plt.axes([0.8, 0.02, 0.1, 0.05])
        axdelete = plt.axes([0.45, 0.02, 0.1, 0.05])

        self.bprev = Button(axprev, 'Poprzedni')
        self.bnext = Button(axnext, 'Następny')
        self.bdelete = Button(axdelete, 'Usuń')

        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.bdelete.on_clicked(self.delete_current)

    def next(self, event=None):
        if self.current_index < len(self.timestamps) - 1:
            self.current_index += 1
            self.display_current()

    def prev(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current()

    def delete_current(self, event=None):
        if not self.timestamps:
            return
        timestamp = self.timestamps[self.current_index]
        camera_path = self.files_camera.get(timestamp)
        depth_camera_path = self.files_depth_camera.get(timestamp)
        distance_sensor_path = self.files_distance_sensor.get(timestamp)

        if not all([camera_path, depth_camera_path, distance_sensor_path]):
            print(f"Incomplete file set for timestamp {timestamp}. Skipping deletion.")
            return

        try:
            os.remove(camera_path)
            os.remove(depth_camera_path)
            os.remove(distance_sensor_path)
            print(f"Deleted set: {timestamp}")

            del self.files_camera[timestamp]
            del self.files_depth_camera[timestamp]
            del self.files_distance_sensor[timestamp]
            del self.timestamps[self.current_index]

            if self.current_index >= len(self.timestamps):
                self.current_index = len(self.timestamps) - 1

            if self.timestamps:
                self.display_current()
            else:
                print("No more sets to display.")
                plt.close(self.fig)
        except Exception as e:
            print(f"Error deleting files for timestamp {timestamp}: {e}")

    def on_key_press(self, event):
        if event.key == 'd':
            self.delete_current()

def main():
    viewer = DataViewer(
        folder_camera="crazyflie_images/camera",
        folder_depth_camera="crazyflie_images/depth_camera",
        folder_distance_sensor="crazyflie_images/distance_sensor",
    )
    plt.show()

if __name__ == "__main__":
    main()







