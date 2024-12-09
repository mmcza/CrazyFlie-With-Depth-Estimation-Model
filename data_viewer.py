import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
import glob

class DataViewer:
    def __init__(self, base_folder, datasets):
        self.base_folder = base_folder
        self.datasets = datasets
        self.file_dict = {}

        for dataset in self.datasets:
            folder_camera = os.path.join(base_folder, dataset, 'camera')
            folder_depth_camera = os.path.join(base_folder, dataset, 'depth_camera')
            folder_distance_sensor = os.path.join(base_folder, dataset, 'distance_sensor')

            files_camera = self.get_files(folder_camera, 'c_')
            files_depth_camera = self.get_files(folder_depth_camera, 'd_')
            files_distance_sensor = self.get_files(folder_distance_sensor, 'ds_')

            common_timestamps = set(files_camera.keys()) & set(files_depth_camera.keys())
            if files_distance_sensor:
                common_timestamps &= set(files_distance_sensor.keys())

            for ts in common_timestamps:
                self.file_dict[(dataset, ts)] = {
                    'camera': files_camera[ts],
                    'depth_camera': files_depth_camera[ts],
                    'distance_sensor': files_distance_sensor.get(ts, None)
                }

        self.entries = sorted(self.file_dict.keys(), key=lambda x: (x[0], x[1]))

        if not self.entries:
            print("No matching files found across datasets.")
            exit()

        self.current_index = 0
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)

        self.create_buttons()
        self.create_textbox()
        self.display_current()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_files(self, folder, prefix):
        if not os.path.exists(folder):
            return {}

        extension = '.txt' if prefix == 'ds_' else '.png'
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
        dataset, timestamp = self.entries[self.current_index]
        file_set = self.file_dict[(dataset, timestamp)]
        camera_path = file_set['camera']
        depth_camera_path = file_set['depth_camera']
        distance_sensor_path = file_set['distance_sensor']

        try:
            img_camera = Image.open(camera_path)
            img_depth = Image.open(depth_camera_path)
        except Exception as e:
            print(f"Error loading images for dataset '{dataset}' and timestamp {timestamp}: {e}")
            return

        self.axs[0, 0].clear()
        self.axs[0, 0].imshow(img_camera)
        self.axs[0, 0].set_title(f"{dataset.capitalize()} - Camera Image ({os.path.basename(camera_path)})")
        self.axs[0, 0].axis('off')

        self.axs[0, 1].clear()
        self.axs[0, 1].imshow(img_depth, cmap='gray')
        self.axs[0, 1].set_title(f"{dataset.capitalize()} - Depth Image ({os.path.basename(depth_camera_path)})")
        self.axs[0, 1].axis('off')

        sensor_data = "N/A"
        if distance_sensor_path:
            try:
                with open(distance_sensor_path, 'r') as f:
                    sensor_data = f.read().strip()
            except Exception as e:
                print(f"Error reading distance sensor data for dataset '{dataset}' and timestamp {timestamp}: {e}")

        self.axs[1, 0].clear()
        self.axs[1, 0].text(0.5, 0.5, sensor_data, fontsize=12, ha='center', va='center')
        self.axs[1, 0].set_title("Distance Sensor Data")
        self.axs[1, 0].axis('off')

        self.axs[1, 1].clear()
        self.axs[1, 1].text(
            0.5, 0.5,
            f"Set {self.current_index + 1} of {len(self.entries)}\nDataset: {dataset}\nTimestamp: {timestamp}",
            fontsize=12, ha='center', va='center'
        )
        self.axs[1, 1].axis('off')

        self.fig.canvas.draw_idle()

    def create_buttons(self):
        axprev = plt.axes([0.1, 0.1, 0.1, 0.05])
        axnext = plt.axes([0.8, 0.1, 0.1, 0.05])
        axdelete = plt.axes([0.45, 0.1, 0.1, 0.05])

        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')
        self.bdelete = Button(axdelete, 'Delete')

        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.bdelete.on_clicked(self.delete_current)

    def create_textbox(self):
        axbox = plt.axes([0.4, 0.275, 0.2, 0.05])
        self.text_box = TextBox(axbox, 'Go To Index', initial="")
        self.text_box.on_submit(self.go_to_sample)

    def go_to_sample(self, text):
        try:
            index = int(text) - 1
            if 0 <= index < len(self.entries):
                self.current_index = index
                self.display_current()
            else:
                print(f"Index out of range. Please enter a value between 1 and {len(self.entries)}.")
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

    def next(self, event=None):
        if self.current_index < len(self.entries) - 1:
            self.current_index += 1
            self.display_current()

    def prev(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current()

    def delete_current(self, event=None):
        if not self.entries:
            return
        dataset, timestamp = self.entries[self.current_index]
        file_set = self.file_dict.get((dataset, timestamp))

        if not file_set:
            return

        try:
            for key, path in file_set.items():
                if path and os.path.exists(path):
                    os.remove(path)

            del self.file_dict[(dataset, timestamp)]
            del self.entries[self.current_index]

            if self.current_index >= len(self.entries):
                self.current_index = len(self.entries) - 1

            if self.entries:
                self.display_current()
            else:
                plt.close(self.fig)
        except Exception as e:
            print(f"Error deleting files for dataset '{dataset}' and timestamp {timestamp}: {e}")

    def on_key_press(self, event):
        if event.key == 'd':
            self.delete_current()
        elif event.key == 'n':
            self.next()
        elif event.key == 'p':
            self.prev()
        elif event.key == 'g':
            try:
                index = int(input("Enter sample index to go to: ")) - 1
                if 0 <= index < len(self.entries):
                    self.current_index = index
                    self.display_current()
                else:
                    print(f"Index out of range. Value between 1 and {len(self.entries)}.")
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

def main():
    base_folder = r"C:\Users\kubac\Documents\GitHub\gra\CrazyFlie-With-Depth-Image-Model\crazyflie_images"
    datasets = ["warehouse", "cafe"]

    viewer = DataViewer(
        base_folder=base_folder,
        datasets=datasets
    )
    plt.show()

if __name__ == "__main__":
    main()











