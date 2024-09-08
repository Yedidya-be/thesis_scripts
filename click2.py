import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import pandas as pd
import os
import cv2


def save_choices_to_csv(selections, titles, index, filename="selections.csv"):
    """Save the selections for a session to a CSV file."""
    # Check if file exists and is not empty (more robust check for whether to include headers)
    file_exists = os.path.isfile(filename)
    file_empty = os.path.getsize(filename) == 0 if file_exists else True
    
    df = pd.DataFrame([selections], columns=titles, index=[index])
    
    # Write DataFrame to CSV
    # If the file doesn't exist or is empty, write headers. Otherwise, don't.
    df.to_csv(filename, mode='a', header=not file_exists or file_empty, index_label="Index")



class InteractiveImageSelector:
    def __init__(self, arrays, titles, save_path="selections.csv", outline=None, index=None):
        self.arrays = arrays
        self.selected = [False] * len(arrays)
        self.titles = titles
        self.outline = outline
        self.index = index  
        self.save_path = save_path
        
        
        n_rows = int(len(arrays)/3)
        # Setup figure and axes for a 3x4 grid
        self.fig, self.base_axes = plt.subplots(n_rows, 3, figsize=(20, 15))
        self.base_axes = self.base_axes.flatten()  # Flatten the array to iterate over it
        self.slider_axes_vmin = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider_axes_vmax = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.button_axes = plt.axes([0.85, 0.01, 0.1, 0.075])
        
        # Define colormaps for each column (assuming RGB for three columns)
        self.colormaps = ['Reds', 'Blues','Greens']

        # Initial vmin and vmax values based on the data range
        all_values = np.concatenate([arr.ravel() for arr in arrays])
        self.vmin, self.vmax = np.min(all_values), np.max(all_values)

        # Display images with different colormaps for each column
        for i, ax in enumerate(self.base_axes[:len(arrays)]):
            cmap = self.colormaps[i % 3]  # Select colormap based on column
            img = ax.imshow(arrays[i], cmap=cmap, vmin=self.vmin, vmax=self.vmax)
            ax.set_title(self.titles[i])
            ax.axis('off')

            # Overlay the outline if it's provided
            if self.outline is not None:
                # Ensure the outline is added on top of the image. Adjust alpha for transparency as needed.
                ax.imshow(self.outline*self.vmax, cmap='Greys', alpha=0.3, vmin=0, vmax=self.vmax)

            # Add a patch as a visible border indicator
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=4, edgecolor='none', facecolor='none')
            ax.add_patch(rect)

        # Hide any unused axes in the grid
        for i in range(len(arrays), len(self.base_axes)):
            self.base_axes[i].axis('off')

        # Sliders
        self.vmin_slider = Slider(self.slider_axes_vmin, 'Vmin', self.vmin, self.vmax, valinit=self.vmin, orientation='horizontal')
        self.vmax_slider = Slider(self.slider_axes_vmax, 'Vmax', self.vmin, self.vmax, valinit=self.vmax/2, orientation='horizontal')
        self.vmin_slider.on_changed(self.update_vmin_vmax)
        self.vmax_slider.on_changed(self.update_vmin_vmax)

        # Button
        self.btn = Button(self.button_axes, 'Done', color='0.85', hovercolor='0.95')
        self.btn.on_clicked(self.confirm_selection)

        # Event connections
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def update_vmin_vmax(self, val):
        for ax in self.base_axes:
            for img in ax.get_images():
                img.set_clim(vmin=self.vmin_slider.val, vmax=self.vmax_slider.val)
        plt.draw()

    def on_click(self, event):
        for i, ax in enumerate(self.base_axes):
            if ax in self.base_axes and ax == event.inaxes:
                self.selected[i] = not self.selected[i]
                rect = ax.patches[0]
                rect.set_edgecolor('lime' if self.selected[i] else 'none')
                plt.draw()
                break

    def confirm_selection(self, event):
        # Create a dict with titles as keys and selections (1 or 0) as values
        selections = {title: (1 if selected else 0) for title, selected in zip(self.titles, self.selected)}
        save_choices_to_csv(selections, self.titles, self.index, self.save_path)
        plt.close(self.fig)


    def display(self):
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

def selection(arrays, titles, save_path="selections.csv", outline = None, index = None, save_name = None):
    selector = InteractiveImageSelector(arrays, titles, save_path, outline, index)
    mng = plt.get_current_fig_manager()
    mng.window.geometry("2200x1500+50+100")
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def extract_outlines(image):
    """
    Extracts the outlines of segmented regions in an image.

    Parameters:
        image (np.array): A grayscale image where different segments are represented by different intensities.

    Returns:
        np.array: An image with only the outlines of the segmented regions.
    """
    # Ensure the image is in the correct format (grayscale)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find all unique labels in the image
    unique_labels = np.unique(image)

    # Create an empty image to draw the outlines
    outlines_image = np.zeros_like(image, dtype=np.uint8)  # Ensure data type is np.uint8

    for label in unique_labels:
        # Create a binary mask for the current label
        mask = np.where(image == label, 1, 0).astype(np.uint8)

        # Find contours for the current label
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the outlines image
        cv2.drawContours(outlines_image, contours, -1, 255, 1)  # Use 255 for white

    return outlines_image