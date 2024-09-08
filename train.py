import pandas as pd
import click2
import cv2
import numpy as np
import tqdm

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
    


def run(temp, titels, automation_summary_path, to_show, save_path="selections.csv", session_length = 300, space = 5, idx_list = None):
    idx_to_choose = np.unique(temp.masks) if idx_list is None else idx_list
       
    for i in tqdm.tqdm(range(session_length)):
        idx = np.random.choice(idx_to_choose, 1, replace=False)[0]
        plots = []
        if idx == 0:
            print('0')
            continue
        for j in to_show:
            plots.append(temp.plot_random_index(idx, j, space=space))
        outline = extract_outlines(temp.plot_random_index(idx, temp.masks, space=space))
        click2.selection(plots, titels, save_path, outline, idx)

