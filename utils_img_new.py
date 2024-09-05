#USE pipeline ENV
import numpy as np
import napari
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from skimage.measure import label,regionprops
from skimage import morphology, measure, io
import seaborn as sns
import itertools
import pandas as pd
# import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import math
from sklearn.neural_network import MLPClassifier
import statistics
import pickle
import os
import skimage
import tqdm
from datetime import datetime
from skimage.measure import regionprops_table
import re
import collections
class Img:

    def __init__(self, path):
        self.path = path
        self.img = skimage.io.imread(self.path)
        self.fov_number = re.search(r'fov_(\d+)', path).group(1)
        self.prediction = None


    def get_segmentation(self, seg_path):
        self.seg_path = seg_path
        self.masks = np.load(self.seg_path,allow_pickle=True)

    def show_img(self):
        viewer = napari.Viewer()
        viewer.add_image(self.img, name = 'phase')
        viewer.show(block=True) #wait until viewer window closes

    def show_masks(self):
        viewer = napari.Viewer()
        viewer.add_image(self.img, name = 'phase')
        viewer.add_labels(self.masks, name='masks')
        viewer.show(block=True) #wait until viewer window closes

    def add_label_layer(self):


        #mark manualy
        viewer = napari.Viewer()
        phase = viewer.add_image(self.img)
        if self.prediction is not None:
            viewer.add_image(self.prediction, colormap = 'inferno', blending = 'additive', name = 'prediction')
        masks = viewer.add_labels(self.masks)
        in_dev = viewer.add_labels(data=np.zeros([self.img.shape[0],self.img.shape[1]]).astype(int),name='in_dev')
        not_in_dev = viewer.add_labels(data=np.zeros([self.img.shape[0],self.img.shape[1]]).astype(int),name='not_in_dev')
        in_dev.brush_size = 3
        not_in_dev.brush_size = 3
        not_in_dev.label = 2
        viewer.show(block=True) #wait until viewer window closes
        self.in_dev = skimage.measure.label(in_dev.data)
        self.not_in_dev = skimage.measure.label(not_in_dev.data)


    def defind_masks_as(self):
        pairs=[]
        not_pairs=[]

        for label in np.unique(self.in_dev):
            temp_pair = np.unique(self.masks[self.in_dev == label])
            #The "ifs" is becuse somtimes when I manualy mark the cells I also mark the background.. (There should be a more sophisticated way of doing this..)
            if 0 in temp_pair:
                temp_pair = temp_pair[1:]
            if len(temp_pair) == 2:
                pairs.append(temp_pair)

        for label in np.unique(self.not_in_dev):
            temp_pair = np.unique(self.masks[self.not_in_dev == label])
            if 0 in temp_pair:
                temp_pair = temp_pair[1:]
            if len(temp_pair) == 2:
                not_pairs.append(temp_pair)

        self.pair = pairs
        self.not_pairs = not_pairs

    def add_saved_masked_labels(self, masks_file, pairs_file, not_pairs_file):
        self.masks = np.load(masks_file)
        self.pair = np.load(pairs_file)
        self.not_pairs = np.load(not_pairs_file)


    def create_candidate_pairs(self, chunk_size=64, img_size = 2048):
        all_comb = []
        # loop over the image with window_size = 64 (we miss the pairs that divide exactly in the border. we can add overlap after.
        for i in range(0, img_size - 1, chunk_size):
            for j in range(0, img_size - 1, chunk_size):
                # take all combination in window. if we use overlap we need to delete duplicates.
                cells_in = np.unique(self.masks[i:i + chunk_size, j:j + chunk_size])
                cells_in = cells_in[cells_in != 0]
                all_comb.extend(list(itertools.combinations(cells_in, 2)))
        self.all_comb = all_comb

    def calc_props(self):
        # Properties to calculate for each region
        properties = ['label','axis_major_length',
                      'axis_minor_length',
                      'extent', 'centroid',
                      'orientation','area']
        # Calculate region properties and convert to DataFrame
        prop_table = regionprops_table(self.masks, properties=properties)
        props_df = pd.DataFrame(prop_table).reset_index()
        props_df['fov'] = self.fov_number
        self.props_df = props_df

        # Define a function to calculate the pole coordinates
        def calculate_poles(row):
            y0, x0 = row['centroid-0'], row['centroid-1']
            orientation = row['orientation']
            axis_major_length = row['axis_major_length']
            # Pole 1
            x1 = x0 - math.sin(orientation) * 0.5 * axis_major_length
            y1 = y0 - math.cos(orientation) * 0.5 * axis_major_length

            # Pole 2
            x2 = x0 + math.sin(orientation) * 0.5 * axis_major_length
            y2 = y0 + math.cos(orientation) * 0.5 * axis_major_length

            # Calculate the starting and ending points of the major axis
            start_point = (x1, y1)
            end_point = (x2, y2)

            return pd.Series([start_point, end_point])

        # Apply the calculate_poles function to each row and store the results in new columns
        self.props_df[['start_point', 'end_point']] = self.props_df.apply(lambda row: calculate_poles(row), axis=1)


    def build_pairs_df(self):
        pairs_df = pd.DataFrame(self.all_comb, columns=["idx1", "idx2"])
        df = self.props_df

        # Duplicate the original DataFrame for the two indices in the pair
        df1 = df.add_prefix('idx1_')
        df2 = df.add_prefix('idx2_')

        # Set 'label' as the index for merging
        df1.set_index('idx1_label', inplace=True)
        df2.set_index('idx2_label', inplace=True)

        # Merge the DataFrames based on the indices in the pairs_df
        pairs_df = pairs_df.merge(df1, left_on='idx1', right_index=True).merge(df2, left_on='idx2', right_index=True)

        # Reset index of the result DataFrame
        pairs_df.reset_index(drop=True, inplace=True)

        # Calculate log2 fold change of 'axis_major_length' between pairs
        pairs_df['logFC_axis_major_length'] = np.abs(np.log2(pairs_df['idx1_axis_major_length'] / pairs_df['idx2_axis_major_length']))

        # Calculate the difference in 'orientation' between pairs
        pairs_df['orientation_difference'] = np.abs(pairs_df['idx1_orientation'] - pairs_df['idx2_orientation'])

        pairs_df['distance'] = np.sqrt((pairs_df['idx1_centroid-0'] - pairs_df['idx2_centroid-0'])**2 + (pairs_df['idx1_centroid-1'] - pairs_df['idx2_centroid-1'])**2)
        pairs_df['axis_dist'] = pairs_df['distance']*2 / (pairs_df['idx1_axis_major_length'] + pairs_df['idx2_axis_major_length'])

        # Remove the unnecessary columns
        # pairs_df = pairs_df.drop(columns=['idx1_axis_major_length', 'idx2_axis_major_length', 'idx1_orientation', 'idx2_orientation'])


        self.pairs_df = pairs_df.drop_duplicates()

        # Custom function to calculate distance between two points
        def calculate_distance(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return distance

        # Function to find minimum distance between two sets of points
        def find_min_distance(row):
            distances = [
                calculate_distance(row['idx1_start_point'], row['idx2_start_point']),
                calculate_distance(row['idx1_start_point'], row['idx2_end_point']),
                calculate_distance(row['idx1_end_point'], row['idx2_start_point']),
                calculate_distance(row['idx1_end_point'], row['idx2_end_point'])
            ]
            return min(distances)

        # Apply the function to create the new column
        self.pairs_df.loc[:, 'min_dist_poles'] = self.pairs_df.apply(find_min_distance, axis=1)


    def predict_division(self, model, demultimlex_file = None):
        X = self.pairs_df[['logFC_axis_major_length', 'orientation_difference', 'axis_dist','min_dist_poles']]
        predictions = model.predict(X)
        self.pairs_df.loc[:, 'is_dividing'] = predictions
        self.pairs_df['log_proba'] = model.predict_proba(X)[:,1]

        # Ensure that your DataFrame is sorted by 'log_proba' in descending order
        self.pairs_df.sort_values('log_proba', ascending=False, inplace=True)

        # A function to check if a row is a duplicate
        def is_duplicate(row):
            if counter[row['idx1']] > 0 or counter[row['idx2']] > 0:
                return True
            else:
                counter[row['idx1']] += 1
                counter[row['idx2']] += 1
                return False

        # Initialize a Counter
        counter = collections.Counter()

        # Apply is_duplicate function to each row to create a boolean mask for duplicates
        mask = self.pairs_df.apply(is_duplicate, axis=1)

        # Drop the duplicates
        self.pairs_df = self.pairs_df[~mask]

        if demultimlex_file:
            pass

        # Add "is_div" column
        combined_values = []
        combined_values.extend(self.pairs_df.loc[self.pairs_df['is_dividing'] == 1, 'idx1'].tolist())
        combined_values.extend(self.pairs_df.loc[self.pairs_df['is_dividing'] == 1, 'idx2'].tolist())
        self.props_df['is_div'] = self.props_df.label.isin(combined_values)

        # Add pair column to props_df
        self.props_df['pair'] =None

        pairs_dictionary = self.pairs_df.loc[self.pairs_df.is_dividing == 1, ['idx1', 'idx2']].set_index('idx1')['idx2'].to_dict()
        # Loop through the dictionary and update the dataframe
        for key, value in pairs_dictionary.items():
            # Create a mask where key is present
            mask = (self.props_df['label'] == key)

            # Replace 'pair' with value where key is present
            self.props_df.loc[mask, 'pair'] = value

            # Add the value of axis_major_length where key is present
            self.props_df.loc[mask, 'axis_major_length'] += self.props_df.loc[self.props_df.label == value,'axis_major_length'].values[0]
            if 'dapi_peaks' in self.props_df.columns:
                self.props_df.loc[mask, 'dapi_peaks'] += self.props_df.loc[self.props_df.label == value,'dapi_peaks'].values[0]

        # Use the pairs_dictionary as LUT to replace the pairs mask
        new_dict = {value: key for key, value in pairs_dictionary.items()}
        self.pairs_mask = np.vectorize(lambda x: new_dict.get(x, x))(self.masks)



        #drop idx2
        idx2 = self.pairs_df['idx2'][self.pairs_df.is_dividing == 1].tolist()
        self.props_df_indevedual = self.props_df
        self.props_df = self.props_df.loc[~self.props_df.label.isin(idx2)]

    def add_cell_by_gene(self, path):
        # Read cell by gene file
        cell_by_gene = pd.read_csv(path, sep = '\t')

        # Merge with props_df_indevedual
        df = cell_by_gene.merge(self.props_df_indevedual, left_on='cell_id', right_on='label')

        # Filter rows where 'pair' is not None
        pairs_df = df.dropna(subset=['pair'])

        # Find valid pairs
        valid_pairs = []
        for _, row in pairs_df.iterrows():
            label, sample_name, pair = row['label'], row['sample_name'], row['pair']
            # Check if the pair exists in the same sample_name with matching labels
            if df[(df['sample_name'] == sample_name) & (df['label'] == pair)].shape[0] > 0:
                valid_pairs.append(label)
                valid_pairs.append(pair)

        # Keep only unique entries in case there are duplicates
        valid_pairs = list(set(valid_pairs))

        # Step 3: Filter the original DataFrame to keep rows where the label is in valid_pairs
        self.props_df_indevedual = df[df['label'].isin(valid_pairs) | df['pair'].isnull()]

        filtered_pairs_df = self.pairs_df[
            self.pairs_df['idx1'].isin(df.index) &
            self.pairs_df['idx2'].isin(df.index)
            ]

        self.pairs_df = filtered_pairs_df

        
        
    def show_prob(self, open_napari = True):
        df = self.pairs_df
        mask =self.masks

        # melt the DataFrame so it will have only one column of idx
        melted = pd.melt(df, id_vars=['log_proba'], value_vars=['idx1', 'idx2'])

        # set log_proba as the index
        melted.set_index('value', inplace=True)

        # make a dictionary out of melted dataframe
        lut = melted['log_proba'].to_dict()

        # create new matrix based on mask and lut
        new_matrix = np.vectorize(lut.get, otypes=[float])(mask)
        new_matrix = np.nan_to_num(new_matrix)
        self.prediction = new_matrix

        if open_napari:
            viewer = napari.Viewer()
            phase = viewer.add_image(self.img)
            masks = viewer.add_labels(self.masks)
            pred = viewer.add_image(new_matrix)

            viewer.show(block=True) #wait until viewer window closes



    def build_all_props_df(self):
            '''
            build properties df from label matrix, return df.
            '''

            props=measure.regionprops_table(self.masks,properties=['label',
                                                                   'area',
                                                                   'axis_major_length',
                                                                   'axis_minor_length',
                                                                   'centroid',
                                                                   'extent',
                                                                   'orientation',
                                                                   'eccentricity',
                                                                   'equivalent_diameter_area',
                                                                   'feret_diameter_max',
                                                                   'perimeter',
                                                                   'solidity',
                                                                   'bbox'])
            props_data = pd.DataFrame(props)
            props_data.index = props_data.label
            self.props_data = props_data


def plot_confusion_mat(y_test, predicted):
    '''
    I use this finction to evaluate the model preformence
    '''
    
    results = confusion_matrix(y_test, predicted)
    strings = strings = np.asarray([['TN = ', 'FP = '],
                                    ['FN = ', 'TP = ']])

    labels = (np.asarray(["{0} {1:.3f}".format(string, value)
                          for string, value in zip(strings.flatten(),
                                                   results.flatten())])
             ).reshape(2, 2)

    fig, ax = plt.subplots()
    ax=sns.heatmap(results, annot=labels, fmt="", ax=ax,
                   xticklabels=['not in division','in devision'],
                  yticklabels=['not in division','in devision'])
    ax.set_xlabel('True labels', fontsize=10)
    ax.set_ylabel('predicted labels', fontsize=10)

    plt.show()
    
def random_forest(X_train, X_test, y_train, y_test, plot = True, random_state=42):
#build random forest model

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=random_state)
    #fit
    rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Mean accuracy score: {accuracy:.3}')
    if plot:
        plot_confusion_mat(y_test,predicted)
    return rf

def plot_box_by_cells_idx(img, masks, cell1, cell2, space=5, print_coor = False, plot_img = False):
    '''
    Retrieves bounding boxes based on cell labels directly from `masks`
    '''
    # Extract properties in a structured array
    props_table = regionprops_table(masks, properties=('label', 'bbox'))

    # Convert to DataFrame for easier querying
    df_props = pd.DataFrame(props_table)

    # Query for the specific cells
    prop1_row = df_props[df_props['label'] == cell1]
    prop2_row = df_props[df_props['label'] == cell2]

    if prop1_row.empty or prop2_row.empty:
        print("One of the cells was not found.")
        return None

    # Extract bounding box information
    prop1_bbox = prop1_row[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values[0]
    prop2_bbox = prop2_row[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values[0]

    # Calculate min and max for rows and columns
    min_r = min(prop1_bbox[0], prop2_bbox[0])
    min_c = min(prop1_bbox[1], prop2_bbox[1])
    max_r = max(prop1_bbox[2], prop2_bbox[2])
    max_c = max(prop1_bbox[3], prop2_bbox[3])

    # Extract the relevant areas
    img_mat = img[min_r - space : max_r + space, min_c - space : max_c + space]
    mask_mat = masks[min_r - space : max_r + space, min_c - space : max_c + space]

    print(min_r, max_r, min_c, max_c) if print_coor else None
    if plot_img:
        return img_mat
    else:
        return mask_mat

def create_training_set(image_directory_path, segmentation_directory_base_path, fov_to_include = range(100), hyb = 0):
    """
    Creates a training set by matching .tif images with their corresponding segmentation files.

    Parameters:
    - image_directory_path: Path to the directory containing .tif images.
    - segmentation_directory_base_path: Base path to the directory containing segmentation files.

    Returns:
    - A list of tuples, where each tuple contains the paths to an image file and its corresponding segmentation file.
    """
    image_set = []

    # Loop over each file in the image directory
    for filename in os.listdir(image_directory_path):
        # Construct the full file path
        file_path = os.path.join(image_directory_path, filename)

        # Check if the current item is a file and ends with 'phase.tif'
        if os.path.isfile(file_path) and filename.endswith('phase.tif'):
            # Extract fov and hyb values from the filename
            parts = filename.split('_')
            if parts[-1] == '0.phase.tif':
                fov = parts[1]
                if int(fov) in fov_to_include:
                    # Construct the path to the segmentation file
                    segmentation_dir = f"fov_{fov}_hyb_{hyb}"
                    segmentation_file = f"fov_{fov}_hyb_{hyb}.seg.npy"
                    segmentation_path = os.path.join(segmentation_directory_base_path, segmentation_dir, segmentation_file)

                    # Check if the segmentation file exists
                    if os.path.exists(segmentation_path):
                        image_set.append((file_path, segmentation_path))
                    else:
                        print(f"Segmentation file not found for {file_path} \n {segmentation_path}")

    return image_set
   
   
def add_value_column(row, pairs, not_pairs):
   # Function to check pairs with order-independent comparison
    pair = tuple(sorted([row['idx1'], row['idx2']]))  # Convert to tuple for immutable and order-independent comparison
    if any(pair == tuple(sorted(p)) for p in pairs):  # Use tuple for comparison
        return 1
    elif any(pair == tuple(sorted(np)) for np in not_pairs):  # Use tuple for comparison
        return 0
    else:
        return None