import numpy as np
from PIL import Image
from skimage.measure import label,regionprops_table
import pandas as pd
import skimage


    
class Img:
    
    def __init__(self, path, fov, automation_summary_path, hybs):

        df = pd.read_excel(automation_summary_path)
        
        # Initialize a list to store the extracted_tuples tuples
        extracted_tuples = []
        
        # Loop through the dataframe rows
        for index, row in df.iterrows():
            if index in hybs:
                # Get the description based on the hybs dictionary
                desc = hybs[index]
                # Loop through the columns for the dyes
                for dye in ['A647', 'A488', 'A550']:
                    # Create a tuple and append it to the extracted_tuples list
                    extracted_tuples.append((f'{desc}', dye, index, row[dye]))
                
        self.extracted_tuples = extracted_tuples
        self.path = path
        self.fov_number = fov
        self.img = skimage.io.imread(fr'{path}/image_projections/fov_{fov}_hyb_0.phase.tif')
        self.masks = np.load(fr'{path}\segmentation\fov_{fov}_hyb_0\fov_{fov}_hyb_0.seg.npy',allow_pickle=True)
        
        # Extract properties in a structured array
        self.props_table = regionprops_table(self.masks, properties=('label', 'bbox'))

        # Convert to DataFrame for easier querying
        self.df_props = pd.DataFrame(self.props_table)
        
        # Dynamically assign images based on extracted_tuples
        for target, channel, hyb, prob in extracted_tuples:
            # For bc1 and bc2, we also need to keep track of the index for each hybridization
            index = extracted_tuples.count((target, channel, hyb, prob))  # This approach assumes sequential ordering
            attr_name = f"{target}_{channel}_{hyb}"

            
            # Construct file path and load the image
            file_path = fr'{path}/image_projections/fov_{fov}_hyb_{hyb}.{channel}.tif'
            setattr(self, attr_name, np.array(Image.open(file_path)))
    
    def plot_random_index(self, idx, layer, space = 5, print_coor = False):
        '''
        Retrieves bounding boxes based on cell labels directly from `masks`
        '''

        # Query for the specific cells
        prop = self.df_props[self.df_props['label'] == idx]

        # Extract bounding box information
        prop_bbox = prop[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values[0]

        # Calculate min and max for rows and columns
        min_r = prop_bbox[0]
        min_c = prop_bbox[1]
        max_r = prop_bbox[2]
        max_c = prop_bbox[3]
        # Extract the relevant areas
        img_mat = layer[min_r - space : max_r + space, min_c - space : max_c + space]

        print(min_r, max_r, min_c, max_c) if print_coor else None
        
        return img_mat

