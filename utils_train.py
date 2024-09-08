import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import glob
import click2
import warnings
warnings.filterwarnings('ignore')


def concat_files(fov_list, hyb_list, directory_path):
    # Create an empty DataFrame for the final concatenated result
    concatenated_df = pd.DataFrame()

    # Create a pattern to match files in the directory
    file_pattern = f"{directory_path}/fov_*.txt"

    # Loop over all files matching the pattern
    for file in glob.glob(file_pattern):
        # Extract `fov`, `hyb`, and `prob` from the file name
        parts = file.split(r'\\')[-1].split('_')
        fov = int(parts[-3])
        hyb = int(parts[-1].split('.')[0])
        prob = file.split('.')[-2]
        # Check if the file's `fov` and `hyb` are in the provided lists
        if fov in fov_list and hyb in hyb_list:
            # Read the file into a DataFrame
            df = pd.read_csv(file, index_col=0, sep = '\t')  # Assuming the index is in the first column

            # Prefix columns with `hyb`, `fov`, and `prob`
            df.columns = [f'hyb_{hyb}_{prob}_fov{fov}_{col}' for col in df.columns]

            # Concatenate with the main DataFrame
            if concatenated_df.empty:
                concatenated_df = df
            else:
                concatenated_df = pd.concat([concatenated_df, df], axis=1)

    return concatenated_df



def get_probes_and_bc(filename, hyb_list, channel):
    # Load the data from the specified Excel file
    data = pd.read_excel(filename)

    # Filter data based on the specified hybridizations
    filtered_data = data[data['hyb'].isin(hyb_list)]

    # Extract and return the probes and their notes for the specified channel
    probs = [row[channel] for index, row in filtered_data.iterrows()]
    bc = [row[-1] for index, row in filtered_data.iterrows()]
    return probs, bc

def create_model(df, automation_summary_path, channel = 'A647', hyb_list = [2, 3, 4], test_size=0.2):


# Assuming get_probes_and_bc is defined elsewhere and returns a list of probes and barcode identifiers
    probs, bc = get_probes_and_bc(automation_summary_path, channel=channel, hyb_list=hyb_list)
    cols_to_include = [col for col in df.columns if channel in col]
    probs = probs + [(i + '.1') for i in probs]

    for prob in probs:
        cols_to_include.append(prob)
    #print(cols_to_include)
    one_color = df[cols_to_include]

    # Splitting the DataFrame into features and targets
    X = one_color.iloc[:, len(hyb_list):]  # Features: All columns except the first 3
    y = one_color.iloc[:, :len(hyb_list)]  # Targets: The first 3 columns

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=18)

    # Initializing the Random Forest model
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fitting the model to the training data
    random_forest_model.fit(X_train, y_train)

    # Making predictions on the test set
    predictions = random_forest_model.predict(X_test)
    predictions_prob = random_forest_model.predict_proba(X_test)

    mae = mean_absolute_error(y_test, predictions, multioutput='raw_values')

    list_arrays = []
    for i in predictions_prob:
        list_arrays.append(1 - i[:, 0])
    probabilities = pd.DataFrame(list_arrays).T

    pred_df = df.iloc[y_test.index]
    pred_df.reset_index(inplace=True)

    # Generate column names based on hyb_list, bc, and probs
    col_names = [
        f'prob_{bc[i]}_hyb{hyb_list[i]}_{channel}_{probs[i]}'
        for i in range(len(hyb_list))
    ]
    pred_df[col_names] = probabilities
    
    pred_df.rename(columns={'Index': 'cell_id'}, inplace=True)

    print("mean absolute error for each target variable:", mae)
    return random_forest_model, pred_df


def plot_by_idx(idx, temp, to_show, titels, save_path = None):
    plots = []
    for j in to_show:
        plots.append(temp.plot_random_index(idx, j, space=5))
    outline = click2.extract_outlines(temp.plot_random_index(idx, temp.masks, space=5))
    click2.selection(plots, titels, save_path, outline, idx, save_name=save_path)
    

def plot_eval(eval_df, true_cols, pred_cols, delta_cols):
    separator = pd.DataFrame(np.nan, index=eval_df.index, columns=["Sep"], dtype='float')
    group_1 = eval_df[true_cols]
    group_2 = eval_df[pred_cols]
    group_3 = eval_df[delta_cols]

    # Concatenate with separator
    data_for_heatmap = pd.concat([group_1, separator, group_2, separator, group_3], axis=1)
    problematic_idx = []
    problematic_cell_id = []

    # Function to handle click event
    def onclick(event):
        # Calculate the row index based on the y-coordinate of the click event
        row_index = int(np.floor(event.ydata))
        cell_id = eval_df.loc[row_index,'cell_id']
        problematic_idx.append(row_index)
        problematic_cell_id.append(cell_id)


        if row_index < len(data_for_heatmap):
            print(f"Clicked on row index: {row_index}, cell ID: {cell_id}")
        else:
            print("Clicked outside of rows")

    data_for_heatmap = pd.concat([group_1, separator, group_2, separator, group_3], axis=1)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data_for_heatmap, annot=True, cmap='viridis', cbar=False, linewidths=2, linecolor='black')

    # Connect the click event to the onclick function
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    plt.xlim(0, data_for_heatmap.shape[1])
    plt.savefig('heatmap.png')
    plt.show()
    
    return data_for_heatmap, problematic_idx, problematic_cell_id
    

    
def predict(model, directory_path, demult_path, automation_summary_path, cutoff = 0.5, channel = 'A647', fov_list=[5], hyb_list=[2,3,4]):


    whole_cell_data = concat_files(fov_list=fov_list, hyb_list=hyb_list, directory_path=directory_path)
    whole_cell_data = whole_cell_data.reindex(sorted(whole_cell_data.columns), axis=1)
    demult = pd.read_csv(demult_path, sep='\t')
    df = whole_cell_data.merge(demult, left_on='cell_id', right_on='cell_id').drop(['cell_id', 'sample_name'], axis=1)

    probs, bc = get_probes_and_bc(automation_summary_path, channel=channel, hyb_list=hyb_list)
    cols_to_include = [col for col in df.columns if channel in col]
    probs = probs + [(i + '.1') for i in probs]

    for prob in probs:
        cols_to_include.append(prob)
    one_color = df[cols_to_include]
    #print(cols_to_include)

    predictions = model.predict(one_color)
    predictions_prob = model.predict_proba(one_color)

    list_arrays = []
    for i in predictions_prob:
        list_arrays.append(1- i[:,0])
    probabilities = pd.DataFrame(list_arrays).T

    cutoff -= 0.5
    probabilities -= cutoff
    # Set negative values to 0
    probabilities = probabilities.clip(lower=0)

    hyb_list = hyb_list
    # Generate column names based on hyb_list, bc, and probs
    col_names = [
        f'prob_{bc[i]}_hyb{hyb_list[i]}_{channel}_{probs[i]}'
        for i in range(len(hyb_list))
    ]
    one_color[col_names] = probabilities
    one_color['cell_id'] = demult['cell_id']

    return one_color
    
    


