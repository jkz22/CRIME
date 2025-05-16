import numpy as np
import re
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# LIME processing functions

def spectra_explainer(data, spectra_length, mode):
    # Initialize a LIME explainer object
    # Note: You might need to adjust the feature names and class names according to your dataset
    explainer = LimeTabularExplainer(training_data=np.array(data), # Use your data here
                                                    mode=mode, # or 'classification' based on your model
                                                    feature_names = [f'*{i}*' for i in range(spectra_length)],
                                                    discretize_continuous=True)
    return explainer


# def model_predict(model, data):
#     return model.predict(data)

def calculate_lime(model, model_predict, categories, explainer, x_axis_values):

    """
    Mass calculate LIME explanations for all instances across all categorical outcomes in the analysis.

    Parameters:
    - explainer: lime explainer
    - categories: list of numpy arrays containing all spectra within each category. 
    A category can correspond to bucketed regression outcomes or to categorical prediction outcomes.
    - x_axis_values: x axis values of the spectra

    Returns:
    - lime_data: list of lime explanations for all instances
    - category_indicator: list indicating which category a given explanation is for
    - spectra_indicator: list indicating where the spectra is in the original dataset
    - mean_spectra_list: list of mean spectra across all categories

    """


    lime_data = []
    category_indicator = []
    spectra_indicator = []
    mean_spectra_list = []
    spectra_length = len(x_axis_values)

    for category_i in range(len(categories)):
        for spectra_i in range(len(categories[category_i])):
            instance_to_explain = categories[category_i][spectra_i] # Adjust index based on which instance you want to explain
            
            exp = explainer.explain_instance(data_row=instance_to_explain, 
                                predict_fn=lambda x: model_predict(x),
                                num_features=spectra_length)

            # Get the list of explanations for all the features
            weights = exp.as_list()
            # print(weights)
            weights = np.array(sort_lime(weights, instance_to_explain, x_axis_values))
            lime_data.append(weights)
            category_indicator.append(category_i)
            spectra_indicator.append(spectra_i)
            mean_spectra_list.append(instance_to_explain)
    return lime_data, category_indicator, spectra_indicator, mean_spectra_list


import re
import numpy as np

def sort_lime(lime_weights, instance_to_explain, x_axis_values):
    """
    Organizes LIME weights for plotting, attempting robust parsing of feature strings.

    Parameters:
    - lime_weights: Output from LIME's exp.as_list(). List of (feature_string, weight).
    - instance_to_explain: The actual data instance (e.g., spectrum) being explained.
                           Used to fill in bounds when LIME gives a one-sided condition.
    - x_axis_values: Pandas Series/DataFrame column or NumPy array of x-axis values (e.g., wavenumbers).

    Returns:
    - A NumPy array where each row is (x_value, lower_bound, upper_bound, weight),
      ordered by the original feature index.
    """
    
    # Ensure x_axis_values is a flat NumPy array for consistent indexing
    if hasattr(x_axis_values, 'values'):
        x_values_flat = x_axis_values.values.flatten()
    else:
        x_values_flat = np.array(x_axis_values).flatten()
        
    num_total_features = len(x_values_flat)
    
    # Temporary list to store parsed data before sorting and full construction
    parsed_explanations = {}

    # Regex to find the feature index number inside asterisks, e.g., *123*
    # This assumes feature names given to LIME were like '*0*', '*1*', etc.
    idx_pattern = re.compile(r"\*(\d+)\*")

    for feature_string, weight in lime_weights:
        idx_match = idx_pattern.search(feature_string)
        
        if not idx_match:
            # This might happen if a feature string from LIME doesn't match the '*index*' pattern.
            # Depending on LIME setup, this could be an unhandled case or an error.
            # print(f"Warning: Could not extract an index from LIME feature string: {feature_string}")
            continue
        
        original_feature_index = int(idx_match.group(1))
        
        # Ensure index is within bounds
        if not (0 <= original_feature_index < num_total_features):
            # print(f"Warning: Parsed index {original_feature_index} is out of bounds for spectrum length {num_total_features}.")
            continue
            
        instance_val_at_index = instance_to_explain[original_feature_index]

        # Default bounds: LIME explanation is about the instance's current value for that feature
        # These will be updated if specific range patterns are matched in the string.
        lower_bound = instance_val_at_index
        upper_bound = instance_val_at_index

        # Attempt to parse bounds from the feature string using a series of common LIME patterns.
        # Values can be integers or floats, e.g., 0.5 or 5. Using (\d+\.?\d*) for flexibility.
        
        # Pattern 1: val1 < *idx* <= val2 (e.g., "0.5 < *1* <= 1.0" or "0.5<*1*<=1.0")
        # Allows for optional spaces around operators
        # Using \*\d+\* instead of just \*idx_match.group(0)\* to ensure the matched index is part of this pattern
        match = re.search(r"(\d+\.?\d*)\s*<\s*\*\d+\*\s*<=\s*(\d+\.?\d*)", feature_string)
        if match:
            lower_bound = float(match.group(1))
            upper_bound = float(match.group(2))
        else:
            # Pattern 2: *idx* <= val2 (e.g., "*1* <= 1.0")
            match = re.search(r"\*\d+\*\s*<=\s*(\d+\.?\d*)", feature_string)
            if match:
                # lower_bound remains instance_val_at_index as per original logic
                upper_bound = float(match.group(1))
            else:
                # Pattern 3: *idx* >= val1 (e.g., "*1* >= 0.5")
                # (LIME often uses >= for lower bounds in simple inequalities)
                match = re.search(r"\*\d+\*\s*>=\s*(\d+\.?\d*)", feature_string)
                if match:
                    lower_bound = float(match.group(1))
                    # upper_bound remains instance_val_at_index
                else:
                    # Pattern 4: *idx* > val1 (e.g., "*1* > 0.5") - less common than >= for LIME
                    match = re.search(r"\*\d+\*\s*>\s*(\d+\.?\d*)", feature_string)
                    if match:
                        lower_bound = float(match.group(1))
                        # upper_bound remains instance_val_at_index
                    else:
                        # Pattern 5: *idx* < val1 (e.g., "*1* < 0.5") - less common than <= for LIME
                        match = re.search(r"\*\d+\*\s*<\s*(\d+\.?\d*)", feature_string)
                        if match:
                            # lower_bound remains instance_val_at_index
                            upper_bound = float(match.group(1))
                        # else:
                            # If no specific range pattern matched, the bounds remain instance_val_at_index.
                            # This means the explanation refers to the feature at its current value.
                            # print(f"Debug: No specific bound pattern matched for: {feature_string}. Using instance value for bounds.")
                            pass
        
        parsed_explanations[original_feature_index] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'weight': weight
        }

    # Construct the final sorted array, ensuring all features are present
    output_list = []
    for i in range(num_total_features):
        x_val = x_values_flat[i]
        if i in parsed_explanations:
            exp_data = parsed_explanations[i]
            output_list.append((x_val, exp_data['lower_bound'], exp_data['upper_bound'], exp_data['weight']))
        else:
            # If a feature was not in LIME's output (e.g., zero weight and omitted, or num_features in explain_instance was too low)
            # Add a default entry: weight 0, bounds are the instance's actual value for that feature.
            instance_val_at_current_index = instance_to_explain[i]
            output_list.append((x_val, instance_val_at_current_index, instance_val_at_current_index, 0.0))
            
    return np.array(output_list)

# Function assumes that spectra are categorized for example according to concentration ranges
# If only one category exists, input the data as a list of one category
# def combine_lime(explainer, categories, range_of_spectra):

#   category_arrays = []
#   num_categories = len(categories)
#   for i in range(len(categories)):
#     # Create lists to hold the data for each column across all arrays
#     column_data = [[] for _ in range(num_categories)]
#     mean_spectra = np.mean(categories[i][0:range_of_spectra], axis=0)
#     spectra_length = len(mean_spectra)
#     for j in range(len(categories[i])):
#       # Select an instance to explain
#       instance_to_explain = categories[i][j]

#       # Generate LIME explanation for this instance
#       exp = explainer.explain_instance(data_row=instance_to_explain, 
#                                       predict_fn=model_predict,
#                                       num_features=spectra_length)
#       # Get the list of explanations for all the features
#       weights = exp.as_list()
#       weights = np.array(sort_lime(weights, mean_spectra))
#       # Append each column of each array to the respective list
#       for col in range(num_categories):
#         column_data[col].append(weights[:,col])
      
#       # Convert lists to arrays and compute the mean across rows (axis=0)
#     mean_columns = [np.mean(np.stack(cols), axis=0) for cols in column_data]
#       # Combine the mean columns back into a single array
#     final_mean_array = np.column_stack(mean_columns)
#     category_arrays.append(final_mean_array)
#   return category_arrays


def plot_lime_global(plot_data, mean_spectra, title, cluster_indices = None, plot_mean_spectra = True, plot_shade = True, plot_clusters = False, figsize=(19, 5)):

    """
    LIME explanation plotting function.
    The function is also used to plot lime explanations for CRIME contexts which represent the mean LIME explanation in set context.
    Toggle options extst for plotting perturbation limits (shade), highlight clusters for CRIME (clusters) or mean spectra for clarity (plot_mean_spectra).

    Parameters:
    - plot_data: input LIME data from earlier function
    - mean_spectra: array of mean spectra for set lime weights 
    - title: figure title
    - cluster_indices: list of highlight cluster indices extracted from corresponding CRIME functions (top_cluster_indices_global)
    - plot_mean_spectra: boolean toggle option for plotting mean spectra for set instance or CRIME context
    - plot_shade: boolean toggle option for plotting LIME perturbation limits. 
      Shades can be sometimes misleading if in a particular crime context the upper limit is near limits but not always leading to a narrower area then in reality.
    - plot_clusters: boolean toggle option for plotting highlighted areas. Areas of interest will remain visible while areas removed in later stages are filled in with black.
    - figsize: figure sizing option

    Returns:
    - fig: a matplotlib figure of the LIME/CRIME explanation

    """


    fig = plt.figure(figsize=figsize)
    
    if plot_mean_spectra:
      # Start plotting the mean spectra, using default black color initially
      last_index = 0
      current_color = 'black'
      plt.plot(plot_data[:, 0], mean_spectra, c = current_color)
      for i in range(1, len(plot_data[:, 0])):
          # Check if current point matches the mean spectra
          if (plot_data[i, 1] == mean_spectra[i]) or (plot_data[i, 2] == mean_spectra[i]):
              color = 'red'
          else:
              color = 'black'
          
          # If color changes, plot the previous segment
          if color != current_color:
              plt.plot(plot_data[last_index:i, 0], mean_spectra[last_index:i], c=current_color, linestyle = 'solid', linewidth = 1)
              last_index = i
              current_color = color
      
      # Plot the last segment
      plt.plot(plot_data[last_index:, 0], mean_spectra[last_index:], c=current_color)

    if(plot_shade):
        # Shade the area between plot_data[:,1] and plot_data[:,2]
        plt.fill_between(plot_data[:, 0], plot_data[:,1], plot_data[:,2], color='teal', alpha=0.5)
        # Separating positive and negative values for plot_data[:,3]
        positives = plot_data[:,3] >= 0
        negatives = plot_data[:,3] < 0

        # Filling positive and negative values with different colors
        plt.fill_between(plot_data[:, 0], 0, plot_data[:,3], where=positives, color='darkgreen', alpha=0.9, label='Positive')
        plt.fill_between(plot_data[:, 0], 0, np.abs(plot_data[:,3]), where=negatives, color='orange', alpha=0.9, label='Negative')
    else:
        # Separating positive and negative values for plot_data[:,3]
        positives = plot_data[:,3] >= 0
        negatives = plot_data[:,3] < 0

        # Filling positive and negative values with different colors
        plt.fill_between(plot_data[:, 0], 0, plot_data[:,3], where=positives, color='darkgreen', alpha=0.9, label='Positive')
        plt.fill_between(plot_data[:, 0], 0, np.abs(plot_data[:,3]), where=negatives, color='orange', alpha=0.9, label='Negative')

    if(plot_clusters):
        # Define colors for top clusters - adjust colors as needed

        # Shade the areas corresponding to cluster indices
        for idx, indices in enumerate(cluster_indices):
            # color = colors[idx % len(colors)]
            for index in indices:
                plt.fill_betweenx([0, mean_spectra[index]], plot_data[index, 0] - 0.5, plot_data[index, 0] + 0.5, color='black', alpha=0.9)


    # plt.ylim(0, 1)
    plt.legend(fontsize = 'large', markerscale = 2)
    plt.title(title, color='black')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Relative Intensity')
    # axs.tick_params(axis='x', direction='in', width=4, labelsize=16, colors='black')
    # axs.tick_params(axis='y', direction='in', width=4, labelsize=16, colors='black')
    plt.show()
    # fig.savefig(f'CRIME{title}.pdf', dpi = 600)
    return(fig)

