# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:33:52 2025

@author: TANGI

The script contains the code for the creation of Figure 5, as well as Figure S.3 and S.4 in the supplementary material.
"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import SAFE modules:
from safepython.util import aggregate_boot # function to aggregate the bootstrap results

from PAWN_modified import pawn_indices_mod, pawn_plot_cdf_mod

#%% load data

result_folder = 'sets_PAWN'
   
#extract result dataframe for id_sim
savedata = pd.read_csv(os.path.join(result_folder, 'savedata.csv')) 

# Load the second CSV file
input_uncertainty = pd.read_csv(os.path.join(result_folder,'inputspace.csv') )

savedata.set_index('id', inplace=True)
input_uncertainty.set_index('id', inplace=True)
    
#%% extract result dataframe

output_sensitivity = pd.DataFrame(index=savedata.index)
output_sensitivity['monetary'] = savedata['monetary']/10**6
output_sensitivity['emissions'] = savedata['emissions']/10**6

name_tech = [ 'desalter_existing','desalter_planned', 'pv_existing_ground', 'pv_existing_roof',
       'pv_planned_roof', 'supply_ship', 'thermo','battery', 'water_storage_existing', 'water_storage_planned']

techlabels =  ['Desalination Plant - Existing', 'Desalination Plant - New', 'Ground PV', 'Roof PV - Existing', 'Roof PV - New', 'Water supply from ships','Diesel generators','Battery','Water storage - Exisiting','Water storage - New']

#labels uncertanty

X_Labels = [
    "Pop$_{\%}$$^{NR}$", "Pop$_{WD}$$^{R}$", "Pop$_{WD}$$^{NR}$", 
    "Ship$_{EF}$", "Ship$_{cost}$", 
    "DSP$_{eff}$$^{new}$", "DSP$_{eff}$$^{old}$", "DSP$_{cost}$$^{new}$", 
    "Fuel$_{cost}$", "Fuel$_{EF}$", 
    "PV$_{area}$", "PV$_{area/p}$", "PV$_{cost}$", "PV$_{prod}$", 
    "AQD$_{loss}$", 
    "WS$_{cap}$$^{new}$", "WS$_{cost}$$^{new}$", "WS$_{loss}$$^{ex}$", 
    "ES$_{exp}$", "ES$_{cost}$", 
    "W$_{CO2}$"
]

objs_labels = ['Cost [M€]', 'Emissions [ktonCO$_2$]']
objs_labels_nounits = ['Cost', 'Emissions']


group_ids =[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7]
    
#%% create plot folder
figure_folder = os.path.join(result_folder,'figures')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)   

#%% plot conditional and unconditional CDFs (Fig S.3 - S.4)

output_labels = ['Cost [M€]', 'Emissions  [ktonCO$_2$]']
    
save_fig = 1
n = 10 # number of conditioning intervals

X_unc = np.array(input_uncertainty, 'float')
    

for i in range(len(output_labels)):
    column_data = output_sensitivity.iloc[:, i]

    Y = np.array(column_data, 'float')
    
    # Compute and plot conditional and unconditional CDFs:
    YF, FU, FC, xc = pawn_plot_cdf_mod(X_unc, Y, n, cbar=True, n_col=3, labelinput=X_Labels, Y_Label = output_labels[i])
    # Adjust the plot size after plotting
    plt.gcf().set_size_inches(12, 25)  # Width=12 inches, Height=8 inches

    # Adjust subplot spacing
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        hspace=0.5, wspace=0.9)
    if save_fig == 1:
        plt.savefig( os.path.join(figure_folder,  'CDFs_' + str(output_sensitivity.columns[i])+'.png'), dpi=250 ,  bbox_inches='tight')
    
    plt.show()

#%% plot conditional and unconditional CDFs only for some variables (Fig 5)

output_labels = ['Cost [M€]', 'Emissions [ktonCO$_2$]']
    
save_fig = 1
n = 10 # number of conditioning intervals

X_unc = np.array(input_uncertainty, 'float')
    
variables_plot = [[8,0,3],[0,9,10]]

for i in range(len(output_labels)):
    column_data = output_sensitivity.iloc[:, i]

    Y = np.array(column_data, 'float')
    
    X_unc_plot = X_unc[:, variables_plot[i]]
    X_Labels_plot = [X_Labels[i] for i in variables_plot[i]]

    # Compute and plot conditional and unconditional CDFs:
    YF, FU, FC, xc = pawn_plot_cdf_mod(X_unc_plot, Y, n, cbar=True, n_col=1, labelinput = X_Labels_plot, Y_Label = output_labels[i])
    # Adjust the plot size after plotting
    plt.gcf().set_size_inches(6,15)  # Width=12 inches, Height=8 inches

    # Adjust subplot spacing
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        hspace=0.5, wspace=0.9)
    
    if save_fig == 1:
        plt.savefig( os.path.join(figure_folder,  'CDFs_detail_vr_' + str(output_sensitivity.columns[i])+'.png'), dpi=250 ,  bbox_inches='tight')
    
    plt.show()

#%% extract PAWN indexes

n = 12  # Number of conditioning intervals
X_unc = np.array(input_uncertainty, dtype=float)
Nboot = 1000

# Get matrix dimensions (num outputs × num inputs)
num_outputs = len(output_sensitivity.columns)
num_inputs = len(input_uncertainty.columns)

# Initialize matrices to store results
KS_max_m_matrix = np.zeros((num_outputs, num_inputs))
KS_max_lb_matrix = np.zeros((num_outputs, num_inputs))
KS_max_ub_matrix = np.zeros((num_outputs, num_inputs))

# Initialize matrices to store results
KS_mean_m_matrix = np.zeros((num_outputs, num_inputs))
KS_mean_lb_matrix = np.zeros((num_outputs, num_inputs))
KS_mean_ub_matrix = np.zeros((num_outputs, num_inputs))

# Compute and store values in matrices
for i in range(num_outputs):
    column_data = output_sensitivity.iloc[:, i]
    Y = np.array(column_data, dtype=float)

    # Compute PAWN indices
    # KS_median, KS_mean, KS_max = PAWN.pawn_indices(X_unc, Y, n, Nboot=Nboot)
    KS_median, KS_mean, KS_max = pawn_indices_mod(X_unc, Y, n, Nboot=Nboot)
    KS_max_m, KS_max_lb, KS_max_ub = aggregate_boot(KS_max)  # shape (num_inputs,)
    KS_mean_m, KS_mean_lb, KS_mean_ub = aggregate_boot(KS_mean)  # shape (num_inputs,)

    # Store values in matrices
    KS_max_m_matrix[i, :] = KS_max_m
    KS_max_lb_matrix[i, :] = KS_max_lb
    KS_max_ub_matrix[i, :] = KS_max_ub
    
    # Store values in matrices
    KS_mean_m_matrix[i, :] = KS_mean_m
    KS_mean_lb_matrix[i, :] = KS_mean_lb
    KS_mean_ub_matrix[i, :] = KS_mean_ub

#%% plot boxplot with confidence intervals (Fig 5)

save_fig = 1
    
def plotPAWNboxplot(K_m, K_lb, K_ub,  save_fig, name = 'PAWNindexes_boxplot'):

    # Unique groups and color mapping
    unique_groups = sorted(set(group_ids))
    group_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))  # Light colors
    
    # Adjust x positions to insert gaps between groups
    adjusted_x_positions = []
    current_x = 1  # Start at 1 for Matplotlib's 1-based x-ticks
    
    for i in range(len(group_ids)):
        if i > 0 and group_ids[i] != group_ids[i - 1]:  # Add space when a new group starts
            current_x += 0.5  # Increase spacing between groups
        adjusted_x_positions.append(current_x)
        current_x += 1  # Normal spacing between boxplots in the same group
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  
    axes = axes.flatten()  # Flatten to easily iterate over
    
    for fig_idx in range(4):  # Iterate through the 4 sets
        ax = axes[fig_idx]  # Select corresponding subplot
        
        # Prepare boxplot data
        box_data = []
        colors = []  # Store colors for each boxplot
    
        for i in range(21):
            box_data.append({
                'whislo': K_lb[fig_idx, i],  # Lower whisker
                'q1': K_lb[fig_idx, i],      # Q1 (Lower boundary of the box)
                'med': K_m[fig_idx, i],      # Median (assuming mean as median)
                'q3': K_ub[fig_idx, i],      # Q3 (Upper boundary of the box)
                'whishi': K_ub[fig_idx, i],  # Upper whisker
                'fliers': []                 # No outliers
            })
            colors.append(group_colors[group_ids[i]])  # Assign color based on group
        
        # Plot the boxplots with custom positions and colors
        bxp_elements = ax.bxp(box_data, positions=adjusted_x_positions, showmeans=False,patch_artist=True)  # Create boxplot
    
        # Recolor each box manually by accessing the 'boxes' key in the bxp return object
        # for box, color in zip(bxp_elements['boxes'], colors):
        #     box.set_facecolor(color)  # Set box color
    
        for patch, clr in zip(bxp_elements["boxes"], colors):
            patch.set_facecolor(clr)
            
        # Add a black dotted horizontal line at y = 0
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    
        # Add grey vertical lines between groups
        for i in range(1, len(group_ids)):
            if group_ids[i] != group_ids[i - 1]:  # Draw line when a new group starts
                ax.axvline(x=adjusted_x_positions[i] - 0.75, color='grey', linestyle='-', linewidth=1)
        
        # Change median line color to black
        for median in bxp_elements['medians']:
            median.set_color('black')  # Set median line color
            # median.set_linewidth(1.5)  # Make the median line slightly thicker for visibility


        # Customize subplot
        # ax.set_title(f"Boxplot Set {fig_idx + 1}")
        # ax.set_xlabel("Boxplot Index")
        ax.set_ylabel('KS$_{MAX}$')
        ax.set_title( f'{objs_labels_nounits[fig_idx]}')
        
        # Set x-ticks and labels correctly
        ax.set_xticks(adjusted_x_positions)  # Adjusted positions with spacing
        ax.set_xticklabels(X_Labels, rotation=90)  # Rotate for readability
        ax.tick_params(axis='x', labelsize=7.7)  # Reduce font size for better fit
        ax.set_ylim(-0.05, 1)  # Set y-axis range
    
    # Increase vertical space between the subplots
    plt.subplots_adjust(hspace=0.4, wspace = 0.3) 
    # Adjust layout to prevent overlap
    # plt.tight_layout()
    
    # Save full figure if required
    if save_fig == 1:
        plt.savefig(os.path.join(figure_folder, name), dpi=250, bbox_inches='tight')
    
    plt.show()

#plot boxplot for the index MEAN
plotPAWNboxplot(KS_mean_m_matrix, KS_mean_lb_matrix, KS_mean_ub_matrix,  save_fig, name = 'PAWNindexesMEAN_boxplot')

#plot boxplot for the index MAX
plotPAWNboxplot(KS_max_m_matrix, KS_max_lb_matrix, KS_max_ub_matrix,  save_fig, name = 'PAWNindexesMAX_boxplot')
