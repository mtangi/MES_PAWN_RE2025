# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:33:52 2025

@author: TANGI

The script contains the code for the creation of Figure 3 and 4, as well as Figure S.2 in the supplementary material

"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import pearsonr, pointbiserialr,spearmanr

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

#energy indipendence: % of energy produced locally 
output_sensitivity['energy_indipendence'] = savedata[['pv_existing_ground.p', 'pv_existing_roof.p','pv_planned_roof.p']].sum(axis=1)/savedata[[ 'pv_existing_ground.p', 'pv_existing_roof.p','pv_planned_roof.p','thermo.p']].sum(axis=1)*100
output_sensitivity['water_indipendence'] = savedata[['desalter_existing.p','desalter_planned.p']].sum(axis=1)/savedata[[ 'desalter_existing.p','desalter_planned.p','supply_ship.p']].sum(axis=1)*100
# output_sensitivity['water_indipendence'] = savedata['supply_ship.p']

baseline_monetary = 14075643.88/10**6
baseline_emissions =  36999130.0/10**6

output_sensitivity['monetary_ok'] = (output_sensitivity['monetary'] < baseline_monetary).astype(int)
output_sensitivity['emissions_ok'] = (output_sensitivity['emissions'] < baseline_emissions).astype(int)

#labels uncertanty
X_Labels = [
    "Pop$_{\%}$$^{NR}$", "Pop$_{WD}$$^{NR}$", "Pop$_{WD}$$^{R}$", 
    "Ship$_{EF}$", "Ship$_{cost}$", 
    "DSP$_{eff}$$^{new}$", "DSP$_{eff}$$^{old}$", "DSP$_{cost}$$^{new}$", 
    "Fuel$_{cost}$", "Fuel$_{EF}$", 
    "PV$_{cap}$", "PV$_{p/area}$", "PV$_{cost}$", "PV$_{prod}$", 
    "AQD$_{loss}$", 
    "WS$_{cap}$$^{new}$", "WS$_{cost}$$^{new}$", "WS$_{loss}$$^{ex}$", 
    "ES$_{exp}$", "ES$_{cost}$", 
    "W$_{CO2}$"
]


#%% create plot folder
figure_folder = os.path.join(result_folder,'figures')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)   
    
#%% define number of solution in each region
# Extract relevant columns
monetary = output_sensitivity['monetary']
emissions = output_sensitivity['emissions']

# Identify the regions
bottom_left = (monetary < baseline_monetary) & (emissions < baseline_emissions)
bottom = (emissions < baseline_emissions)
bottom_right = (monetary >= baseline_monetary) & (emissions < baseline_emissions)
left = (monetary < baseline_monetary)
top = (emissions >= baseline_emissions)
top_left = (monetary < baseline_monetary) & (emissions >= baseline_emissions)
top_right = (monetary >= baseline_monetary) & (emissions >= baseline_emissions)
rest = (monetary >= baseline_monetary) & (emissions >= baseline_emissions)

# Count points in each region
counts = {
    "Bottom-Left (Below Both Baselines)": bottom_left.sum(),
    "Bottom (Below Emissions Only)": bottom.sum(),
    "Left (Below Monetary Only)": left.sum(),
    "Top-Right (Above Both Baselines)": rest.sum()
}

# Count points in each region
counts_2 = {
    "Bottom-Left": bottom_left.sum(),
    "Bottom-Right": bottom_right.sum(),
    "Top-Left": top_left.sum(),
    "Top-Right": top_right.sum()
}

# Extract representative points (closest to the mean in each region)
def get_representative_point(region_mask):
    """Finds the point closest to the mean of the region."""
    subset = output_sensitivity[region_mask]
    if len(subset) == 0:
        return None  # If no points in this region, return None
    mean_point = np.array([subset["monetary"].mean(), subset["emissions"].mean()])
    distances = np.linalg.norm(subset[['monetary', 'emissions']].values - mean_point, axis=1)
    return subset.iloc[np.argmin(distances)]  # Return the row closest to the mean

representative_points = {
    "Top-Right": get_representative_point(rest),
    "Bottom-Left": get_representative_point(bottom_left),
    "Top-Left": get_representative_point(left),
    "Bottom-Right": get_representative_point(bottom)
}

# Print results
print("Point Counts per Region:")
for region, count in counts.items():
    prc = round(count/len(monetary),3)*100
    print(f"{region}: {count} points [{prc} %]")

# Print results
print("Point Counts per Region:")
for region, count in counts_2.items():
    prc = round(count/len(monetary),3)*100
    print(f"{region}: {count} points [{prc} %]")

print("\nRepresentative Points:")
for region, point in representative_points.items():
    if point is not None:
        print(f"{region}: Monetary = {point['monetary']}, Emissions = {point['emissions']}")
    else:
        print(f"{region}: No points in this region.")

#%% load baseline sensitivity results

result_folder = 'sets_PAWN'
   
#extract result dataframe for id_sim
baseline_data = pd.read_csv(os.path.join(result_folder, 'savedata.csv')) 

baseline_range =pd.DataFrame(columns =['monetary','emissions'], index = ['min','q25','avg','q75','max'] )

baseline_range.loc['min'] = baseline_data.min()
baseline_range.loc['q25'] = baseline_data.quantile(0.25)
baseline_range.loc['avg'] = baseline_data.mean()
baseline_range.loc['q75'] = baseline_data.quantile(0.75)
baseline_range.loc['max'] = baseline_data.max()

baseline_range = baseline_range/10**6 
#%% plot scatter of results compared to baseline (Fig 3a-3b)

save_fig = 1
 
#point size
s=15

# Define baselines
baseline_monetary = baseline_range.loc['avg']['monetary']
baseline_emissions = baseline_range.loc['avg']['emissions']
# baseline_monetary = 14075643.88/10**6
# baseline_emissions =  36999130.0/10**6

# Extract relevant columns
monetary = output_sensitivity['monetary']
emissions = output_sensitivity['emissions']

# Create a single figure with a 2x2 subplot layout
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes = axes.flatten()  # Flatten to easily iterate


color_base = np.array([
    [0.9, 0.9, 0.9],
    [0.165, 0.812, 0.11],
    [0.78, 0.831, 0.067],
    [0.031, 0.82, 0.796]
], dtype=float)

# Classify points into four regions
regions = {
    "above_both": [],
    "below_costs": [],
    "below_emissions": [],
    "below_both": []
}

for i, (m, e) in enumerate(zip(monetary, emissions)):
    if m < baseline_monetary and e < baseline_emissions:
        regions["below_both"].append(i)  
    elif m < baseline_monetary:
        regions["below_costs"].append(i)
    elif e < baseline_emissions:
        regions["below_emissions"].append(i)
    else:
        regions["above_both"].append(i)

# Compute the average point for each region and find the closest actual point
representative_points = {}
for key, indices in regions.items():
    if indices:
        avg_m = np.mean(monetary.iloc[indices])
        avg_e = np.mean(emissions.iloc[indices])
        closest_idx = min(indices, key=lambda i: (monetary.iloc[i] - avg_m) ** 2 + (emissions.iloc[i] - avg_e) ** 2)
        representative_points[key] = closest_idx
    else:
        representative_points[key] = None
 
    
# Compute the average point for each region
region_averages = {}
for key, indices in regions.items():
    if indices:
        avg_m = np.mean(monetary.iloc[indices])
        avg_e = np.mean(emissions.iloc[indices])
        region_averages[key] = (avg_m, avg_e)
    else:
        region_averages[key] = None

# Find the representative points that best form a rectangle
representative_points = {}
for key, indices in regions.items():
    if indices:
        avg_m, avg_e = region_averages[key]
        closest_idx = min(indices, key=lambda i: abs(monetary.iloc[i] - avg_m) + abs(emissions.iloc[i] - avg_e))
        representative_points[key] = closest_idx
    else:
        representative_points[key] = None
  
representative_points = {'above_both': 16049,
  'below_costs': 17163,
  'below_emissions': 21203,
  'below_both': 3736}

        
# Scatter plot 1  
# Define colors based on conditions using RGB values
colors = [
    color_base[1,:] if m < baseline_monetary and e < baseline_emissions else  
    color_base[2,:]  if m < baseline_monetary else  
    color_base[3,:]  if e < baseline_emissions else  
    color_base[0,:] 
    for m, e in zip(monetary, emissions)
]

axes[0].scatter(monetary, emissions, c=colors, edgecolors="None", s=s)
axes[0].axvline(x=baseline_monetary, color='red')
axes[0].axhline(y=baseline_emissions, color='red')
axes[0].set_xlabel("Costs [M€/year]")
axes[0].set_ylabel("Emissions [ktonCO$_2$/year]")
# axes[0].set_title("Scatter Plot of Monetary vs Emissions")
# Add legend


for key, idx in representative_points.items():
    if idx is not None:
        axes[0].scatter(monetary.iloc[idx], emissions.iloc[idx], s=50, edgecolor='red', facecolor='none', linewidth=2)

legend_patches = [
    mpatches.Patch(color=color_base[0,:] , label="Above Both Baselines"),
    mpatches.Patch(color=color_base[2,:], label="Below Costs Only"),
    mpatches.Patch(color=color_base[3,:], label="Below Emissions Only"),
    mpatches.Patch(color=color_base[1,:], label="Below Both Baselines")
]
axes[0].legend(handles=legend_patches)

# Add text annotations
axes[0].text(
    x=monetary.max() -6.1,  # Align with the leftmost part of the plot
    y=baseline_emissions * 1.01,  # Slightly above the horizontal line
    s="Baseline - CO$_2$ emissions",
    color="red",
    fontsize=10,
    verticalalignment="bottom"
)

axes[0].text(
    x=baseline_monetary * 1.010,  # Slightly to the right of the vertical line
    y=emissions.max()-6.5,  # Align with the upper part of the 
    s="Baseline - Costs",
    color="red",
    fontsize=10,
    rotation=270,  # Vertical orientation
    horizontalalignment="left"
)

# Scatter plot 2

# colors = savedata['supply_ship.p']
water_demand = (savedata['desalter_planned.p']+savedata['desalter_existing.p']+savedata['supply_ship.p'])/10**6
colors =  savedata['supply_ship.p']/10**6/water_demand
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="3%", pad=0.1)  # Create separate axis for colorbar

s2 = axes[1].scatter(monetary, emissions, c=colors, edgecolors="None", cmap = 'coolwarm', s=s)
axes[1].axvline(x=baseline_monetary, color='red')
axes[1].axhline(y=baseline_emissions, color='red')
axes[1].set_xlabel("Costs [M€/year]")
axes[1].set_ylabel("Emissions [ktonCO$_2$/year]")
# Add colorbar
cbar = plt.colorbar(s2,  aspect=50, cax = cax)
cbar.set_label('Water supply via ships [% of demand satisfied]', rotation=270, labelpad=15, fontsize = 9, fontstyle="italic")
plt.tight_layout()

# Scatter plot 3

# colors = input_uncertainty['weight_co2']

# divider = make_axes_locatable(axes[2])
# cax = divider.append_axes("right", size="5%", pad=0.1)  # Create separate axis for colorbar

# s3 = axes[2].scatter(monetary, emissions, c=colors, edgecolors="None", cmap = 'PiYG', s=s)
# axes[2].axvline(x=baseline_monetary, color='red')
# axes[2].axhline(y=baseline_emissions, color='red')
# axes[2].set_xlabel("Costs [M€]")
# axes[2].set_ylabel("Emissions [ktonCO$_2$]")
# # axes[2].set_title("CO$_2$ relative weight")

# # Add colorbar
# cbar = plt.colorbar(s3,  aspect=50, cax = cax)
# cbar.set_label('CO$_2$ relative weight', rotation=270, labelpad=15, fontsize = 9, fontstyle="italic")
# plt.tight_layout()
# fig.subplots_adjust(wspace=0.3)

# Add transparent banners for IQR (Q25–Q75)
for ax in axes[:2]:  # Apply to both scatter plots
    ax.axvspan(
        baseline_range.loc['q25', 'monetary'],
        baseline_range.loc['q75', 'monetary'],
        color='red',
        alpha=0.15,
        label='Cost IQR' if ax == axes[0] else None
    )
    ax.axhspan(
        baseline_range.loc['q25', 'emissions'],
        baseline_range.loc['q75', 'emissions'],
        color='red',
        alpha=0.15,
        label='Emission IQR' if ax == axes[0] else None
    )
    
if save_fig == 1:
    plt.savefig( os.path.join(figure_folder,  'Scatterplot_solution'), dpi=250 ,  bbox_inches='tight')

# plt.show()

#%% plot pies of selected points  (Fig 3c)

import matplotlib.patches as mpatches

# plotsim_id = [1050,1092,566,788,10,562]
plotsim_id = [value +1 for value in representative_points.values()]
fontlabel = 24

fonttext = 30

#define colors
colors = {
    "desalter_existing": "#4a7ae2",
    "desalter_planned": "#4a1ad5",
    "pv_existing_ground": "#b2e24a",
    "pv_existing_roof": "#dde24a",
    "pv_planned_roof": "#fff700",
    "supply_ship": "#a7277e",
    "thermo": "#5e5e5e",
    "battery": "#f000ff",
    "water_storage_existing": "#7ca86f",
    "water_storage_planned": "#a8e497"
}
techlabels =  ['Desalination Plant - Existing', 'Desalination Plant - New', 'Ground PV', 'Roof PV - Existing', 'Roof PV - New', 'Water supply from ships','Diesel generators','Electricity storage','Water storage - Exisiting','Water storage - New']


colors = pd.Series(colors, name="color")
colors = colors.to_frame()
colors["techlabels"] = techlabels

technames =  modified_names = [name + ".p" for name in name_tech[:-3]]
car = ['Electricity','Water']

min_bound = 1

energy_vector = [1,1,0,0,0,1,0]
# Define min and max pie sizes
min_pie_size = 0.7  # Minimum pie size as a fraction of max size
max_pie_size = 1.0  # Maximum pie size

# Compute total values for each pie (electricity & water separately)
total_values_electricity = []
total_values_water = []

for i in range(len(plotsim_id)):
    total_values_electricity.append(savedata.loc[plotsim_id[i]][[name for name, value in zip(technames, energy_vector) if value == 0]].sum())
    total_values_water.append(savedata.loc[plotsim_id[i]][[name for name, value in zip(technames, energy_vector) if value == 1]].sum())

# Normalize sizes for each energy vector (relative to max pie in that group)
min_electricity = min(total_values_electricity) if total_values_electricity else 1
min_water = min(total_values_water) if total_values_water else 1
max_electricity = max(total_values_electricity) if total_values_electricity else 1
max_water = max(total_values_water) if total_values_water else 1

scaled_sizes_electricity = [min_pie_size + (max_pie_size - min_pie_size) * (val - min_electricity)/(max_electricity - min_electricity) for val in total_values_electricity]
scaled_sizes_water = [min_pie_size + (max_pie_size - min_pie_size) * (val - min_water)/(max_water - min_water) for val in total_values_water]
autopct_format = lambda p: f'{p:.0f}%' if p > 0 else ''

# Find maximum pie sizes to align text boxes
max_radius_electricity = max(scaled_sizes_electricity)
max_radius_water = max(scaled_sizes_water)

#create a pie for each active carrier
fig, ax = plt.subplots(2, len(plotsim_id), constrained_layout=True, figsize=(37,15))

# Now plot the pies with scaled sizes
for i in range(len(plotsim_id)):

    # Electricity pie (energy_vector = 0)
    energy_tech = [name for name, value in zip(technames, energy_vector) if value == 0]
    frac = savedata.loc[plotsim_id[i]][energy_tech]
    frac.index = [index[:-2] if index.endswith('.p') else index for index in frac.index]
    frac = frac[frac/sum(frac)>0.005]  # Remove negligible contributions
    colors_plot = colors.loc[frac.index]['color']

    wedges, texts, autotexts = ax[0,i].pie(
    frac, colors=colors_plot, autopct=autopct_format, startangle=90, 
    textprops={'fontsize': fontlabel}, radius=scaled_sizes_electricity[i],
    pctdistance=1.2,  # Move percentage labels outward
    labeldistance=0  # Move labels outward
    )

    # ax[i, 0].pie(frac, colors=colors_plot, autopct=autopct_format, 
    #               startangle=90, textprops={'fontsize': fontlabel}, radius=scaled_sizes_electricity[i])
    
    # Add total value text below the pie
    # Align Total Value Text for Electricity
    ax[0,i].text(0, -1.5* max_radius_electricity,  f'{total_values_electricity[i]/ 10**6 :.2f}  GWh/year ', 
                  ha='center', va='center', fontsize=fonttext, bbox=dict(facecolor='white', edgecolor='white'))
    # Water pie (energy_vector = 1)
    water_tech = [name for name, value in zip(technames, energy_vector) if value == 1]
    frac = savedata.loc[plotsim_id[i]][water_tech]
    frac.index = [index[:-2] if index.endswith('.p') else index for index in frac.index]
    frac = frac[frac/sum(frac)>0.01]
    colors_plot = colors.loc[frac.index]['color']

    wedges, texts, autotexts = ax[1,i].pie(
        frac, colors=colors_plot, autopct=autopct_format, startangle=90, 
        textprops={'fontsize': fontlabel},
        radius=scaled_sizes_water[i],
        pctdistance=1.2,  
        labeldistance=1.2  
    )
    # Add total value text below the pie
    
    # Align Total Value Text for Water
    ax[1,i].text(0, -1.5 * max_radius_water,  f'{total_values_water[i]/ 10**6:.3f} Mm$^3$/year ', 
                 ha='center', va='center', fontsize=fonttext, bbox=dict(facecolor='white', edgecolor='white'))

import matplotlib.patches as mpatches

colorlegend = pd.Series(colors["color"].values, index=colors["techlabels"], name="color")

# Split colors into two groups based on energy_vector
group_1 = {tech: colorlegend[tech] for tech, value in zip(techlabels, energy_vector) if value == 0}
group_2 = {tech: colorlegend[tech] for tech, value in zip(techlabels, energy_vector) if value == 1}

# Create legend handles for each group
legend_patches_1 = [mpatches.Patch(color=color, label=tech) for tech, color in group_1.items()]
legend_patches_2 = [mpatches.Patch(color=color, label=tech) for tech, color in group_2.items()]

# # Create a figure legend with two columns
# legend = fig.legend(
#     handles=legend_patches_1 + legend_patches_2,
#     loc="lower center",
#     bbox_to_anchor=(0.5, 0.1), 
#     ncol=2, 
#     fontsize=fonttext,
#     # title="Technology Categories",
#     columnspacing=2.0,
#     frameon=False
# )

# # Set legend labels in two columns
# for i, text in enumerate(legend.get_texts()):
#     if i < len(legend_patches_1):
#         text.set_ha("left")  # Left-align group 1
#     else:
#         text.set_ha("right")  # Right-align group 2

# Adjust layout to prevent overlapping
plt.subplots_adjust(bottom=0.2)

# Save the figure
if save_fig == 1:
    plt.savefig(os.path.join(figure_folder, 'pies.png'), dpi=250, bbox_inches='tight')
plt.show()

#%% extract uncertainty parameters for the selecterd points

input_uncertainty_data = input_uncertainty.loc[plotsim_id]
savedata_data = savedata.loc[plotsim_id]

#%% plot scatters with uncertanty and objectives (Fig S.2b)

save_fig = 1

# Define baselines
baseline = [14075643.88 / 10**6 ,36999130.0 / 10**6]  # 14.07564388

# Extract relevant columns
objs = output_sensitivity[['monetary','emissions']]
objs_labels = ['Cost [M€]', 'Emissions [ktonCO$_2$]']

#define plot ids
plot_id = [0,3,8,9,10,20]
# plot_id = range(len(X_Labels))
           
# Create a DataFrame to store correlation coefficients
index = [X_Labels[i] for i in plot_id if i < len(X_Labels)]
correlation_df = pd.DataFrame(index=index, columns=objs_labels)

# Create a single figure with a 2x2 subplot layout
fig, axes = plt.subplots(len(plot_id), 2, figsize=(7, 3*len(plot_id)))

for i in range(len(plot_id)):
    idp = plot_id[i]
    x_axis = np.array(input_uncertainty.iloc[:, idp], dtype=float)

    for j in range(len(objs_labels)):
        y_axis = np.array(objs.iloc[:, j], dtype=float)
        
        # Compute Pearson correlation coefficient
        r, _ = pointbiserialr(y_axis, x_axis)
        correlation_df.iloc[i, j] = r
        
        # Assign colors based on baseline
        clr = (0, 0.6, 0) if j==1 else (0.6, 0, 0)
        
        colors = [
            clr if m < baseline[j] else (0.9, 0.9, 0.9)  
            for m in y_axis
        ]
        
        edge_colors = [(0.8, 0.8, 0.8) if col == (0.9, 0.9, 0.9) else [0,0,0] for col in colors]

        # Scatter plot
        axes[i, j].scatter(x_axis, y_axis, c=colors, edgecolors=edge_colors, cmap='coolwarm', s=20)
        axes[i, j].axhline(y=baseline[j], color='red')
        axes[i, j].set_ylabel(objs_labels[j], fontsize=12)
        # axes[i, j].set_xlabel(X_Labels[idp], fontsize=14)
        
        # Add correlation coefficient to the plot
        axes[i, j].text(
            0.05, 0.95, f"r = {r:.3f}",
            transform=axes[i, j].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )
        
        # if i < 1:  
        #     axes[i, j].set_title(objs_labels[j])

plt.tight_layout()

# Save the figure
if save_fig == 1:
    plt.savefig(os.path.join(figure_folder, 'Scatterplot_variables.png'), dpi=250, bbox_inches='tight')
plt.show()

#%% plot correlation coefficient (Fig S.2a)

# Corresponding group numbers
group_id = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7]

# Extract relevant columns
objs = output_sensitivity[['monetary','emissions']]
objs_labels = ['Cost [M€]', 'Emissions [ktonCO$_2$]']

# Create a DataFrame to store correlation coefficients
correlation_df = pd.DataFrame(index=X_Labels, columns=objs_labels)

for i in range(len(X_Labels)):
    y_axis = np.array(input_uncertainty.iloc[:, i], dtype=float)

    for j in range(len(objs_labels)):
        x_axis = np.array(objs.iloc[:, j], dtype=float)
        
        # Compute Pearson correlation coefficient
        r, _ = pearsonr(x_axis, y_axis)
        correlation_df.iloc[i, j] = r          
        
        # # Compute Spearman correlation (rank-based)
        # r_s, _ = spearmanr(x_axis, y_axis)
        # correlation_df.iloc[i, j] = r_s
        
        # # Compute Biserial correlation
        # biserial_r, _ = pointbiserialr(y_axis, x_axis)
        # correlation_df.iloc[i, j] = biserial_r

# Plot the correlation DataFrame as a heatmap with red-to-blue scale
plt.figure(figsize=(10,8))
sns.heatmap(
    correlation_df.astype(float), 
    annot=True, 
    cmap="coolwarm",  # Red-blue color scale
    center=0,  # Ensures white represents zero correlation
    cbar=True, 
    fmt=".2f", 
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)

# Add horizontal lines to separate groups
group_boundaries = [i for i in range(1, len(group_id)) if group_id[i] != group_id[i-1]]
for boundary in group_boundaries:
    plt.hlines(y=boundary, xmin=0.007, xmax=len(objs_labels), colors=[0.1, 0.1 ,0.1], linewidth=1.5)


# Add a title
# plt.title("Correlation Between Uncertainty Variables and Objective Performance")

# Save the heatmap as an image
heatmap_path = os.path.join(figure_folder, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=250, bbox_inches='tight')

# Display the heatmap
plt.show()
