# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:37:47 2025

@author: TANGI

The script contains the code for the creation of Figure 4 and Figure 6, as well as Figure S.5 to S.10 in the supplementary material
"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import SAFE modules:
from safepython.util import aggregate_boot # function to aggregate the bootstrap results

from PAWN_modified import pawn_indices_mod, pawn_plot_cdf_mod
import matplotlib as mpl

#%% load data

result_folder = 'sets_PAWN'
   
#extract result dataframe for id_sim
savedata = pd.read_csv(os.path.join(result_folder, 'savedata.csv')) 

# Load the second CSV file
input_uncertainty = pd.read_csv(os.path.join(result_folder,'inputspace.csv') )

savedata.set_index('id', inplace=True)
input_uncertainty.set_index('id', inplace=True)
    

#create result folder
figure_folder = os.path.join(result_folder,'figures')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)   
    
    
#%% extract result dataframe

output_sensitivity_tech = pd.DataFrame(index=savedata.index)
output_sensitivity_tech['monetary'] = savedata['monetary']/10**6
output_sensitivity_tech['emissions'] = savedata['emissions']/10**6

#calculate prc of potential solar producibility being actually used
PV_data = pd.read_csv('pv_island.csv')
PV_data = PV_data.set_index('Index')['Lampedusa']  
    
prod_year = pd.Series(index = range(2010,2022))
for y in range(2010,2022):
    prod_year[y] = pd.read_csv(f"Lampedusa/timeseries_data/pv_resource_{y}.csv").iloc[:, 1].sum()
    
year_list = pd.read_csv('Lampedusa/timeseries_data/producibility_year_list.csv').iloc[:, -2:].to_numpy() 

#Define tecnology adoption metrics

water_demand = (savedata['desalter_planned.p']+savedata['desalter_existing.p']+savedata['supply_ship.p'])/10**6

output_sensitivity_tech['new_des_prod'] =  savedata['desalter_planned.p']/10**6/water_demand

output_sensitivity_tech['old_des_prod'] =  savedata['desalter_existing.p']/10**6/water_demand

output_sensitivity_tech['ship_prod'] =  savedata['supply_ship.p']/10**6/water_demand

output_sensitivity_tech['WS_new_cap'] =  savedata['water_storage_planned']/ input_uncertainty['waterstorage_new_cap_max']

#Define tecnology adoption metrics for pv and battery use
pv_prd_prc = []
battery_cap_max = []

for i in range (len(savedata['pv_planned_roof.p'])):
    res_area_max = float(max(0,float(PV_data['building_extension_island']) * input_uncertainty['prc_area_available'].iloc[i] * 10**6 - float(PV_data['PV_power_installed_isola_roof'])*input_uncertainty['pv_resource_area_per_energy_cap'].iloc[i] ) ) 
    cap_max = res_area_max / input_uncertainty['pv_resource_area_per_energy_cap'].iloc[i]
    
    #find the name of the year in the list with avg producibility closest to the selected value
    year = int(year_list[np.argmin(np.abs(year_list[:,1] - input_uncertainty['pv_year'].iloc[i] )),0])
           
    pv_prod = savedata['pv_planned_roof.p'].iloc[i]
    
    pv_max_prod = prod_year[year]*cap_max
    
    # Handle division by zero or NaN
    if np.isnan(pv_max_prod) or pv_max_prod == 0:
        pv_prd_prc.append(1)  # Append NaN if invalid
    else:
        pv_prd_prc.append(round(pv_prod/pv_max_prod,6))
        
    battery_cap_max.append(   cap_max  * input_uncertainty['battery_cap_per_pv_cap'].iloc[i]  )
    
output_sensitivity_tech['ES_cap'] =  savedata['battery']/battery_cap_max

output_sensitivity_tech['pv_prd_prc'] =  pv_prd_prc

#define labels

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

objs_labels = ['Cost [M€/year]', 'Emissions [ktonCO$_2$/year]', 'New desalter /n production [% of demand /nsatisfied]', 'Existing desalter production [% of demand /nsatisfied]', 'Water supply from ships [% of demand /nsatisfied]', 'Batteries capacity installed [% of potential capacity]','WS capacity installed [% of potential capacity]','PV relative production [%]']

#group the uncertainty sources based on the technology
group_ids =[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7]

#%% Plot parallel plot (Fig 4)

def parallel(J_s, showPDF = True, direction=None, xticks=None, labelxup = None, labelxdown = None,  n_bins = 30, mn_mx = None, colormap=None,  class_color = None, savepath = None, colorbarlabel = None, objs_labels  =None, dpi = 250, figsize = (37,15)):
    # plot a parallel plot of dataframe J_s, which contains the value of the performances of each portfolio for each considered objective 
      
    #define fontsizes:
    fontlabels = 22.5

    # objectives name
    cols = J_s.columns
    obj_names = list(cols)
    
    # number of objective
    nO = len(obj_names)
    
    # direction report for each objective a 1 is is has to be maximise, -1 if minimized
    if direction is None:
        direction = -np.ones(len(obj_names))
    
    #normalize objectives
    # J=np.absolute(J_s[:])
    J=J_s[:]
    
    # min and max
    if mn_mx is None:
        mn = np.array(J.min(axis=0))
        mX = np.array(J.max(axis=0))
    else:
        mn =  np.array(mn_mx.loc['lb'])
        mX = np.array(mn_mx.loc['ub'])
    
    for i in range(len(obj_names)):
        
        if direction[i] ==1:      
            J[obj_names[i]] = 1-(J_s[obj_names[i]] - mn[i])/(mX[i] - mn[i])
            
        else:
            J[obj_names[i]] = (J_s[obj_names[i]] - mn[i])/(mX[i] - mn[i])
            a= mn[i]
            mn[i] = mX[i]
            mX[i] = a

    mn_mx = np.round(np.vstack((mn, mX)).reshape(2, nO),3)[:]
    mn_mx = pd.DataFrame(mn_mx, columns = obj_names)

    mx=[]
    mn=[]
    for i in range(len(obj_names)):
            mini=str(mn_mx[obj_names[i]][1])
            maxi=str(mn_mx[obj_names[i]][0])
            mx.append(maxi)
            mn.append(mini)

    if cols is None:
        df = J
    else:
        df = J[cols]
    
    # if no class column is given, create a new class "Color" with a separate integer value for each portfolio
    if class_color is None:
        J_s['Colors'] = np.linspace(1,0, J.shape[0])
        class_color = 'Colors'
    
    nobjs = len(obj_names)
    n = len(J)
    class_col = J[class_color]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)
    
    # determine values to use for xticks
    ncols = len(df.columns)
    x = range(ncols)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    #plot parallel plot lines
    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), linewidth=1)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

   #plot histogram
    max_yvals = 12
            
    if showPDF == True:
        x=np.append(x,len(x))
        
        # Plot histograms on top subplot
        for i, obj in enumerate(obj_names):
            data = J_s[obj].dropna()

            reduction = 2 #pdf reduction compared to the whole axis
            data_norm = (data - mn_mx[obj].min())/(mn_mx[obj].max() - mn_mx[obj].min())
            data_norm = pd.concat([data_norm, pd.Series([0, 1 ])], ignore_index=True)
            hist_vals, bin_edges = np.histogram(data_norm, bins=n_bins, density=True)
            hist_vals[hist_vals > max_yvals] = max_yvals
            y_scaled = hist_vals / max_yvals
            heigth = (1)/n_bins
            ax.bar(x=i, width= y_scaled/reduction, height= heigth , bottom=bin_edges[:-1], color='black', alpha=0.6, zorder=10, align='edge', edgecolor='None')

    ax.set_xlim(x[0], x[-1]-1/(reduction+0.1))
    
    ax.set_xticks(x[0:-1])
    ax.set_xticks(np.arange(nobjs))
      
    ax2 = ax.twiny()
    ax2.set_xlim(x[0], x[-1]-1/(reduction+0.1))
    ax2.set_xticks(np.arange(nobjs))
    
    if labelxdown is None:
        
        labelxdown = []  
        for i in range(0,len(obj_names)):
            
            if objs_labels  is None:
                labelxdown.append(mn[i]+'\n\n'+obj_names[i])   
            else:
                labelxdown.append(mn[i]+'\n\n'+objs_labels[i])     
            
    ax.set_xticklabels(labelxdown,fontsize=fontlabels)
        
    if labelxup is None:
        
        labelxup = []  
        for i in range(0,len(obj_names)):
            labelxup.append(mx[i])  
            
    ax2.set_xticklabels(labelxup,fontsize=fontlabels)
    
    ax.get_yaxis().set_visible([])
    
    bounds = np.linspace(class_min,class_max,10)
    
    cbticks = np.linspace(np.amin(J_s[class_color]),np.amax(J_s[class_color]),10)
    
    cbticks = np.round(cbticks,decimals=2)
    
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, label =cbticks, boundaries=bounds, format='%.2f')
    cb.set_ticklabels(cbticks)
    
     # Example font dictionary
    font_properties = {'fontsize': 22.5}

    # Apply font properties to tick labels
    cb.ax.set_yticklabels(cbticks, fontdict=font_properties)

    if colorbarlabel is None:
        colorbarlabel = class_color
    cb.set_label(colorbarlabel, rotation=270, labelpad=50, fontsize=fontlabels) 
        
    if savepath is not None:
        plt.savefig(savepath, dpi=dpi ,  bbox_inches='tight')


# labelxup = ['152 M€', '0.28 MT', '240 T', '75%']
# labelxdown = ['62 M€ \nAnnual net costs', '0.1 MT \nAnnual CO$_2$ Emissions', ' 0 T  \nPM$_x$emissions','25% \nElectricity \nfrom import']

lb = [output_sensitivity_tech['monetary'].min() , output_sensitivity_tech['emissions'].min(), round(output_sensitivity_tech['new_des_prod'].min()), round(output_sensitivity_tech['old_des_prod'].min()), max(output_sensitivity_tech['ship_prod'].min(),0), 0,0,0 ]
ub = [output_sensitivity_tech['monetary'].max() , output_sensitivity_tech['emissions'].max(), output_sensitivity_tech['new_des_prod'].max(), output_sensitivity_tech['old_des_prod'].max(), output_sensitivity_tech['ship_prod'].max(), 1,1 ,1]

lb = [output_sensitivity_tech['monetary'].min() , output_sensitivity_tech['emissions'].min(), 0,0,0, 0, 0,0 ]
ub = [output_sensitivity_tech['monetary'].max() , output_sensitivity_tech['emissions'].max(), 1,1,1, 1,1 ,1]

mn_mx = pd.DataFrame(columns = output_sensitivity_tech.columns, index = ['lb','ub'])
mn_mx.loc['lb'] = lb
mn_mx.loc['ub'] = ub
mn_mx = mn_mx.astype(float)
objs_labels = ['Cost \n[M€/year]', 'Emissions \n[ktonCO$_2$/year]','New desalter \nproduction \n[% of demand satisfied]', 'Existing desalter \nproduction \n[% of demand satisfied]', 'Water supply \nfrom ships \n[% of demand satisfied]', 'Batteries capacity installed \n[% of potential capacity]','WS capacity installed \n[% of potential capacity]','PV relative production \n[% of potential production]']

class_color_id = 4
class_color = output_sensitivity_tech.columns[class_color_id]
colorbarlabel = 'Water supply from ships [% of demand satisfied]'

parallel(output_sensitivity_tech, savepath = os.path.join(figure_folder, 'parallel_tech_ship.png') , class_color = class_color, showPDF = True, n_bins = 60 , colormap = 'coolwarm', colorbarlabel=colorbarlabel, mn_mx=mn_mx, objs_labels  =objs_labels , figsize = (50,15))

#%% parallel highlights

def parallel_highlight(J_s, representative_points, direction=None, xticks=None, labelxup=None,
                       labelxdown=None, mn_mx=None, colormap=None, class_color=None,
                       savepath=None, colorbarlabel=None, objs_labels=None,
                       dpi=250, figsize=(37, 15)):

    fontlabels = 22.5
    obj_names = list(J_s.columns)
    nO = len(obj_names)

    if direction is None:
        direction = -np.ones(len(obj_names))

    J = J_s.copy()

    if mn_mx is None:
        mn = np.array(J.min(axis=0))
        mX = np.array(J.max(axis=0))
    else:
        mn = np.array(mn_mx.loc['lb'])
        mX = np.array(mn_mx.loc['ub'])

    for i in range(len(obj_names)):
        if direction[i] == 1:
            J[obj_names[i]] = 1 - (J_s[obj_names[i]] - mn[i]) / (mX[i] - mn[i])
        else:
            J[obj_names[i]] = (J_s[obj_names[i]] - mn[i]) / (mX[i] - mn[i])
            mn[i], mX[i] = mX[i], mn[i]

    mn_mx_df = pd.DataFrame(np.round(np.vstack((mn, mX)).reshape(2, nO), 3), columns=obj_names)

    mx, mn_str = [], []
    for i in range(len(obj_names)):
        mx.append(str(mn_mx_df[obj_names[i]][0]))
        mn_str.append(str(mn_mx_df[obj_names[i]][1]))

    df = J[obj_names]

    if class_color is None:
        J_s['Colors'] = np.linspace(1, 0, J.shape[0])
        class_color = 'Colors'

    class_col = J[class_color]
    class_min = class_col.min()
    class_max = class_col.max()

    x = list(range(len(obj_names)))
    plt.figure(figsize=figsize)
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    rep_ids = set(representative_points.values())

    for i in range(len(J)):
        y = df.iloc[i].values
        portfolio_id = J_s.index[i]
        if portfolio_id in rep_ids:
            kls = class_col.iat[i]
            color = Colorm((kls - class_min) / (class_max - class_min))
            linewidth = 10
            zorder = 5
        else:
            color = 'lightgrey'
            linewidth = 0.8
            zorder = 1

        ax.plot(x, y, color=color, linewidth=linewidth, zorder=zorder)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xlim(x[0], x[-1])

    ax.set_xticks(x)
    ax2 = ax.twiny()
    ax2.set_xlim(x[0], x[-1])
    ax2.set_xticks(x)

    if labelxdown is None:
        labelxdown = [mn_str[i] + '\n\n' + (objs_labels[i] if objs_labels else obj_names[i])
                      for i in range(len(obj_names))]

    ax.set_xticklabels(labelxdown, fontsize=fontlabels)

    if labelxup is None:
        labelxup = mx

    ax2.set_xticklabels(labelxup, fontsize=fontlabels)
    ax.get_yaxis().set_visible(False)

    bounds = np.linspace(class_min, class_max, 10)
    cbticks = np.round(np.linspace(class_min, class_max, 10), 2)

    cax, _ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional',
                                    ticks=bounds, label=cbticks,
                                    boundaries=bounds, format='%.2f')
    cb.set_ticklabels(cbticks)
    cb.ax.set_yticklabels(cbticks, fontdict={'fontsize': 22.5})

    if colorbarlabel is None:
        colorbarlabel = class_color
    cb.set_label(colorbarlabel, rotation=270, labelpad=50, fontsize=fontlabels)

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight')

representative_points = {
    'above_both': 16050,
    'below_costs': 17164,
    'below_emissions': 21204,
    'below_both': 3737
}

parallel_highlight(output_sensitivity_tech, representative_points, savepath = os.path.join(figure_folder, 'parallel_tech_higlights.png') , class_color = class_color, colormap = 'coolwarm', colorbarlabel=colorbarlabel, mn_mx=mn_mx, objs_labels  =objs_labels , figsize = (50,15))

#%% plot conditional and unconditional CDFs

objs_labels = ['Cost \n[M€/year]', 'Emissions \n[ktonCO$_2$/year]','New desalter production \n[% of demand satisfied]', 'Existing desalter production \n[% of demand satisfied]', 'Water supply from ships \n[% of demand satisfied]', 'Batteries capacity installed \n[% of potential capacity]','WS capacity installed \n[% of potential capacity]','PV relative production \n[% of potential production]']
 
save_fig = 0
n = 10 # number of conditioning intervals

X_unc = np.array(input_uncertainty, 'float')
    

for i in range(2,len(objs_labels)):
    column_data = output_sensitivity_tech.iloc[:, i]

    Y = np.array(column_data, 'float')
    Y = np.nan_to_num(Y, nan=0.0) #remove Nans
    
    # Compute and plot conditional and unconditional CDFs:
    YF, FU, FC, xc = pawn_plot_cdf_mod(X_unc, Y, n, cbar=True, n_col=3, labelinput=X_Labels, Y_Label = objs_labels[i])
    # Adjust the plot size after plotting
    plt.gcf().set_size_inches(12, 25)  # Width=12 inches, Height=8 inches

    # Adjust subplot spacing
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        hspace=0.5, wspace=0.9)
    if save_fig == 1:
        plt.savefig( os.path.join(figure_folder,  'CDFs_tech_' + str(output_sensitivity_tech.columns[i])+'.png'), dpi=250 ,  bbox_inches='tight')
    
    plt.show()
    
#%% extract PAWN indexes

n = 10  # Number of conditioning intervals
X_unc = np.array(input_uncertainty, dtype=float)
Nboot = 1000

# Get matrix dimensions (num outputs × num inputs)
num_outputs = len(output_sensitivity_tech.columns)
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

    column_data = output_sensitivity_tech.iloc[:, i]
    Y = np.array(column_data, dtype=float)
    
    
    Y [Y  < 0.0000001] = 0
    
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
    
#%% chechered table (max)
save_fig = 0

objs_labels = ['Cost \n[M€/year]', 'Emissions \n[ktonCO$_2$/year]','New desalter \nproduction \n[% of demand \nsatisfied]', 'Existing desalter \nproduction \n[% of demand \nsatisfied]', 'Water supply \nfrom ships \n[% of demand \nsatisfied]', 'Batteries capacity \n installed \n[% of potential \ncapacity]','WS capacity \ninstalled \n[% of potential \ncapacity]','PV relative \nproduction \n[% of potential \nproduction]']

def plotPAWNheatmap(K_m, K_lb, K_ub,  save_fig, name = 'PAWNindexes_heatmap', plot_IDs = None, figsize = (13, 15)):

    # Extract group boundaries
    group_ids =[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7]
    group_changes = np.where(np.diff(group_ids) != 0)[0] + 1  # Get indices where group changes
    
    # Get matrix dimensions (num outputs × num inputs)
    if plot_IDs is None:
        plot_IDs = range(K_m.shape[0])
    num_outputs = len(plot_IDs)
    num_inputs = K_m.shape[1]

    selected_labels = [objs_labels[i] for i in list(plot_IDs)]

    # **Transpose matrices** to flip the table (inputs as rows, outputs as columns)
    K_m  = K_m[plot_IDs,:].T
    K_lb = K_lb[plot_IDs,:].T
    K_ub = K_ub[plot_IDs,:].T
    
    # Normalize each column independently
    norm_values = np.zeros_like(K_m)
    for j in range(num_outputs):  # Normalize per output column
        col_min = np.min(K_m[:, j])
        col_max = np.max(K_m[:, j])
        norm_values[:, j] = (K_m[:, j] - col_min) / (col_max - col_min + 1e-6)
    
    # Create a single grayscale heatmap (column-wise normalization)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(norm_values, cmap="Greys", aspect="auto")
    
    # Add text inside each cell
    for i in range(num_inputs):  # Inputs as rows
        for j in range(num_outputs):  # Outputs as columns
            text = f"{K_m[i, j]:.2f}\n[{K_lb[i, j]:.2f}, {K_ub[i, j]:.2f}]"
            ax.text(j, i, text, ha="center", va="center", fontsize=14, color="red")
    
    # Set ticks and labels (swapped)
    ax.set_xticks(np.arange(num_outputs))
    ax.set_yticks(np.arange(num_inputs))
    ax.set_xticklabels(selected_labels, rotation=0, fontsize=14)  # Outputs now on x-axis
    ax.set_yticklabels(X_Labels, fontsize=14)  # Inputs now on y-axis
    
    # **Add thick black vertical lines between columns**
    for j in range(1, num_outputs):  # Avoid first column
        ax.vlines(j - 0.5, ymin=-0.5, ymax=num_inputs - 0.5, colors="black", linewidth=2)
    
    # **Add thick black horizontal lines at group separations**
    for g in group_changes:
        ax.hlines(g - 0.5, xmin=-0.5, xmax=num_outputs - 0.5, colors="black", linewidth=2)
    
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if required
    if save_fig == 1:
        plt.savefig(os.path.join(figure_folder, name), dpi=250, bbox_inches='tight')
    
    plt.show()

plot_IDs = range(2,len(KS_max_m_matrix))
#plot checkered table for the index MAX
plotPAWNheatmap(KS_max_m_matrix, KS_max_lb_matrix, KS_max_ub_matrix,  save_fig, name = 'PAWNindexesMAXtech_heatmap', plot_IDs = plot_IDs)

#plot checkered table for the index MEAN
# plotPAWNheatmap(KS_mean_m_matrix, KS_mean_lb_matrix, KS_mean_ub_matrix,  save_fig, name = 'PAWNindexesMEANtech_heatmap', plot_IDs = plot_IDs)
