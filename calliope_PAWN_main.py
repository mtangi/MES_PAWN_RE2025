# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:50:38 2025

@author: TANGI

The script contains the code to generate the samples of uncertainty sources via Latin hypercube sampling, to run CALLIOPE for each sample via parallel computing to obtain optimized future configurations, and to collect and export the results
"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import pandas as pd

import datetime
import scipy.stats as st

# Import SAFE modules:
from safepython.sampling import AAT_sampling # module to perform the input sampling

import csv

from joblib import Parallel, delayed

from lampedusa_changescenario import change_sc

import shutil

#%% define uncertainty sources range

data = {
    'expop_increase': [-0.10, 0.25],
    'wd_res': [0.22, 0.3], #+-0.2%
    'wd_expop': [0.16, 0.26], #+-0.2%
    
    'EF_ship': [0.006, 0.1],
    'cost_water': [6, 12],
    
    'desalter_eff_old': [6, 8],
    'desalter_eff_new': [5, 7],
    'cost_desalter': [5688552.14, 7110690.175 ],  # total cost, to be divided by efficiency
    
    'cost_fuel': [0.0741,  0.1065 ],  # https://sisen.mase.gov.it/dgsaie/prezzi-annuali-carburanti?pid=3
    'emission_fuel': [0.34, 0.425],
    
    'prc_area_available': [0.0, 0.05],  
    'pv_resource_area_per_energy_cap': [4, 8],
    'pv_energy_cap_cost': [1005.6, 1508], #+-0.2%
    'pv_year': [0.17596, 0.187173],
    
    'acqueduct_losses': [0, 0.3], #report 2022 Legambiente https://www.legambiente.it/wp-content/uploads/2022/06/IsoleSostenibili22.pdf
    'waterstorage_new_cap_max': [0, 10000],
    'waterstorage_new_storage_cap_cost': [ 2560.0, 3840.0], #+-0.2%
    'waterstorage_ex_loss': [0, 0.01],
    
    'battery_cap_per_pv_cap': [1, 3],
    'battery_storage_cap_cost': [640.0, 960.0], #+-0.2%
    
    'weight_co2': [0, 1]
}
# Create DataFrame and populate with data
uncertanty_variables = pd.DataFrame.from_dict(data, orient='index', columns=['min', 'max'])

# Corresponding group numbers
uncertanty_variables['group_id'] = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7]

#load csv file with the average producibility and the year in ascending order
year_list = pd.read_csv('Lampedusa/timeseries_data/producibility_year_list.csv').iloc[:, -2:].to_numpy()

#%% define result folder

#define folder with results
result_folder = 'sets_PAWN_'+ datetime.datetime.now().strftime("%Y%m%d_%H%M")
if not os.path.exists(result_folder):
    # if the folder directory is not present 
    # then create it.
    os.makedirs(result_folder)        

#create csv with results
file_name =   os.path.join(result_folder,'savedata.csv') 

#create csv file for other results
column_names = ['id','monetary', 'emissions', 'monetary.w','emissions.w', 
                'desalter_existing','desalter_planned', 'pv_existing_ground', 'pv_existing_roof',
                      'pv_planned_roof', 'supply_ship', 'thermo','battery', 'water_storage_existing', 'water_storage_planned',
                 'desalter_existing.p','desalter_planned.p', 'pv_existing_ground.p', 'pv_existing_roof.p',
                       'pv_planned_roof.p', 'supply_ship.p', 'thermo.p','battery.p', 'water_storage_existing.p', 'water_storage_planned.p',
                'runtime']

with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)
    
#%% sample inputs space 

N = 25000  #  Number of samples

M = len(data) # number of uncertain parameters
xmin = list(uncertanty_variables['min'])
xmax = list(uncertanty_variables['max'])

# Parameter distributions:
distr_fun = [st.uniform] * M # uniform distribution

# The shape parameters of the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

samp_strat = 'lhs' # Latin Hypercube
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, N)

#save sample as csv
df = pd.DataFrame(X, columns=uncertanty_variables.index)
filename = os.path.join(result_folder, 'inputspace.csv')
df.to_csv(filename, index=False)
        
#%% run simulations for all samples and save the results (parallel computing with 24 cores)

# Convert dictionary values to a NumPy array
index = list(data.keys())  #

num_cores = 24

def runsim(row, i,  result_folder):
    
    data_uncertanty = pd.Series(row, index=index)
    
    #find the name of the year in the list with avg producibility closest to the selected value
    data_uncertanty.loc['pv_year'] = int(year_list[np.argmin(np.abs(year_list[:,1] - float(data_uncertanty.loc['pv_year']))),0])
    
    #create folder for the scenario setting
    scenario_folder = os.path.join(result_folder, 'scenario_'+f"{i:05d}")

    #change input data according to uncertanties     
    change_sc(scenario_folder, data_uncertanty)
    
    # Modify the data by modifying the energy_cap value in the model data
    ph = os.path.join(scenario_folder, "model_Lampedusa.yaml")

    varsstr = str(i) + ',\'' + ph + '\'' + ',\'' + file_name + '\''
        
    #name of the python scritp to run calliope (to avoid memory leak bug)
    name_script = "python -c \"from calliope_PAWN_function import *; calliope4pawn(" + varsstr + ")\""
    
    # Chiamo script python che esegue il modello
    os.system(name_script)
    
    #remove the folder with the scenario settings to save space
    try:
        to_remove = os.path.join(scenario_folder)
        os.chmod(to_remove, 0o777)
        # Remove the folder and all its contents
        shutil.rmtree(to_remove)
    except FileNotFoundError:
        print(f"Folder '{to_remove}' does not exist.")
    except PermissionError:
        print(f"Permission denied: unable to delete '{to_remove}'.")
    except Exception as e:
        print(f"An error occurred: {e}")    
     
Parallel(n_jobs=num_cores, backend="loky")(delayed(runsim)(row, i, result_folder) for i, row in enumerate(X, start=1) )

