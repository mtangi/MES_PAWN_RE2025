# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:37:16 2024

@author: TANGI

This calliope function is created to be coupled with a multiobjective evolutionary algorithm (MOEA) with variables equal 
to the location of the model and the year of the simulation
"""

import time
import calliope
import csv
import os
import stat

#%% function which includes the storage and hydrogen technologies 

def calliope4pawn(sim_id, ph, result_file_path):

    t_loop = time.time()

    #remove warnings from calliope
    calliope.set_log_verbosity('error', include_solver_output=False)

    # define CALLIOPE model   
    model = calliope.Model(ph)

    obj_weight = [model.run_config['objective_options']['cost_class']['monetary'], model.run_config['objective_options']['cost_class']['emissions']] 

    name_tech = [ 'desalter_existing','desalter_planned', 'pv_existing_ground', 'pv_existing_roof',
           'pv_planned_roof', 'supply_ship', 'thermo','battery', 'water_storage_existing', 'water_storage_planned']
    name_tech_storage = ['battery','water_storage_existing' , 'water_storage_planned']

    try:
        model.run()
        objs = model.get_formatted_array('cost').sum(['locs','techs']).to_pandas().reindex(['monetary', 'emissions'])

        #save also energy cap and storage cap
        energy_cap = model.get_formatted_array('energy_cap').sum(['locs']).to_pandas()[name_tech].T
        storage_cap = list(model.get_formatted_array('storage_cap').sum(['locs']).to_pandas()[name_tech_storage].T)
        energy_cap[name_tech_storage] = storage_cap
        energy_cap = list(energy_cap)
        
        # find which carrier is used in the system
        car = model.get_formatted_array('carrier_prod')['carriers'].to_pandas().T
        cp_el = model.get_formatted_array('carrier_prod').loc[{'carriers':car[0]}].sum(['locs','timesteps']).to_pandas()[name_tech] #production of each tech for carrier i
        cp_wt = model.get_formatted_array('carrier_prod').loc[{'carriers':car[1]}].sum(['locs','timesteps']).to_pandas()[name_tech]#production of each tech for carrier i
        cp = list(cp_el + cp_wt )
        
        # Set full permissions
        os.chmod(result_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        #load file for saving results
        with open(result_file_path, mode='a', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow([sim_id] + list(objs) + obj_weight + energy_cap + cp + [time.strftime('%H:%M:%S', time.gmtime(time.time() - t_loop))])
        
    except:
        #load file for saving results
        with open(result_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sim_id] + [0,0] + obj_weight + [0] * 21)

#%% function for th baseline scenario

def calliope4pawn_baseline(sim_id, ph, result_file_path):

    t_loop = time.time()

    #remove warnings from calliope
    calliope.set_log_verbosity('error', include_solver_output=False)

    # define CALLIOPE model   
    model = calliope.Model(ph)

    obj_weight = [model.run_config['objective_options']['cost_class']['monetary'], model.run_config['objective_options']['cost_class']['emissions']] 

    name_tech = [ 'desalter_existing', 'pv_existing_ground', 'pv_existing_roof',
            'supply_ship', 'thermo','water_storage_existing']
    name_tech_storage = ['water_storage_existing']

    try:
        model.run()
        objs = model.get_formatted_array('cost').sum(['locs','techs']).to_pandas().reindex(['monetary', 'emissions'])

        #save also energy cap and storage cap
        energy_cap = model.get_formatted_array('energy_cap').sum(['locs']).to_pandas()[name_tech].T
        storage_cap = list(model.get_formatted_array('storage_cap').sum(['locs']).to_pandas()[name_tech_storage].T)
        energy_cap[name_tech_storage] = storage_cap
        energy_cap = list(energy_cap)
        
        # find which carrier is used in the system
        car = model.get_formatted_array('carrier_prod')['carriers'].to_pandas().T
        cp_el = model.get_formatted_array('carrier_prod').loc[{'carriers':car[0]}].sum(['locs','timesteps']).to_pandas()[name_tech] #production of each tech for carrier i
        cp_wt = model.get_formatted_array('carrier_prod').loc[{'carriers':car[1]}].sum(['locs','timesteps']).to_pandas()[name_tech]#production of each tech for carrier i
        cp = list(cp_el + cp_wt )
        
        # Set full permissions
        os.chmod(result_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        #load file for saving results
        with open(result_file_path, mode='a', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow([sim_id] + list(objs) + obj_weight + energy_cap + cp + [time.strftime('%H:%M:%S', time.gmtime(time.time() - t_loop))])
        
    except:
        #load file for saving results
        with open(result_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sim_id] + [0,0] + obj_weight + [0] * 21)

