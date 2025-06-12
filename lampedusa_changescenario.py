# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:19:46 2024

@author: TANGI

The script contains functions that change the input configuration files in CALLIOPE for the energy system according to the value of the uncertainty sources
"""

import os
import shutil
import yaml
import pandas as pd
from calculate_demand import *


#%% change scenario (for future simulations)

def change_sc(destination_folder, data_uncertanty, model_yaml = "model_Lampedusa.yaml", tech_yaml = "techs_Lampedusa.yaml", location_yaml = 'locations_Lampedusa.yaml'):
           
    island = 'Lampedusa'

    #create folder for temporary files
    if not os.path.exists(destination_folder):
        # if the folder directory is not present 
        # then create it.
        os.makedirs(destination_folder)
    
    ########################################## timeseries data ###########################################################
    
    #change the PUN and the pv_resource in timeseries_data to match the sim type
    path = 'timeseries_data'
    
    #copy the solar producibility file 
    filename_pv = 'pv_resource_'+ str(int(data_uncertanty['pv_year'])) +'.csv'
    new_filename_pv = 'pv_resource.csv'
    shutil.copy(os.path.join(island, path, filename_pv), os.path.join(destination_folder, new_filename_pv))
    
    # change the water and energy demands according to the changes in population
    island_demands("", destination_folder, expop_increase = data_uncertanty['expop_increase'],  wd_res = data_uncertanty['wd_res'], wd_expop = data_uncertanty['wd_expop'])
                   
    #copy the ship capacity file 
    shutil.copy(os.path.join(island,'timeseries_data', 'ship_resource.csv'), os.path.join(destination_folder, 'ship_resource.csv'))
    
    ########################################## model YAML ################################################################
    
    # Path to the YAML file
    shutil.copy(model_yaml, os.path.join(destination_folder, model_yaml))
    yaml_model = os.path.join(destination_folder, model_yaml)
    
    with open(yaml_model, 'r') as file:
        data = yaml.safe_load(file)
    
    # Convert the data to string, replace the old string with the new string, and convert back to YAML
    data['model']['timeseries_data_path'] = ""
    data['run']['objective_options']['cost_class']['emissions'] = float(data_uncertanty['weight_co2'])
    data['run']['objective_options']['cost_class']['monetary'] = 1 - float(data_uncertanty['weight_co2'])

    with open(yaml_model, 'w') as file:
        yaml.safe_dump(data, file)

    ########################################## tech YAML ###############################################################
    
    # Path to the YAML file
    shutil.copy(tech_yaml, os.path.join(destination_folder, tech_yaml))
    yaml_tech = os.path.join(destination_folder, tech_yaml)
        
    with open(yaml_tech, 'r') as file:
        data = yaml.safe_load(file)
    
    #change fuel costs and emission for the generators
    data['techs']['thermo']['costs']['monetary']['om_con'] = float(data_uncertanty['cost_fuel'])  #€/m3
    data['techs']['thermo']['costs']['emissions']['om_con'] =  float(data_uncertanty['emission_fuel'])  #€/m3
    
    #change ship emission factor based on the distance (round trip)
    ship_distance = float(pd.read_csv('boat_trip.csv')[island][1]) #km
    data['techs']['supply_ship']['costs']['emissions']['om_con'] = float(data_uncertanty['EF_ship'])*ship_distance*2  #€/m3
    
    #change ship efficiency based on the aqueduct losses
    data['techs']['supply_ship']['constraints']['energy_eff'] = float(1 - data_uncertanty['acqueduct_losses']) 
    
    #change import water cost
    data['techs']['supply_ship']['costs']['monetary']['om_con'] = float(data_uncertanty['cost_water'])   #€/m3
    
    # #change ship capacity
    # ship_capacity = float(pd.read_csv('boat_trip.csv')[island][2]) #km
    # data['techs']['supply_ship']['constraints']['energy_cap_max'] = ship_capacity  #m3/trip

    #change desalter data  (account for the transmission losses of the aqueduct by lowering the efficiency)
    data['techs']['desalter_existing']['constraints']['energy_eff'] = float(1/data_uncertanty['desalter_eff_old'] * (1 - data_uncertanty['acqueduct_losses']) ) 
    data['techs']['desalter_planned']['constraints']['energy_eff'] = float(1/data_uncertanty['desalter_eff_new'] * (1 - data_uncertanty['acqueduct_losses']) ) 
    data['techs']['desalter_planned']['costs']['monetary']['energy_cap'] = float(data_uncertanty['cost_desalter'] / data['techs']['desalter_planned']['constraints']['energy_cap_max'])
    
    #change pv parameters
    PV_data = pd.read_csv('pv_island.csv')
    PV_data = PV_data.set_index('Index')[island]  
    
    data['techs']['pv_existing_roof']['constraints']['energy_cap_equals'] = float(PV_data['PV_power_installed_isola_roof'])
    data['techs']['pv_existing_ground']['constraints']['energy_cap_equals'] = float(PV_data['PV_power_installed_isola_ground'])
    
    # change available area for roof pv by changing both the prc of roof area available and the surface requirements for already existing roof pv
    data['techs']['pv_planned_roof']['constraints']['resource_area_max'] = float(max(0,float(PV_data['building_extension_island']) * data_uncertanty['prc_area_available'] * 10**6 - float(PV_data['PV_power_installed_isola_roof'])*data_uncertanty['pv_resource_area_per_energy_cap']) ) 
    
    # change other data for pv
    data['techs']['pv_planned_roof']['constraints']['resource_area_per_energy_cap'] = float(data_uncertanty['pv_resource_area_per_energy_cap']) 
    data['techs']['pv_planned_roof']['costs']['monetary']['energy_cap'] = float(data_uncertanty['pv_energy_cap_cost']) 
    
    #change water storage capacity data (capacity, loss rate and efficiency due to aqueduct losses)
    data['techs']['water_storage_planned']['constraints']['storage_cap_max'] = float(data_uncertanty['waterstorage_new_cap_max'])
    data['techs']['water_storage_existing']['constraints']['storage_loss'] = float(data_uncertanty['waterstorage_ex_loss'])
    data['techs']['water_storage_planned']['costs']['monetary']['storage_cap'] = float(data_uncertanty['waterstorage_new_storage_cap_cost'])
    

    #change the data for the new battery storage 
    data['techs']['battery']['constraints']['storage_cap_max'] = float(data['techs']['pv_planned_roof']['constraints']['resource_area_max'] / data_uncertanty['pv_resource_area_per_energy_cap']  * data_uncertanty['battery_cap_per_pv_cap'])
    data['techs']['battery']['constraints']['energy_cap_max'] = float(data['techs']['pv_planned_roof']['constraints']['resource_area_max'] / data_uncertanty['pv_resource_area_per_energy_cap']  * data_uncertanty['battery_cap_per_pv_cap'])
    data['techs']['battery']['costs']['monetary']['storage_cap'] = float(data_uncertanty['battery_storage_cap_cost'])
    
    #save data in new yaml
    with open(yaml_tech, 'w') as file:
        yaml.safe_dump(data, file)
            
    ########################################## location YAML #############################################################
    
    shutil.copy(location_yaml, os.path.join(destination_folder, location_yaml))
        
    return destination_folder
            
#%% #%% change scenario (for baseline simulations)

def change_sc_baseline(destination_folder, data_uncertanty, model_yaml = "model_Lampedusa_baseline.yaml", tech_yaml = "techs_Lampedusa.yaml", location_yaml = 'locations_Lampedusa_baseline.yaml'):
           
    island = 'Lampedusa'

    #create folder for temporary files
    if not os.path.exists(destination_folder):
        # if the folder directory is not present 
        # then create it.
        os.makedirs(destination_folder)
    
    ########################################## timeseries data ###########################################################
    
    #change the PUN and the pv_resource in timeseries_data to match the sim type
    path = 'timeseries_data'
    
    #copy the solar producibility file 
    filename_pv = 'pv_resource_'+ str(int(data_uncertanty['pv_year'])) +'.csv'
    new_filename_pv = 'pv_resource.csv'
    shutil.copy(os.path.join(island, path, filename_pv), os.path.join(destination_folder, new_filename_pv))
    
    # change the water and energy demands according to the changes in population
    island_demands("", destination_folder, expop_increase = 0,  wd_res = data_uncertanty['wd_res'], wd_expop = data_uncertanty['wd_expop'])
                   
    #copy the ship capacity file 
    shutil.copy(os.path.join(island,'timeseries_data', 'ship_resource.csv'), os.path.join(destination_folder, 'ship_resource.csv'))
    
    ########################################## model YAML ################################################################
    
    # Path to the YAML file
    shutil.copy(model_yaml, os.path.join(destination_folder, model_yaml))
    yaml_model = os.path.join(destination_folder, model_yaml)
    
    with open(yaml_model, 'r') as file:
        data = yaml.safe_load(file)
    
    # Convert the data to string, replace the old string with the new string, and convert back to YAML
    data['model']['timeseries_data_path'] = ""

    with open(yaml_model, 'w') as file:
        yaml.safe_dump(data, file)


    ########################################## tech YAML ###############################################################
    
    # Path to the YAML file
    shutil.copy(tech_yaml, os.path.join(destination_folder, tech_yaml))
    yaml_tech = os.path.join(destination_folder, tech_yaml)
        
    with open(yaml_tech, 'r') as file:
        data = yaml.safe_load(file)
    
    #change fuel costs and emission for the generators
    data['techs']['thermo']['costs']['emissions']['om_con'] =  float(data_uncertanty['emission_fuel'])  #€/m3
    
    #change ship emission factor based on the distance (round trip)
    ship_distance = float(pd.read_csv('boat_trip.csv')[island][1]) #km
    data['techs']['supply_ship']['costs']['emissions']['om_con'] = float(data_uncertanty['EF_ship'])*ship_distance*2  #€/m3
    
    #change ship efficiency based on the aqueduct losses
    data['techs']['supply_ship']['constraints']['energy_eff'] = float(1 - data_uncertanty['acqueduct_losses']) 
    
    #change desalter data  (account for the transmission losses of the aqueduct by lowering the efficiency)
    data['techs']['desalter_existing']['constraints']['energy_eff'] = float(1/data_uncertanty['desalter_eff_old'] * (1 - data_uncertanty['acqueduct_losses']) ) 

    #change pv parameters
    PV_data = pd.read_csv('pv_island.csv')
    PV_data = PV_data.set_index('Index')[island]  
    
    data['techs']['pv_existing_roof']['constraints']['energy_cap_equals'] = float(PV_data['PV_power_installed_isola_roof'])
    data['techs']['pv_existing_ground']['constraints']['energy_cap_equals'] = float(PV_data['PV_power_installed_isola_ground'])
    
    #change water storage capacity data (capacity, loss rate and efficiency due to aqueduct losses)
    data['techs']['water_storage_existing']['constraints']['storage_loss'] = float(data_uncertanty['waterstorage_ex_loss'])

    #save data in new yaml
    with open(yaml_tech, 'w') as file:
        yaml.safe_dump(data, file)
            
    ########################################## location YAML #############################################################
    
    shutil.copy(location_yaml, os.path.join(destination_folder, location_yaml))
        
    return destination_folder
            

    
    
