# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:14:24 2024

@author: TANGI

The script contains the function "island_demands" that changes the water and energy demand of the system according to the value of the uncertainty sources
"""

import pandas as pd
import numpy as np
import datetime
import os
from platypus import *
import openpyxl
#%%

# Initialize variables
year = 2022  # Example year

# Number of days in each month (non-leap year) with Italian names
days_in_month = {
    'Gennaio': 31, 'Febbraio': 28, 'Marzo': 31, 'Aprile': 30, 'Maggio': 31,
    'Giugno': 30, 'Luglio': 31, 'Agosto': 31, 'Settembre': 30,
    'Ottobre': 31, 'Novembre': 30, 'Dicembre': 31
}

# Define the summer months using Italian names
summer_months = ['Giugno', 'Luglio', 'Agosto']

#%% Define function 

def island_demands(path_in = "", path_save = "", expop_increase = 0,  wd_res = 0.26, wd_expop = 0.2, migrant = 0 ):
    
    ################################ ENERGY DEMAND #####################################
    
    # Load the Excel file for the energy demands
    file_path = os.path.join(path_in, 'dati_energia_Lampedusa.xlsx')  # Replace with your actual file path
    dmnd = pd.ExcelFile(file_path).parse("Dati").set_index("Mesi")
    
    # Load the Excel file for the diurnal energy profiles
    file_path = os.path.join(path_in,'diurnal_profiles_power.xlsx')  # Replace with your actual file path
    hourcrv = pd.ExcelFile(file_path).parse("Lampedusa").set_index("Ore")
    
    # Create an empty DataFrame to store the hourly demand
    df = pd.DataFrame()
    hourly_demand_energy = pd.DataFrame()
    
    start_date = datetime.datetime(year, 1, 1)  # Start from January 1st
    
    # Loop through each month and calculate the hourly demand
    for index, row in dmnd.iterrows():
        month = row.name
        # row = row.drop('MESI')
        days = days_in_month[month]  # Get the correct number of days for the month
        #increase the categories "Alberghi" and "Altro" according to the increase in external population
        montly_demand = row['Domestico'] + row['Illum. pubblica']  + (row['Alberghi'] + row['Altro'] + row['PA'] + row['Pescicoltura'] + row['Aeroporto'] )*(1+expop_increase)     
        # montly_demand = row['Totale no dissalatori']
        daily_demand = montly_demand / -days  # Divide monthly demand by the exact number of days
    
        if month in summer_months:
            # Use summer curve
            hourly_curve = hourcrv['Estivo - curva']
        else:
            # Use winter curve
            hourly_curve = hourcrv['Invernale - curva']
        
        # Calculate hourly demand by multiplying daily demand by the curve
        hourly_energy = daily_demand * hourly_curve
        
        # Repeat for the exact number of days in the month
        hourly_energy_full_month = pd.concat([hourly_energy] * days, ignore_index=True)
        
        # Add the hourly demand for the current month to the main DataFrame
        df = pd.concat([df, hourly_energy_full_month], ignore_index=True)
        
        # Generate timestamps for each hour in the month
        for day in range(days):
            for hour in range(24):
                timestamp = (start_date + datetime.timedelta(days=day, hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
                # Create a new row as a DataFrame
                new_row = pd.DataFrame({'timestamp': [timestamp], 0: [hourly_energy_full_month.iloc[day * 24 + hour]]})
                
                # Concatenate the new row with the existing DataFrame
                hourly_demand_energy = pd.concat([hourly_demand_energy, new_row], ignore_index=True)

        # Update the start date to the next month
        start_date += datetime.timedelta(days=days)
        
    # hourly_demand2.set_index('timestamp', inplace=True)
    # Rename a column from 'old_name' to 'new_name'
    hourly_demand_energy = hourly_demand_energy.rename(columns={0: 'X1'})
    
    # Export to a CSV file
    hourly_demand_energy.to_csv(os.path.join(path_save, 'hourly_energy_demand.csv') , index=False)
        
    ################################# WATER DEMAND #####################################
     
    #load resident and maximum population numbers
    stimepop = pd.read_csv(os.path.join(path_in,'stime_pop.csv'))
    stimepop = stimepop.set_index('Isola').loc[['Lampedusa']]
    
    #use the patters of energy demand to estimate population shifts every month and water demand
    pattern = (dmnd['Totale no dissalatori']- dmnd['Totale no dissalatori'].min())/(dmnd['Totale no dissalatori'].max() - dmnd['Totale no dissalatori'].min())
    # pop_month = pattern*float((stimepop['Max'] - stimepop['Residenti'])*(1+expop_increase)) + float(stimepop['Residenti'])
    
    #daily water demand per month
    water_demand_month = pattern*float((stimepop['Max'] - stimepop['Residenti'])*(1+expop_increase))*wd_expop + float(stimepop['Residenti'])*wd_res
    
    # Load the Excel file for the diurnal energy profiles
    file_path = os.path.join(path_in,'diurnal_profiles_water.csv') 
    hourcrv = pd.read_csv(file_path).set_index("hour")
    
    # Create an empty DataFrame to store the hourly demand
    df = pd.DataFrame()
    hourly_demand_water = pd.DataFrame()
    
    start_date = datetime.datetime(year, 1, 1)  # Start from January 1st
    
    for index, value in water_demand_month.items():

        month = index
        days = days_in_month[month]  # Get the correct number of days for the month
        daily_demand = value
        
        hourly_curve = hourcrv['water']
        
        # Calculate hourly demand by multiplying daily demand by the curve
        hourly_water = -daily_demand * hourly_curve
        
        # Repeat for the exact number of days in the month
        hourly_water_full_month = pd.concat([hourly_water] * days, ignore_index=True)
        
        # Add the hourly demand for the current month to the main DataFrame
        df = pd.concat([df, hourly_water_full_month], ignore_index=True)
        
        # Generate timestamps for each hour in the month
        for day in range(days):
            for hour in range(24):
                timestamp = (start_date + datetime.timedelta(days=day, hours=hour)).strftime('%Y-%m-%d %H:%M:%S')
                # Create a new row as a DataFrame
                new_row = pd.DataFrame({'timestamp': [timestamp], 0: [hourly_water_full_month.iloc[day * 24 + hour]]})
                
                # Concatenate the new row with the existing DataFrame
                hourly_demand_water = pd.concat([hourly_demand_water, new_row], ignore_index=True)
        
        # Update the start date to the next month
        start_date += datetime.timedelta(days=days)

    # Rename a column from 'old_name' to 'new_name'
    hourly_demand_water = hourly_demand_water.rename(columns={0: 'X1'})
    
    # Export to a CSV file
    hourly_demand_water.to_csv(os.path.join(path_save, 'hourly_water_demand.csv') , index=False)
     
     
        