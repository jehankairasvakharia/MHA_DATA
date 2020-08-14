"""HI"""
# might need to do
#!pip install IP2Location

#imports
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import IP2Location


#set up pandas DataFrame with values from csv file
#the Qualtrics data uses the first TWO rows as headers so we've denoted that using the header= tag
data = pd.read_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/July_2020_MHA_Pilot.csv', header=[0,1])



#check to see if pandas is working and the csv has correctly imported:
print("1. HEADER CHECK:")
print(data.head(10))
print('\n\n')
#check to see size of data
print("\n2. SIZE CHECK: (rows, columns)")
print(data.shape)

# Now this is where stuff gets weird!

# Lets first copy the data into three dataframes for Design1 Design2 and Design3

design1 = data.copy(deep=True)
design2 = data.copy(deep=True)
design3 = data.copy(deep=True)

# Design Shapes
print("3. design shapes check: (rows,columns)")
print(design1.shape)
print(design2.shape)
print(design3.shape)



# Okay cool now lets start TRIMMING each of these! We want to remove all the records that have Thompson Sampling
# as the policy for each of the designs
ur1 = design1[  design1[('policy1','policy1')]   == 'uniform_random']
ur2 = design2[  design2[('policy21','policy21')]   == 'uniform_random']
ur3 = design3[  design3[('policy_3_1','policy_3_1')]   == 'uniform_random']

# now lets check to see how many records we have for Uniform Random for each of the 3 designs
print('4. UniformRandom-Only shapes check: (rows, columns)')
print(ur1.shape)
print(ur2.shape)
print(ur3.shape)


# Remove rows whose worker ID is in the invalid list
datasets = [ur1, ur2, ur3]
invalid = ['sd', 'test', 'kk', 'TEST', '1234']

new_data = []
for ur in datasets:
    ur = ur[~ur.Worker_ID.iloc[:,0].isin(invalid)]
    new_data.append(ur)
datasets = new_data
ur1, ur2, ur3 = datasets[0], datasets[1], datasets[2]

print("5. Shape check after removing invalid workers")
for ur in datasets:
  print(ur.shape)

# Keep the first instance of each worker ID, IP address
new_data = []
for ur in datasets:
    ur = ur.drop_duplicates(subset=[('V6', 'IPAddress')])
    ur = ur.drop_duplicates(subset=[('Worker_ID', 'If you consent to participate in the study, please enter your Worker ID: ')])
    new_data.append(ur)
datasets = new_data
ur1, ur2, ur3 = datasets[0], datasets[1], datasets[2]

print("6. Shape check afte removing duplicate workers, IP addresses")
for ur in datasets:
  print(ur.shape)

database = IP2Location.IP2Location()

#will upload this bin file to Bandit Deployments/Data Analysis folder on google drive! Pls find there and import into this (lets also figure out how to streamline this for later)
database.open("IP2LOCATION-LITE-DB3.BIN")

new_data = []
for ur in datasets:
    # Get countries corresponding to the IP addresses
    ip_list = []
    for ip in ur[("V6", "IPAddress")]:
        rec = database.get_all(ip)
        ip_country = rec.country_long
        ip_list.append(ip_country)

    # Filter out non-US IP addresses
    ur[("IPCountry", "country")] = ip_list
    ur = ur[ur[("IPCountry", "country")] == "United States of America"]
    new_data.append(ur)
datasets = new_data
ur1, ur2, ur3 = datasets[0], datasets[1], datasets[2]

print("7.Shape check after removing non-US submissions.")
for ur in datasets:
  print(ur.shape)

# Check whether there are any participants not passing the topic check in ur1/ur2/ur3
print(ur1['study_about'].iloc[:,0].value_counts())
print(ur2['study_about'].iloc[:,0].value_counts())
print(ur3['study_about'].iloc[:,0].value_counts())

# Remove all rows where participants who did not pass the topic check
ur1 = ur1[  ur1['study_about'].iloc[:,0]   == 1.0]
ur2 = ur2[  ur2['study_about'].iloc[:,0]   == 1.0]
ur3 = ur3[  ur3['study_about'].iloc[:,0]   == 1.0]

# Check how many records we have after removing participants who asked to be removed
print('8. Participants-Passed-Topic-Check-Only shapes check: (rows, columns)')
print(ur1.shape)
print(ur2.shape)
print(ur3.shape)

# Check whether there are any participants asked to be removed in ur1/ur2/ur3
print(ur1['exclude'].iloc[:,0].value_counts())
print(ur2['exclude'].iloc[:,0].value_counts())
print(ur3['exclude'].iloc[:,0].value_counts())

# Remove all rows where participants asked to be excluded
ur1 = ur1[  ur1['exclude'].iloc[:,0]   == 1.0]
ur2 = ur2[  ur2['exclude'].iloc[:,0]   == 1.0]
ur3 = ur3[  ur3['exclude'].iloc[:,0]   == 1.0]

# Check how many records we have after removing participants who asked to be removed
print('9. Included-Participants-Only shapes check: (rows, columns)')
print(ur1.shape)
print(ur2.shape)
print(ur3.shape)

# Remove rows with all elements empty
ur1 = ur1.dropna(how='all')
ur2 = ur2.dropna(how='all')
ur3 = ur3.dropna(how='all')

# Check how many records we have after removing empty rows
print('10. Shapes check after removing empty rows: (rows, columns)')
print(ur1.shape)
print(ur2.shape)
print(ur3.shape)


# Add unique ID starting from 1
ID = np.arange(1, 1+len(ur1))
ur1.insert (0, "ID", ID)
ur2.insert (0, "ID", np.arange(1, 1+len(ur2)))
ur3.insert (0, "ID", np.arange(1, 1+len(ur3)))


# Remove identifying info, i.e., IPAddress, ResponseID and Worker_ID by setting them to NA.
import numpy as np
ur1['V6', 'IPAddress'] = np.nan
ur1['V1', 'ResponseID'] = np.nan
ur1['Worker_ID', 'If you consent to participate in the study, please enter your Worker ID: '] = np.nan

ur2['V6', 'IPAddress'] = np.nan
ur2['V1', 'ResponseID'] = np.nan
ur2['Worker_ID', 'If you consent to participate in the study, please enter your Worker ID: '] = np.nan

ur3['V6', 'IPAddress'] = np.nan
ur3['V1', 'ResponseID'] = np.nan
ur3['Worker_ID', 'If you consent to participate in the study, please enter your Worker ID: '] = np.nan

# Overview of how ur1 looks like after adding ID and removing identifying info
ur1.head()

#now to export all this wonderful data to three separate CSVs
ur1.to_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/design_1_clean.csv')
ur2.to_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/design_2_clean.csv')
ur3.to_csv('C:/Users/Public/PYTHON_PROJECTS/MHA_data/CSVBOIS/design_3_clean.csv')