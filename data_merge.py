import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
def xls2csv(xFile, nYear):
    # append_df = pd.DataFrame({'Unnamed: 3': ['{2020}-01-01'], 'Unnamed: 4': ['2020-02-01'], 'Unnamed: 5': ['2020-03-01'], 'Unnamed: 6': ['2020-04-01'], 'Unnamed: 7': ['2020-05-01'], 'Unnamed: 8': ['2020-06-01'], 'Unnamed: 9': ['2020-07-01'], 'Unnamed: 10': ['2020-08-01'], 'Unnamed: 11': ['2020-09-01'], 'Unnamed: 12': ['2020-10-01'], 'Unnamed: 13': ['2020-11-01'], 'Unnamed: 14': ['2020-12-01'], 'Unnamed: 16': ['DATE']})

    append_dict = {}
    for i in range(12):
        if i < 9:
            if nYear >= 2020:
                append_dict['Unnamed: {}'.format(i+3)] = ['{}-0{}-01'.format(nYear, i+1)]
            else:
                append_dict['Unnamed: {}'.format(i+2)] = ['{}-0{}-01'.format(nYear, i+1)]
                
        else:
            if nYear >= 2020:
                append_dict['Unnamed: {}'.format(i+3)] = ['{}-{}-01'.format(nYear, i+1)]
            else:
                append_dict['Unnamed: {}'.format(i+2)] = ['{}-{}-01'.format(nYear, i+1)]
    if nYear >= 2020:
        append_dict['Unnamed: 16'] = ['DATE']
    else:
        append_dict['Unnamed: 15'] = ['DATE']
    append_df = pd.DataFrame(append_dict)
    new_col = []; new_row = []
    for i in range(14):
        new_col.append('{}'.format(i))
    for j in range(12):
        new_row.append('{}'.format(j))
#################Read File###############################
    if nYear < 2020:
        xls_data = pd.read_excel(xFile, index_col=0)
    else:
        xls_data = pd.read_excel(xFile, index_col=1)
    # print(xls_data)
########################################################    
    xls_data.to_csv('temp.csv')
    df = pd.read_csv('temp.csv')
    print(df)
    # df = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 15'], axis = 1)
    if nYear >= 2020:
        df = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 2', 'Unnamed: 15'], axis = 1)
    else:
        df = df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 14'], axis = 1)
    # if nYear == 2018 or nYear == 2020:
    print(df)
    if nYear >= 2020:
        df = df[9:23]
    else:
        df = df[5:18]
    print(df)
    df = df._append(append_df)
    print("---------------df---------------\n", df)
    temp_cols = df.columns.tolist()
    # print(temp_cols)
    new_cols = temp_cols[-1:] + temp_cols[:-1]
    # print(new_cols)
    df = df[new_cols]
    df = df.iloc[::-1]
    # print("---------------df---------------\n", df)
    # print(df.iloc[0])
    # print(df)
    # print(df.T)
    dfT =df.T
    # print(dfT)
    # print(dfT.iloc[0])
    dfT = dfT.set_axis(list(dfT.iloc[0]), axis = 1)
    print("--------------dfT------------\n", dfT)
    if nYear >= 2020:
        dfT = dfT.drop(['Unnamed: 16'], axis = 0)
    else:
        dfT = dfT.drop(['Unnamed: 15'], axis = 0)
    print("--------------dfT------------\n", dfT)
    dfT = dfT.set_axis(new_row, axis = 0)
    #add finance column
    dfT_shape = dfT.shape
    # print(dfT)
    if dfT_shape[1] == 14:
        ###append finance########
        finance = [100] * 12
        dfT['Insurance and financial services'] = finance

        temp_cols = dfT.columns.tolist()
        new_cols = temp_cols[0:2] + temp_cols[-1:] + temp_cols[2:-1]
        dfT = dfT[new_cols]
        print(dfT)
    dfT.rename(columns = {'Personal care, social protection and miscellaneous goods and services' : 'Miscellaneous goods and services', 'Restaurants and accommodation services' : 'Restaurants and hotels', 'Recreation, sport and culture' : 'Recreation and culture', 'Information and communication' : 'Communication'}, inplace = True)

    return dfT
df18 = xls2csv("Consumer Price Index 2018.xls", 2018)
df19 = xls2csv("Consumer Price Index 2018.xls", 2019)
df20 = xls2csv("Consumer Price Index 2020.xls", 2020)
df21 = xls2csv("Consumer Price Index 2021.xls", 2021)
df22 = xls2csv("Consumer Price Index 2022.xls", 2022)

##########################Conver base-2021 to base-2014 for 2022#####################
print(df22.head())
for i in range(12):
    for j in range(1, 15):
        df22.iloc[i, j] = float(df22.iloc[i, j]) * float(df21.iloc[i, j]) / 100
        
# print(df22)
#####################################################################################
df = df18
df = df._append(df19, ignore_index = True)
df = df._append(df20, ignore_index = True)
df = df._append(df21, ignore_index = True)
df = df._append(df22, ignore_index = True)
# print("df20\n", df20)
# print("df21\n", df21)
# print("df22\n", df22)
print(df)
###################################delete temp file####################################
if os.path.exists("temp.csv"):
  os.remove("temp.csv")
#######################################################################################
df.to_csv("CPI.csv")

# dff = pd.read_csv('CPI.csv', index_col=0)
dff = pd.read_csv('CPI.csv')
dff = dff.loc[:, ~dff.columns.str.contains('^Unnamed')]
# print(dff)

