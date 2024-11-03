"""
 Important to note that python's float type has the same precision as the C++
 double.
"""
import sys  # isort:skip

sys.path.append("../pydp")  # isort:skip

# stdlib
import os
from pathlib import Path

# pydp absolute
import pydp as dp
from pydp.algorithms.laplacian import BoundedSum

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time as timepackage

#Reading df with pandas
path_to_file = 'household_power_consumption'
df = pd.read_csv(path_to_file, sep= ";", header =None)
df.columns= ["Date", "Time","Global_active_power"," Global_reactive_power", "Voltage", "Global Intensity","Sub Metering 1", "Sub Metering 2", "Sub Metering 3"]

#Separating time column into np array
time = df.loc[1:,"Time"]
time_vector = np.array(time)
#print("Time Stamps: ", time_vector)

#creating np array for date for the price vector to reference
date = df.loc[1:, "Date"]
date_vector = np.array(date)
df = df.apply(pd.to_numeric, errors ='coerce')

#unit price vector
price =[]
for i in date_vector:
    datetime_object = datetime.strptime(i,'%d/%m/%Y')
    month = datetime_object.strftime("%m")
    half_year = month == '01' or month == '02' or month == '03' or month == '04'or month == '05' or month == '06'
    year = datetime_object.strftime("%Y")
    if year == '2006':
        price.append(0.095)
    elif year =='2007':
        if half_year == True:
            price.append(0.0924)
        else:
            price.append(0.0921)
    elif year == '2008':
        if half_year == True:
            price.append(0.0914)
        else:
            price.append(0.0910)
    elif year == '2009':
        if half_year == True:
            price.append(0.0914)
        else:
            price.append(0.0910)
    elif year == '2010':
        if half_year == True:
            price.append(0.0940)
        else:
            price.append( 0.0995)
price = np.array(price)
#print ("Price: ", price)

#3 submetering data columns
df.drop(columns = ["Date","Time","Global_active_power"," Global_reactive_power", "Voltage", "Global Intensity"], inplace = True)
df.drop(index=0, inplace = True)
data_in_numpy = np.array(df)

# Multiplies price with consumption value
for i in range(len(df)):
    df.loc[i + 1, "Sub Metering 1"] = (df.loc[i + 1, "Sub Metering 1"] * price[i])
    df.loc[i+1, "Sub Metering 2"] = (df.loc[i+1, "Sub Metering 2"] * price[i])
    df.loc[i + 1, "Sub Metering 3"] = (df.loc[i + 1, "Sub Metering 3"] * price[i])
    '''
for i in range(len(df)):
    print(df.loc[i+1, "Sub Metering 1"], df.loc[i+1, "Sub Metering 2"], df.loc[
        i+1,"Sub Metering 3"])
'''
# Creating a class
class Submetering:

    # Constructor
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self._epsilon = epsilon
        self._privacy_budget = float(1)
        self._df = df

    # Function to return real metering sum in dataset.
    '''
    def sum_metering(self) -> float:
        self._df = df.sum(axis = 1)
        print("\nSumming Rows\n", self._df)
'''
    # Function to return the DP sum of metering data.
    def private_sum(self, privacy_budget: float, row) -> int:
        x = BoundedSum(
            epsilon=privacy_budget,
            lower_bound=0,
            upper_bound= 100,
            dtype="float",
        )
        return x.quick_result(list(row))

# get absolute path
c = Submetering(1)
path = Path(os.path.dirname(os.path.abspath(__file__)))

#Print real sum
'''
print("Sum:\t\n" + str(df.sum(axis = 1)))
sum = [(df.sum(axis = 1))]
print(sum)
'''

#Prints private sum
private_sum_2000 = []
private_sum_500 = []
private_sum_300 = []
real_sum = []
row_list = df.loc[1, "Sub Metering 1"], df.loc[1, "Sub Metering 2"],df.loc[1, "Sub Metering 3"]
for j in range(len(df)):
    row_list = df.loc[j + 1, "Sub Metering 1"], df.loc[j + 1, "Sub Metering " \
                                                              "2"],df.loc[j +
                                                                          1, "Sub Metering 3"]
    #print("Private Sum:\t" + str(c.private_sum(2000, row_list)))
    t = timepackage.process_time_ns()
    private_sum_2000.append(str(c.private_sum(2000, row_list)))
    private_sum_2000[j] = float(private_sum_2000[j])
    print(timepackage.process_time_ns() - t)
    private_sum_500.append(str(c.private_sum(500, row_list)))
    private_sum_500[j] = float(private_sum_500[j])
    private_sum_300.append(str(c.private_sum(300, row_list)))
    private_sum_300[j] = float(private_sum_300[j])
    real_sum.append(df.loc[j + 1, "Sub Metering 1"] + df.loc[j + 1,
                                                           "Sub Metering "
                                                               "2"] + df.loc[j +
                                                                          1, "Sub Metering 3"])
'''
# Printing arrays
print("Real Sum: ", real_sum)
print("Private Sum 2000: ", private_sum_2000)
print("Private Sum 500: ", private_sum_500)
print("Private Sum 100: ", private_sum_300)
'''

# Ticks
y_ticks = np.linspace(min(private_sum_2000), max(private_sum_2000),
                      num = 7)
plt.yticks(y_ticks)
plt.xticks(np.arange(0, len(time)+1, 65))

# Plot
plt.plot(np.array(time), private_sum_300, label = "Private Sum, Privacy "
                                                  "Budget = 300")
plt.plot(np.array(time), private_sum_500, label = "Private Sum, Privacy "
                                                  "Budget = 500")
plt.plot(np.array(time), private_sum_2000, label = "Private Sum, Privacy "
                                                   "Budget = 2000")
plt.plot(np.array(time), real_sum, label = "Real Sum")
plt.legend()

# Plot labels
plt.ylabel('Price')
plt.xlabel('Datetime')
plt.title("Differential Privacy")
plt.show()



