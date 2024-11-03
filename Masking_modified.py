# -*- coding: utf-8 -*-
# Encryption Using Masking Method

import numpy as np
from numpy import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes # added brokenaxes package - Shourya
import math
# import time


#Reading df with pandas
# !!! Important: make sure you run the codes with household_power_consumption_no_q.txt !!!
path_to_file = 'household_power_consumption.txt'
df = pd.read_csv(path_to_file, sep= ";", header =None)
df.columns  = ["Date", "Time","Global_active_power"," Global_reactive_power", "Voltage", "Global Intensity","Sub Metering 1", "Sub Metering 2", "Sub Metering 3"]

#Separating time column into np array
time = df.loc[1:,"Time"]
time_vector = np.array(time)

#creating np array for date for the price vector to reference
date = df.loc[1:, "Date"]
date_vector = np.array(date)


#3 submetering data columns
df.drop(columns = ["Date","Time","Global_active_power"," Global_reactive_power", "Voltage", "Global Intensity"], inplace = True)
df.drop(index=0, inplace = True)
data_in_numpy = np.array(df)

tlen = len(time_vector) # length of time vector


alice_measurement_str = np.array(data_in_numpy[range(0, 2049280), 0])
bob_measurement_str = np.array(data_in_numpy[range(0, 2049280), 1])
charles_measurement_str = np.array(data_in_numpy[range(0, 2049280), 2])


alice_measurement = alice_measurement_str.astype(np.float)
bob_measurement = bob_measurement_str.astype(np.float)
charles_measurement = charles_measurement_str.astype(np.float)
time_vec = np.array([i for i in range(len(alice_measurement))])

# constants used for testing the program
# AB_key = 10
# AC_key = 20
# BC_key = 30
bound = 10**6
AU_key = 123456789
BU_key = 187654329
CU_key = 198765432
# alice_measurement = np.array([1000, 2000, 3000, 4000, 5000])
# bob_measurement = np.array([6000, 7000, 8000, 9000, 10000])
# charles_measurement = np.array([1230, 2120, 3150, 4890, 5120])
large_number_1 = 5*1000+123
large_number_2 = 6*1000+321
large_number_3 = 7*1000+213


# pseudorandom generator
def pseudorandom_generator(seed_value, upper_bound):
    np.random.seed(seed_value)
    num = np.random.randint(0, upper_bound)
    return num


'''
# generate random for AB, AC, BC
AB_pseudorandom = pseudorandom_generator(AB_key, bound)
AC_pseudorandom = pseudorandom_generator(AC_key, bound)
BC_pseudorandom = pseudorandom_generator(BC_key, bound)
'''


def hms_to_s(time_lst):
    time_stamp_lst = []
    for i in time_lst:
        h, m, s = i.split(':')
        time_stamp = int(h) * 3600 + int(m) * 60 + int(s)
        time_stamp_lst.append(time_stamp)
    return time_stamp_lst

'''
for i in time_vector:
    time_vector_test = []
    for q in range(20):
        time_vector_test.append(i)
    else:
        break
'''

mask_lst = []
time_stamp_lst = hms_to_s(time_vector)
for i in time_stamp_lst:
    mask_lst.append(pseudorandom_generator(i, bound))


# for i in time_vector:
#     mask_lst.append(pseudorandom_generator(hms_to_s(i), bound))

mask = np.array(mask_lst)

# masking
alice_masked_measurement = alice_measurement + mask
bob_masked_measurement = bob_measurement + mask
charles_masked_measurement = charles_measurement + mask

#for i in alice_masked_measurement:
#    print('the masked measurement is', i)



# encryption function
def encryption_function(m, k, n):
    sum_list = m+k
    encrypted_data = sum_list % n
    return encrypted_data
# implement this (m+n)%k


alice_encrypted = encryption_function(alice_masked_measurement, AU_key, large_number_1)
bob_encrypted = encryption_function(bob_masked_measurement, BU_key, large_number_2)
charles_encrypted = encryption_function(charles_masked_measurement, CU_key, large_number_3)


# for i in alice_encrypted:
#     print('the encrypted value is', i)

x_a = alice_measurement
x_b = bob_measurement
x_c = charles_measurement
y_a = alice_encrypted
y_b = bob_encrypted
y_c = charles_encrypted


# I made some edits below this line to 'break' the y-axis so that our plots appear cleaner
# I am using brokenaxes package for it - Shourya

time_vec = np.array([i for i in range(len(x_a))])

'''
for i in time_stamp_lst:
    time_stamp_lst_test = []
    for q in range(20):
        time_stamp_lst_test.append(i)
'''

time_vector_lst = list(time_vector)
colon = ':'
for idx, ele in enumerate(time_vector_lst):
    time_vector_lst[idx] = ele.replace(colon,'')
# time_vector_stripped = time_vector_lst.replace(':','')
time_vector_int_lst = [int(i) for i in time_vector_lst]
time_vector_int = np.array(time_vector_int_lst)


fig1 = plt.figure(1)
bax = brokenaxes(xlims = ((0,0),(0,100)), ylims=((0,100),(100,np.max(y_a))))
bax.plot(time_vector,x_a, label='real')
bax.plot(time_vector,y_a, label='encrypted')
plt.title('Alice: Masking', fontsize=20)
bax.set_xlabel("Timestamp", fontsize=18)
bax.set_ylabel("Consumption", fontsize=18)
bax.legend(loc='upper left')
plt.show()


fig2 = plt.figure(2)
bax = brokenaxes(xlims =  ((0,0),(0,100)), ylims=((0,100),(100,np.max(y_b))))
bax.plot(time_vector,x_b, label='real')
bax.plot(time_vector,y_b, label='encrypted')
plt.title('Bob: Masking', fontsize=20)
bax.set_xlabel("Timestamp", fontsize=18)
bax.set_ylabel("Consumption", fontsize=18)
bax.legend(loc='upper left')
plt.show()

fig3 = plt.figure(3)
bax = brokenaxes(xlims =  ((0,0),(0,100)), ylims=((0,100),(100,np.max(y_c))))
bax.plot(time_vector,x_c, label='real')
bax.plot(time_vector,y_c, label='encrypted')
plt.title('Charles: Masking', fontsize=20)
bax.set_xlabel("Timestamp", fontsize=18)
bax.set_ylabel("Consumption", fontsize=18)
bax.legend(loc='upper left')
plt.show()

