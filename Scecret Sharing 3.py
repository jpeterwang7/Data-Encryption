# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from phe import paillier
import random
import matplotlib.pyplot as plt
import time as timepackage

# Reading df with pandas
path_to_file = 'household_power_consumption (2).txt'
df = pd.read_csv(path_to_file, sep=";", header=None)
df.columns = ["Date", "Time", "Global_active_power", " Global_reactive_power", "Voltage", "Global Intensity",
              "Sub Metering 1", "Sub Metering 2", "Sub Metering 3"]

# Separating time column into np array
time = df.loc[1:, "Time"]
time_vector = np.array(time)
time = list(time_vector)
# print("Time Stamps: ", time)

# creating np array for date for the price vector to reference
date = df.loc[1:, "Date"]
date_vector = np.array(date)

# unit price vector
price = []
for i in date_vector:
    datetime_object = datetime.strptime(i, '%d/%m/%Y')
    month = datetime_object.strftime("%m")
    half_year = month == '01' or month == '02' or month == '03' or month == '04' or month == '05' or month == '06'
    year = datetime_object.strftime("%Y")
    if year == '2006':
        price.append(0.095)
    if year == '2007':
        if half_year == True:
            price.append(0.0924)
        else:
            price.append(0.0921)
    if year == '2008':
        if half_year == True:
            price.append(0.0914)
        else:
            price.append(0.0910)
    if year == '2009':
        if half_year == True:
            price.append(0.0914)
        else:
            price.append(0.0910)
    if year == '2010':
        if half_year == True:
            price.append(0.0940)
        else:
            price.append(0.0995)
price = np.array(price)
price = list(price)
# print("Price: ", price)

# 3 submetering data columns
df.drop(columns=["Date", "Time", "Global_active_power", " Global_reactive_power", "Voltage", "Global Intensity"],
        inplace=True)
df.drop(index=0, inplace=True)
data_in_numpy = np.array(df)

tlen = len(time_vector)  # length of time vector

alice_total = []
bob_total = []
charles_total = []
for i in range(tlen):
    alice_reading = data_in_numpy[i, 0]
    alice_total.append(alice_reading)
    bob_reading = data_in_numpy[i, 1]
    bob_total.append(bob_reading)
    charles_reading = data_in_numpy[i, 2]
    charles_total.append(charles_reading)

# print(alice_total)
# print(bob_total)
# print(charles_total)

# TODO: Let submetering1, submetering2, submetering3 be COSTS (=reading*price) instead of READINGS

def valid_number(x):
    # check whether the input number is a valid floating point number
    is_valid = False
    try:
        y = float(x)
    except ValueError:
        return is_valid
    if np.isnan(float(x)) or np.isinf(float(x)): # rule out NaNs or infinities
        return is_valid
    else:
        is_valid = True
        return is_valid

submetering1 = alice_total
submetering1 = [(float(0) if not valid_number(x) else float(x)) for x in submetering1]
submetering2 = bob_total
submetering2 = [(float(0) if not valid_number(x) else float(x)) for x in submetering2]
submetering3 = charles_total
submetering3 = [(float(0) if not valid_number(x) else float(x)) for x in submetering3]
submeter1=[]
submeter2=[]
submeter3=[]
for x in submetering1:
    submeter1.append(submetering1[int(x)]*price[int(x)])
    submeter2.append(submetering2[int(x)]*price[int(x)])
    submeter3.append(submetering3[int(x)]*price[int(x)])
submetering1=submeter1
submetering2=submeter2
submetering3=submeter3

# assuming submetering1, submetering2, submetering3 are COSTS

# split alice's costs into 3
submetering1 = [int(100*x) for x in submetering1] # convert costs to [whatever (1/100) a Euro is] and then to int
alice_mix = np.zeros(shape=(tlen,3))
for i in range(len(submetering1)):
    x = submetering1[i]
    number1 = random.randint(0, x) # take part of x randomly
    x = x - number1 # update x
    number2 = random.randint(0, x) # take another part of x randomly
    x = x - number2 # update x
    number3 = x # set number3 to be remaining part of x
    # Now, we set values in the alice_mix matrix
    alice_mix[i,0] = number1
    alice_mix[i,1] = number2
    alice_mix[i,2] = number3
alice_mix = alice_mix.astype(int) # convert to int


# split bob into 3
submetering2 = [int(100*x) for x in submetering2] # convert costs to cents and then to int
bob_mix = np.zeros(shape=(tlen,3))
for i in range(len(submetering2)):
    x = submetering2[i]
    number1 = random.randint(0, x) # take part of x randomly
    x = x - number1 # update x
    number2 = random.randint(0, x) # take another part of x randomly
    x = x - number2 # update x
    number3 = x # set number3 to be remaining part of x
    # Now, we set values in the alice_mix matrix
    bob_mix[i,0] = number1
    bob_mix[i,1] = number2
    bob_mix[i,2] = number3
bob_mix = bob_mix.astype(int) # convert to int


# split charles into 3
submetering3 = [int(100*x) for x in submetering3] # convert costs to cents and then to int
charles_mix = np.zeros(shape=(tlen,3))
for i in range(len(submetering3)):
    x = submetering3[i]
    number1 = random.randint(0, x) # take part of x randomly
    x = x - number1 # update x
    number2 = random.randint(0, x) # take another part of x randomly
    x = x - number2 # update x
    number3 = x # set number3 to be remaining part of x
    # Now, we set values in the alice_mix matrix
    charles_mix[i,0] = number1
    charles_mix[i,1] = number2
    charles_mix[i,2] = number3
charles_mix = charles_mix.astype(int) # convert to int

# generate key pairs
public_a, private_a = paillier.generate_paillier_keypair(n_length=64)
public_b, private_b = paillier.generate_paillier_keypair(n_length=64)
public_c, private_c = paillier.generate_paillier_keypair(n_length=64)

# noisy list
alice_noisy=np.zeros(shape=(tlen,))
bob_noisy=np.zeros(shape=(tlen,))
charles_noisy=np.zeros(shape=(tlen,))

# utility_secret_for_alice=0
# utility_secret_for_bob=0
# utility_secret_for_charles=0
# a=0
# b=1
# c=2
# utilitys_secret_for_alice=0
# utilitys_secret_for_bob=0
# utilitys_secret_for_charles=0
# for x in alice_mix:
t = timepackage.process_time()
for i in range(tlen):
    if i % 10000 == 0:
        print('percent done:'+str(i/tlen*100)+'\n')
    e_a1 = public_a.encrypt(int(alice_mix[i,0]))
    e_a2 = public_b.encrypt(int(alice_mix[i,1]))
    e_a3 = public_c.encrypt(int(alice_mix[i,2]))
    e_b1 = public_a.encrypt(int(bob_mix[i,0]))
    e_b2 = public_b.encrypt(int(bob_mix[i,1]))
    e_b3 = public_c.encrypt(int(bob_mix[i,2]))
    e_c1 = public_a.encrypt(int(charles_mix[i,0]))
    e_c2 = public_b.encrypt(int(charles_mix[i,1]))
    e_c3 = public_c.encrypt(int(charles_mix[i,2]))
    # alice_noisy.append (e_a1)
    # bob_noisy.append(e_a2)
    # charles_noisy.append(e_a3)
    # alice_noisy.append(e_b1)
    # bob_noisy.append(e_b2)
    # charles_noisy.append(e_b3)
    # alice_noisy.append(e_c1)
    # bob_noisy.append(e_c2)
    # charles_noisy.append(e_c3)
    alice_noisy[i] = private_a.decrypt(e_a1 + e_b1 + e_c1)
    bob_noisy[i] = private_b.decrypt(e_a2 + e_b2 + e_c2)
    charles_noisy[i] = private_c.decrypt(e_a3 + e_b3 + e_c3)
    # code whose time is to be measured
    # utilitys_secret_for_alice = e_a1 + e_b1 + e_c1 + utilitys_secret_for_alice
    # utilitys_secret_for_bob = e_a2 + e_b2 + e_c2 + utilitys_secret_for_bob
    # utilitys_secret_for_charles = e_a3 + e_b3 + e_c3 +utilitys_secret_for_charles
    # a = a + 3
    # b = b + 3
    # c = c + 3
elapsed_time = timepackage.process_time() - t
print("Elapsed time is " + str(elapsed_time) + " seconds")
def xtick():
    count = 0
    loop = 1
    horizontal_ticks = []
    horizontal_ticks.append(time[0])
    for x in time:
        count = count + 1
        if count == 1000:
            count = 0
            horizontal_ticks.append(time[1000 * loop])
            loop = loop + 1
    ax1.set_xticks([0, 1000, 2000, 3000], horizontal_ticks)
    ax2.set_xticks([0, 1000, 2000, 3000], horizontal_ticks)

fig1,ax = plt.subplots(nrows=2,ncols=1)
ax1 = ax[0]
ax1.plot(submetering1,alpha=0.6,label='alice orig')
ax1.plot(submetering2,alpha=0.6,label='bob orig')
ax1.plot(submetering3,alpha=0.6,label='charles orig')
ax1.legend()
ax2 = ax[1]
ax2.plot(alice_noisy,alpha=0.6,label='alice noisy')
ax2.plot(bob_noisy,alpha=0.6,label='bob noisy')
ax2.plot(charles_noisy,alpha=0.6,label='charles noisy')
ax2.legend()
xtick()
plt.show()

orig_total = np.array(submetering1)+np.array(submetering2)+np.array(submetering3)
noisy_total = np.array(alice_noisy)+np.array(bob_noisy)+np.array(charles_noisy)

fig2,ax = plt.subplots(nrows=2,ncols=1)

ax1 = ax[0]
ax1.plot(orig_total,alpha=0.6,label='orig total')
ax1.legend()
xtick()
ax2 = ax[1]
ax2.plot(noisy_total,alpha=0.6,label='noisy total')
ax2.legend()
xtick()
plt.show()

# TODO: ADD datetime element to x-axis
# TODO: Use only 3000 data points (rows) from the dataset instead of full dataset
# TODO: Get figures



# alice sends e_a1, e_a2, e_a3 to utility
# bob sends e_b1, e_b2, e_b3 to utility
# charles sends e_c3, e_c3, e_c3 to utility
# utility calculates a 'fudged' value for alice to decrypt through homomorphic encrpytion
# remember, at  this point, although utility can add values due to HE, it can't read them
alice_decrypts = private_a.decrypt(utilitys_secret_for_alice)
alice_reports_to_util = alice_decrypts
print(alice_reports_to_util)
bob_decrypts = private_b.decrypt(utilitys_secret_for_bob)
bob_reports_to_util = bob_decrypts
print(bob_reports_to_util)
charles_decrypts = private_c.decrypt(utilitys_secret_for_charles)
charles_reports_to_util = charles_decrypts
print(charles_reports_to_util)

# IMPORTANT: sum of 'real' values = sum of fudged values
# ^ do the same for bob and charles
for i in range(len(time)):
    str=time[i]
    time[i]= str[:-3]
# Set up the figure the way we want it to look
print(alice_noisy)
a_noise=[]
b_noise=[]
c_noise=[]
for x in alice_noisy:
    a=private_a.decrypt(alice_noisy[x])
    a_noise.append(a)
print(a_noise)
plt.figure()
time = np.linspace(0, 1, len(submetering1))
plt.plot(time, submetering1, label="Alice Real Reading")
plt.plot(time, submetering2, label="Bob Real Reading")
plt.plot(time, submetering3, label="Charles Real Reading")
plt.plot(time, a_noise, label="Alice Noisy Reading")
#plt.plot(np.array(time), np.array(b_noise), label="Bob Noisy Reading")
#plt.plot(np.array(time), np.array(c_noise), label="Charles Noisy Reading")
plt.legend(loc="lower right")
plt.xlabel('Time')
plt.ylabel('Power Recorded')

plt.title('Secret Sharing')
plt.show()