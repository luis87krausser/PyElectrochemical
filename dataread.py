#!/usr/bin/env python
# coding: utf-8

# Script to open and graph data from electrochemical 
# based on PyEIS from Kristian B. Knudsen 
# (kknu@berkeley.edu || kristianbknudsen@gmail.com)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)

def correct_text_exp(text_header):
    '''Corrects the text of '*.csv' and '*.txt' 
    files into readable parameters without 
    spaces, ., or /
    '''
    if text_header == 'freq / Hz' or text_header == 'Freq(Hz)':
        return 'f'
    elif text_header == 'neg. Phase / °' or text_header == 'Phase(deg)':
        return 'Z_phase'
    elif text_header == 'Idc / uA':
        return 'I_avg'
    elif text_header == 'Z / Ohm' or text_header == 'Module (Ohms)':
        return 'Z_mag'
    elif text_header == "Z' / Ohm" or text_header == 'ZR(Ohms)':
        return 're'
    elif text_header == "Z'' / Ohm" or text_header == 'Zi(Ohms)':
        return 'im'
    elif text_header == 'Cs / F':
        return 'Cs'
    if text_header == 'V' or text_header == 'WE1 Volts(mV)':
        return 'volt'
    elif text_header == 'µA' or text_header == 'mA' or text_header == 'A':
        return 'current'
    elif text_header == 'WE1 Idir(mA)' or text_header == 'WE1 Idir(uA)':
        return 'current'
    elif text_header == 'WE1 Idir(A)':
        return 'current'
    elif text_header == 's' or text_header == 'Time(sec)':
        return 'time'
    elif text_header == 'T(sec)':
        return 'time'
    else:
        return text_header

"""
Reading functions


Inputs
-----------
    - path: path of datafile(s) as a string
    - file_name: datafile(s) including extension, e.g. ['EIS_data1', 'EIS_data2']
"""


def Teq4CV_expread(path,file_input):
    """Function that return dataframe from *.txt 
    Cyclic Voltammetry data 
    form Teq4 potentiostat/galvanostat"""

    exp_test_header_names = pd.read_csv(path + file_input,
                   sep='\t',
                   usecols=range(0, 3),
                   skipfooter = 1,
                   engine='python',
                   encoding='utf-8')
    names_exp = []
    for j in range(len(exp_test_header_names.columns)):
        names_exp.append(correct_text_exp(exp_test_header_names.columns[j]))

    
    return pd.read_csv(path + file_input, sep = '\t',
                       usecols=range(0, 3),
                       skiprows=int(1), names = names_exp,
                       skipfooter = 1,
                       engine='python',
                       encoding='utf-8')

def Teq4Amp_expread(path,file_input):
    """Function that return dataframe from *.txt 
    amperometrics data 
    form Teq4 potentiostat/galvanostat"""

    exp_test_header_names = pd.read_csv(path + file_input,
                   sep='\t',
                   usecols=range(0, 3),
                   skipfooter = 1,
                   engine='python',
                   encoding='utf-8')
    names_exp = []
    for j in range(len(exp_test_header_names.columns)):
        names_exp.append(correct_text_exp(exp_test_header_names.columns[j]))

    
    return pd.read_csv(path + file_input, sep = '\t',
                       usecols=range(0, 3),
                       skiprows=int(1), names = names_exp,
                       skipfooter = 1,
                       engine='python',
                       encoding='utf-8')


def Teq4EIS_expread(path,file_input):
    """Function that return dataframe from *.txt 
    impedance data from Teq4 
    potentiostat/galvanostat"""

    exp_test_header_names = pd.read_csv(path + file_input,
                   sep='\t',
                   usecols=range(0, 6),
                   skipfooter = 1,
                   engine='python',
                   encoding='utf-8')
    names_exp = []
    for j in range(len(exp_test_header_names.columns)):
        names_exp.append(correct_text_exp(exp_test_header_names.columns[j]))

    
    return pd.read_csv(path + file_input, sep = '\t',
                       usecols=range(0, 6),
                       skiprows=int(1), names = names_exp,
                       skipfooter = 1,
                       engine='python',
                       encoding='utf-8')

def PS4EIS_expread(path,file_input):
    """Function that return a dataframe from cvs impedance data 
    from PalmSens4 potentiostat/galvanostat"""

    exp_test_header_names = pd.read_csv(path + file_input,
                   usecols=range(0, 7),
                   skiprows=int(5),
                   skipfooter = 2,
                   engine='python',
                   encoding='utf_16_le')
    names_exp = []
    for j in range(len(exp_test_header_names.columns)):
        names_exp.append(correct_text_exp(exp_test_header_names.columns[j]))

    
    return pd.read_csv(path + file_input, sep = ',',
                       usecols=range(0, 7),
                       skiprows=int(6), names = names_exp,
                       skipfooter = 2,
                       engine='python',
                       encoding='utf_16_le')


def PS4CV_expread(path, file_name):
    """Function that return a dataframe from cvs cyclic 
    voltammetry data 
    from PalmSens4 potentiostat/galvanostat"""
    exp_test_header_names = pd.read_csv(path + file_name, sep=',', 
                                   skiprows=int(5), 
                                   encoding='utf_16_le') #locates number of skiplines
    names_exp = []
    for j in range(len(exp_test_header_names.columns)):
        names_exp.append(correct_text_exp(exp_test_header_names.columns[j])) #reads coloumn text
    
    return pd.read_csv(path + file_name, sep = ',', 
                       skiprows=int(6), names = names_exp,
                       skipfooter = 2,
                       engine='python',
                       encoding='utf_16_le')


def PS4Amp_expread(path, file_name):
    """Function that return dataframe from *.txt 
    amperometrics data 
    from PalmSens4 potentiostat/galvanostat"""
    exp_test_header_names = pd.read_csv(path + file_name, sep=',', 
                                   skiprows=int(5), 
                                   encoding='utf_16_le') #locates number of skiplines
    names_exp = []
    for j in range(len(exp_test_header_names.columns)):
        names_exp.append(correct_text_exp(exp_test_header_names.columns[j])) #reads coloumn text
    
    
    return pd.read_csv(path + file_name, sep = ',', 
                       skiprows=int(6), names = names_exp,
                       skipfooter = 2,
                       engine='python',
                       encoding='utf_16_le')

"""
Plotting functions


Inputs
-----------
    - df: dataframe(s) including extension
    - g_xlim/g_xlim: Change the x/y-axis limits on plot, if equal to 'none' state auto value.
    - savefig: if not equal to 'none', save the figure in '.eps' format. Otherwise, just show the graphic on the screen.
"""

def CV_plot(df, g_xlim='none', g_ylim='none', savefig = 'none'):
    """Return a cyclic voltamperogram
    from a dataframe containing the 
    'volt' and 'current' variables"""
    plt.figure(dpi=120)
    plt.plot(df['volt'].values, df['current'].values)
    plt.title(r'$\bf Cyclic\; voltamperogram$', fontsize=20)
    plt.xlabel(r'$\bf Volt$', fontsize=20)
    plt.ylabel(r'$\bf Current$', fontsize=20)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')

    plt.show()


def amp_plot(df, g_xlim='none', g_ylim='none', savefig = 'none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and plot an amperometrics register"""
    plt.figure(dpi=120)
    plt.plot(df['time'].values, df['current'].values)
    plt.title(r'\bf Amperometric Plot', fontsize=20)
    plt.xlabel(r'$\bf Time$', fontsize=20)
    plt.ylabel(r'$\bf Current$', fontsize=20)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
            plt.savefig(savefig, format='eps')
            
    plt.show()

def Nyq_plot(df, g_xlim='none', g_ylim='none', savefig='none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and make a Nyquist plot"""
    plt.figure(dpi=120)
    plt.scatter(df['re'].values, df['im'].values)
    plt.title(r'Nyquist Plot', fontsize=20)
    plt.xlabel(r'$\bf Z_{real}\; (\Omega)$', fontsize=20)
    plt.ylabel(r'$\bf Z_{im}\; (\Omega)$', fontsize=20)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')
            
    plt.show()

def Bode_plot(df, g_xlim='none', g_ylim='none', savefig='none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and make a Bode graph"""
    plt.figure(dpi=120)
    plt.scatter(np.log10(df['f'].values), df['im'].values)
    plt.title(r'Bode Plot', fontsize=20)
    plt.xlabel(r'$\bf \log(f)\; (\mathrm{Hz})$', fontsize=20)
    plt.ylabel(r'$\bf Z_{im}\; (\Omega)$', fontsize=20)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')
    plt.show()
