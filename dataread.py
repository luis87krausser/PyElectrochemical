#!/usr/bin/env python
# coding: utf-8

# Script to open and graph data from electrochemical 
# based on PyEIS from Kristian B. Knudsen 
# (kknu@berkeley.edu || kristianbknudsen@gmail.com)

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

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
    from Teq4 potentiostat/galvanostat"""

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
    from Teq4 potentiostat/galvanostat"""

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
    - g_title: Name of the graph. For latex code use r'$\bf My\,Title$'
    - g_xlim/g_xlim: Change the x/y-axis limits on plot, if equal to 'none' state auto value.
    - g_xlabel/g_ylabel: Name of the x/y-axis. For latex code use r'$\bf my\,units (unit)$'
    - savefig: if not equal to 'none', save the figure in '.eps' format. Otherwise, just show the graphic on the screen.
"""

def CV_plot(df, g_title = 'none', g_xlabel = 'none', g_ylabel = 'none', 
             g_xlim='none', g_ylim='none', savefig = 'none'):
    """Return a cyclic voltamperogram
    from a dataframe containing the 
    'volt' and 'current' variables"""
    plt.figure(dpi=120)
    plt.plot(df['volt'].values, df['current'].values)
    if g_title == 'none':
        plt.title(r'$\bf Cyclic\; voltamperogram$', fontsize=20)
    elif g_title != 'none':
        plt.title(g_title, fontsize=20)
    if g_xlabel == 'none':
        plt.xlabel(r'$\bf Volt$', fontsize=18)
    elif g_xlabel != 'none':
        plt.xlabel(g_xlabel, fontsize=18)
    if g_ylabel == 'none':
        plt.ylabel(r'$\bf Current$', fontsize=18)
    elif g_ylabel != 'none':
        plt.ylabel(g_ylabel, fontsize=18)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')

    plt.show()


def amp_plot(df, g_title = 'none', g_xlabel = 'none', g_ylabel = 'none',
             g_xlim='none', g_ylim='none', savefig = 'none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and plot an amperometrics register"""
    plt.figure(dpi=120)
    plt.plot(df['time'].values, df['current'].values)
    if g_title == 'none':
        plt.title(r'\bf Amperometric Plot', fontsize=20)
    elif g_title != 'none':
        plt.title(g_title, fontsize=20)
    if g_xlabel == 'none':
        plt.xlabel(r'$\bf Time$', fontsize=18)
    elif g_xlabel != 'none':
        plt.xlabel(g_xlabel, fontsize = 18)
    if g_ylabel == 'none':
        plt.ylabel(r'$\bf Current$', fontsize=18)
    elif g_ylabel != 'none':
        plt.ylabel(g_ylabel, fontsize=18)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
            plt.savefig(savefig, format='eps')
            
    plt.show()

def Nyq_plot(df, g_title = 'none', g_xlim='none', g_ylim='none', savefig='none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and make a Nyquist plot"""
    plt.figure(dpi=120)
    plt.scatter(df['re'].values, df['im'].values)
    plt.xlabel(r'$\bf Z_{real}\; (\Omega)$', fontsize=20)
    plt.ylabel(r'$\bf Z_{im}\; (\Omega)$', fontsize=20)
    if g_title == 'none':
        plt.title(r'\bf Nyquist Plot', fontsize=20)
    elif g_title != 'none':
        plt.title(g_title, fontsize=20)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')
            
    plt.show()

def Bode_plot(df, g_title = 'none', g_xlim='none', g_ylim='none', savefig='none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and make a Bode graph
    log(f) vs Z_im"""
    plt.figure(dpi=120)
    plt.scatter(np.log10(df['f'].values), df['im'].values)
    plt.xlabel(r'$\bf \log(f)\; (\mathrm{Hz})$', fontsize=20)
    plt.ylabel(r'$\bf Z_{im}\; (\Omega)$', fontsize=20)
    if g_title == 'none':
        plt.title(r'\bf Bode Plot', fontsize=20)
    elif g_title != 'none':
        plt.title(g_title, fontsize=20)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')
    plt.show()

def bookBode_plot(df, g_title = 'none', g_xlim='none', g_ylim='none', savefig='none'):
    """Take a pandas dataframe from a *.txt 
    of Teq4 and make a Bode graph
    according to book style
    Zmodule vs log(f)"""
    fig = plt.figure(dpi=120)
    ax1 = fig.add_subplot(111)
    ax1.plot(np.log10(df['f'].values), df['Z_mag'].values,'bo')
    ax1.set_ylabel(r'$\bf \mid Z \mid\; [\Omega]$', color='b',fontsize=20)
    ax1.set_xlabel(r'$\bf \log(f)\; (\mathrm{Hz})$', fontsize=20)

    ax2 = ax1.twinx()
    ax2.plot(np.log10(df['f'].values), df['Z_phase'].values,'ro')
    ax2.set_ylabel(r'$\bf \varphi\; [^{\circ}]$', color='r', fontsize=20)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    if g_title == 'none':
        plt.title(r'\bf Bode Plot', fontsize=20)
    elif g_title != 'none':
        plt.title(g_title, fontsize=20)
    if g_xlim != 'none':
        plt.xlim(g_xlim[0], g_xlim[1])
    if g_ylim != 'none':
        plt.ylim(g_ylim[0], g_ylim[1])
    if savefig != 'none':
        plt.savefig(savefig, format='eps')
    plt.show()

def homemadeCV_expread(path, file):
    """Function that return dataframe from *.txt 
    Cyclic Voltammetry data 
    from our homemade potentiostat"""
    read_one = pd.read_csv(path + file,
                           sep='\s+', header = None,
                           names = ['volt', 'current'],
                           engine = 'python')
    #Now, the error in measuring the potential difference 
    #that the instrument has is corrected.
    d = {'volt': -read_one['volt'].values, 'current': read_one['current'].values}
    return pd.DataFrame(data=d)    

def FilterVC(df,b='none',a='none'):
    """Function that filters the data corresponding 
    to cyclic voltammetry experiments.
    Return a dataframe with the values filtered in 
    columns called volt and current."""
    if b == 'none':
        f = 4 #order of the filter
    elif b != 'none':
        f = b
    if a == 'none':
        g = 0.05 # The denominator coefficient 
                 #vector of the filter.
    elif a != 'none':
        g = a
    b, a = butter(f, g)
    y = df['current'].values
    zi = lfilter_zi(b, a)
    z, _ = lfilter(b, a, y, zi=zi*y[0])
    mid = len(df['volt'].values)//2
    # Apply the filter again, to have a result filtered 
    #at an order the same as filtfilt.
    z2, _ = lfilter(b, a, z, zi=zi*z[0])
    # Use filtfilt to apply the filter.
    x_f1 = filtfilt(b, a, df['volt'].values[0:mid])
    y_f1 = filtfilt(b, a, df['current'].values[0:mid])
    x_f2 = filtfilt(b, a, df['volt'].values[mid:])
    y_f2 = filtfilt(b, a, df['current'].values[mid:])
    xfiltered = np.concatenate([x_f1, x_f2])
    yfiltered = np.concatenate([y_f1, y_f2])
    d = {'volt': xfiltered, 'current': yfiltered}
    return pd.DataFrame(data=d)

def homemadeAMP_expread(path, file_input):
    return pd.read_csv(path + file_input,
                       sep='\s+', header = None,
                       names = ['time', 'current'],
                       engine = 'python')

def FilterAMP(df,b='none',a='none'):
    """Function that filters the data corresponding 
    to amperometric experiments.
    Return a dataframe with the values filtered in 
    columns called time and current."""
    if b == 'none':
        f = 4 #order of the filter
    elif b != 'none':
        f = b
    if a == 'none':
        g = 0.05 # The denominator coefficient 
                 #vector of the filter.
    elif a != 'none':
        g = a
    b, a = butter(f, g)
    y = df['current'].values
    zi = lfilter_zi(b, a)
    z, _ = lfilter(b, a, y, zi=zi*y[0])
    mid = len(df['current'].values)//2
    # Apply the filter again, to have a result filtered 
    #at an order the same as filtfilt.
    z2, _ = lfilter(b, a, z, zi=zi*z[0])
    # Use filtfilt to apply the filter.
    xfiltered = filtfilt(b, a, df['time'].values)
    yfiltered = filtfilt(b, a, df['current'].values)
    d = {'time': xfiltered, 'current': yfiltered}
    return pd.DataFrame(data=d)

def FiltKalmanVC(df):
    V = df['volt'].values
    I = df['current'].values
    fls = FixedLagSmoother(dim_x=2, dim_z=1, N=8)

    fls.x = np.array([0., .5])
    fls.F = np.array([[1.,1.],
                     [0.,1.]])

    fls.H = np.array([[1.,0.]])
    fls.P *= 1 #state matrix
    fls.R *= 5. #
    fls.Q *= 0.00001 # noise matrix

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., .05])
    kf.F = np.array([[1.,1.],
                     [0.,1.]])
    kf.H = np.array([[1.,0.]])
    kf.P *= 1
    kf.R *= 5.
    kf.Q *= 0.00001

    zs1 = V
    for z in zs1:
            fls.smooth(z)  
        
    kf_x1, _, _, _ = kf.batch_filter(zs1)
    x1_smooth = np.array(fls.xSmooth)[:,0]
    
    zs2 = I
    for z in zs2:
            fls.smooth(z)  
    kf_x2, _, _, _ = kf.batch_filter(zs2)
    x2_smooth = np.array(fls.xSmooth)[:,0]
    d = {'volt': kf_x1[:, 0], 'current': kf_x2[:, 0]}
    return pd.DataFrame(data=d)

def FiltKalmanAMP(df):
    t = df['time'].values
    I = df['current'].values
    fls = FixedLagSmoother(dim_x=2, dim_z=1, N=8)

    fls.x = np.array([0., .5])
    fls.F = np.array([[1.,1.],
                     [0.,1.]])

    fls.H = np.array([[1.,0.]])
    fls.P *= 1 #state matrix
    fls.R *= 5. #
    fls.Q *= 0.00001 # noise matrix

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., .05])
    kf.F = np.array([[1.,1.],
                     [0.,1.]])
    kf.H = np.array([[1.,0.]])
    kf.P *= 1
    kf.R *= 5.
    kf.Q *= 0.00001

    zs1 = t
    for z in zs1:
            fls.smooth(z)  
        
    kf_x1, _, _, _ = kf.batch_filter(zs1)
    x1_smooth = np.array(fls.xSmooth)[:,0]
    
    zs2 = I
    for z in zs2:
            fls.smooth(z)  
    kf_x2, _, _, _ = kf.batch_filter(zs2)
    x2_smooth = np.array(fls.xSmooth)[:,0]
    d = {'time': kf_x1[:, 0], 'current': kf_x2[:, 0]}
    return pd.DataFrame(data=d)
