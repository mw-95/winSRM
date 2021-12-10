"""
This is a python implementation of the winSRM Snowmelt Runoff Model by Rango,Martinec & Roberts. 
For additional information on how the Model works, please refer to:
        
        SRM Users Manual, WinSRM Version 1.11, Feb. 2008
        https://jornada.nmsu.edu/bibliography/08-023.pdf
"""



# Module is finished for now
# Author: M.Witt
# marius.witt@stud-mail.uni-wuerzburg.de

import math
import os
import glob
import re
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import scipy.signal

def dict_zip(v,k = ['X','Y','Cs','Cp','ddf']):
    p = dict(zip(k,v))
    return p

def truncate(number, digits) -> float:
    """
    Helper function to truncate floats
    """
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def data_check(x,start = 2001,stop = 2021, start_day = '-09-01', end_day = '-08-31', multi = True):
    """ This function checks for the completeness of the Model inputs (T,P,Q).
        Inputs:
        x = either the output of synthesize(), or Data_xxxx.csv dataframe for a basin.
        start,stop = range of years, for which data should be checked. Does not inclued stop!
                     (defaults to 2001-2020)
        start_day, end_day = start- and end of the timeperiod
        multi = True if timeperiod includes 2 years
    """
    years = np.arange(start,stop,1)
    for year in years:
        if multi == True:       
            start_date = str(year-1) + start_day
            end_date = str(year) + end_day
        else:
            start_date = str(year) + start_day
            end_date = str(year) + end_day
     
        tmp_frame = x[(x["Date"] >= start_date) & (x["Date"] <= end_date)]
        
        a = sum(np.isnan(tmp_frame['T.mean']))
        b = sum(np.isnan(tmp_frame['Precipitation']))
        c = sum(np.isnan(tmp_frame['Runoff']))
        
        if a == b == c == 0:
            print(str(year) + ': Year is ok')
        else:
            print(str(year) + ': Data missing!')
            print('T    '+' P   '+'  Q')
            print(str(a) + ' , ' + str(b) + ' , ' + str(c))
        
    

def xy_param(tmp_frame, min_ratio = 0.2, max_ratio = 0.2, x_guess = 1, y_guess = 0.05):
    """ Helper function, used to get the optimal x/y parameters for a basin in each year of the
        optimizing process.
        min_ratio = determines the search radius at lower boundary of the recession plot. 
        max_ratio = determines the search radius at upper boundary of the recession plot.
        x_guess = first guess value for x, needed for nonlinear solver
        y_guess = first guess value for y, needed for nonlinear solver
        Output: optimal x/y combination for a given year 
    """
    runoff = {'Qn' : tmp_frame['Runoff'],
              'Qn+1' : tmp_frame['Runoff']
             }
    # Create a Dataframe
    df = pd.DataFrame(runoff, columns = ['Qn', 'Qn+1'])
    # Shift runoff values by one
    df['Qn+1'] = df['Qn+1'].shift(-1)
    # Drop the last NaN row
    df = df[:-1]
    # Calculate recession coefficient K
    df['k'] = (df['Qn+1'] / df['Qn'])
    
    
    # Get Qn/Qn+1 tuples with minimal k (lower boundary)
    # to solve the nonlinear equation system
    min_qn = min(df['Qn'])
    
    delta_min = min_ratio * min_qn
    if delta_min <= 1:
        delta_min = 1
    elif delta_min >= 5:
        delta_min = 5
    else:
        delta_min = delta_min
    
    max_qn = max(df['Qn'])
    
    delta_max = max_ratio * max_qn
    if delta_max <= 5:
        delta_max = 5
    else:
        delta_max = delta_max
    
    # Sort by k to get the lower boundary (true recession)
    tmp1 = df[df['Qn'] <= min_qn + delta_min ]
    tmp1 = tmp1.sort_values(by = 'k')
    tmp2 = df[df['Qn'] >= max_qn - delta_max]
    tmp2 = tmp2.sort_values(by = 'k')
    
    # Get inputs for the nonlinear equation system
    k1 = tmp1.iloc[0,2]
    k2 = tmp2.iloc[0,2]
    q1 = tmp1.iloc[0,0]
    q2 = tmp2.iloc[0,0]
    
    # Solve equation system for x/y
    xn = sp.Symbol('xn')
    yn = sp.Symbol('yn')
    ans = sp.nsolve([xn * q1 ** -yn - k1, xn * q2 ** -yn - k2], [xn,yn], [x_guess,y_guess])
    
    return ans


def r_squared(param_array,q_measured):
    # Create array
    r2_array = np.zeros((len(param_array[0])))
    mean_q = np.mean(q_measured[:])
    
    for h in range(len(param_array[0])):
        ssd = np.sum((q_measured[:] - param_array[:,h])**2)
        q_sum = np.sum((q_measured[:] - mean_q)**2)
        
        r2_array[h] = 1 - (ssd / q_sum)
    
    max_r2 = np.amax(r2_array)
    index = np.where(r2_array == max_r2)
    
    return index,max_r2,r2_array

def r_squared_help(q_model,q_measured):
    # Create array

    mean_q = np.mean(q_measured[:])
    ssd = np.sum((q_measured[:] - q_model[:])**2)
    q_sum = np.sum((q_measured[:] - mean_q)**2)
        
    r2 = 1 - (ssd / q_sum)

    return r2

def dv_help(q_model,q_measured):
    
    measured = sum(q_measured)
    modeled = sum(q_model)
    
    dv = ((measured - modeled) / measured) * -100
    
    return dv

class Setup_Model():
    """
    This class is used for setting up the Model and Optimizer, as well as synthesizing data from multiple 
    meteorological stations if needed.
    """
    def __init__(self,basin,path = 'C:/Users/Marius/Desktop/pysrm/Data',info = 'DEM_Statistics_all_basins',height = 
                 'Synth_Station_Heights',stations = 'Meteo_Stations_Info',year = 2020,start = 2001,stop = 2021, multi = True,
                start_day = '-09-01', end_day = '-08-31'):
        """
        These parameters are to be provided when initializing Setup_Model:
        basin    = basin identifier, as string (eg. '0004')
        path     = Path to root folder with data (only when it differs from default)
        info     = Name of csv file with Basin statistics (only when it differs from default)
        height   = Name of csv file heights of synthetic station (only when it differs from default)
        stations = Name of csv file with Meteo Station info (only when it differs from default)
        year     = Year to be modeled, as int [only for get_data()]
        start,stop = Range of years to be optimized/synthesized on, does not include stop. as int.
                     [for get_data_opti() and synthesize()]
        multi    = bool check, True if Timespan over multiple years
        start_day, end_day = Start- and Enddays, '-mm-dd'
        """
        # basin   = basin number
        # path    = path to folder with data
        # info    = path to basin_info csv
        # height  = path to basin height info csv
        #stations = path to station info csv
        self.basin     = basin
        self.path      = path
        self.year      = year
        self.info      = info
        self.height    = height
        self.stations  = stations
        self.start     = start
        self.stop      = stop
        self.multi     = multi
        self.start_day = start_day
        self.end_day   = end_day
    def get_data(self):
        """
        Use this to get the correct input for the Model. 
        No inputs are needed for this function.
        """
        # set directory
        os.chdir(self.path)
        # define start- and enddate
        if self.multi == True:
            start_date = str(self.year-1)+str(self.start_day)
            end_date   = str(self.year)+str(self.end_day)
        else:
            start_date = str(self.year)+str(self.start_day)
            end_date   = str(self.year)+str(self.end_day)
        # get corresponding data
        data = pd.read_csv(str(self.basin)+'/'+'Data_'+str(self.basin)+'.csv')
        data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
        data = data.dropna(axis=1,how='all')
        # get basin info from lookup table
        basin_info = pd.read_csv(self.info+'.csv',encoding='unicode_escape')
        basin_info = basin_info.loc[basin_info['Gauge'] == int(self.basin)]
        # get height information
        height_pd = pd.read_csv(self.basin+'/'+self.height+'.csv')
        # get optimal parameters 
        parameters = pd.read_csv(str(self.basin)+'/Parameters_'+self.basin+'.csv',encoding='unicode_escape')
        parameters = parameters.set_index('Year')
        parameters = parameters.loc[self.year]
        return data,basin_info,height_pd,parameters
    def get_data_free(self):
        """
        Use this to get the correct input for the Model. 
        No inputs are needed for this function.
        """
        # set directory
        os.chdir(self.path)
        # define start- and enddate
        if self.multi == True:
            start_date = str(self.year-1)+str(self.start_day)
            end_date   = str(self.year)+str(self.end_day)
        else:
            start_date = str(self.year)+str(self.start_day)
            end_date   = str(self.year)+str(self.end_day)
        # get corresponding data
        data = pd.read_csv(str(self.basin)+'/'+'Data_'+str(self.basin)+'.csv')
        data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
        data = data.dropna(axis=1,how='all')
        # get basin info from lookup table
        basin_info = pd.read_csv(self.info+'.csv',encoding='unicode_escape')
        basin_info = basin_info.loc[basin_info['Gauge'] == int(self.basin)]
        # get height information
        height_pd = pd.read_csv(str(self.basin)+'/'+self.height+'.csv')
        return data,basin_info,height_pd
    def get_data_opti(self):
        """
        Use this to get the correct input for the Optimizer. 
        No inputs are needed for this function.
        """
        # set directory
        os.chdir(self.path)
        # define start- and enddate
        if self.multi == True:
            start_date = str(self.start-1)+str(self.start_day)
            end_date   = str(self.stop-1)+str(self.end_day)
        else:
            start_date = str(self.start)+str(self.start_day)
            end_date   = str(self.stop)+str(self.end_day)
        # get corresponding data
        data = pd.read_csv(str(self.basin)+'/'+'Data_'+str(self.basin)+'.csv')
        data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
        data = data.dropna(axis=1,how='all')
        # get basin info from lookup table
        basin_info = pd.read_csv(self.info+'.csv',encoding='unicode_escape')
        basin_info = basin_info.loc[basin_info['Gauge'] == int(self.basin)]
        # get height information
        height_pd = pd.read_csv(str(self.basin)+'/'+self.height+'.csv')
        # create numpy array for iteration
        year_iter = np.arange(self.start,self.stop,1)
        return data,basin_info,height_pd,year_iter,self.basin
    def synthesize(self):
        """
        Use this to synthesize Data from multiple Meteo stations. 
        No inputs are needed for this function.
        """
        # set directory
        os.chdir(self.path+'/'+self.basin)
        # Get all files in the folder
        result = glob.glob('*.{}'.format('csv'))
        # define start- and enddate
        if self.multi == True:
            start_date = str(self.start-1)+str(self.start_day)
            end_date   = str(self.stop-1)+str(self.end_day)
        else:
            start_date = str(self.start)+str(self.start_day)
            end_date   = str(self.stop)+str(self.end_day)
        # Get station info csv
        info = pd.read_csv(self.path+'/'+self.stations+'.csv')
        info = info.drop(['Unnamed: 0'],axis = 1)
        # Split up result into different categories
        discharge = [x for x in result if re.search('discharge',x)]
        snow = [x for x in result if re.search('snow_cover',x)]
        meteo = [x for x in result if re.search('meteorology',x)]
        height = np.full((len(meteo)), np.NaN)
        # Get the height for each station from the lookup table
        for i in range(len(meteo)):
            station_id = int(meteo[i][0:6])
            info_temp = info.loc[info['StationID'] == station_id]
            height[i] = float(info_temp['Elevation'])
        # Get discharge and snow data into pandas df
        discharge_pd = pd.read_csv(discharge[0],encoding = 'unicode_escape')
        snow_pd = pd.read_csv(snow[0],encoding = 'unicode_escape')
        # Create numpy array to store the precipitation/temperature data for each meteo station
        tmp_t = np.full((len(discharge_pd),len(meteo)), np.NaN)
        tmp_p = np.full((len(discharge_pd),len(meteo)), np.NaN)
        # Create numpy array to store end result 
        t_arr = np.full((len(discharge_pd)), np.NaN)
        p_arr = np.full((len(discharge_pd)), np.NaN)
        # NaN-arrays for weighting
        t_weight = np.full((len(meteo)), np.NaN)
        p_weight = np.full((len(meteo)), np.NaN)
        # Get the weights for each station
        for i in range(len(meteo)):
            tmp_df = pd.read_csv(meteo[i])
            tmp_t[:,i] = tmp_df['T mean [deg C]'] 
            tmp_p[:,i] = tmp_df['Daily Precipitation [mm]']
            t_weight[i] = 1 - (sum(np.isnan(tmp_t[:,i])) / len(tmp_df))
            p_weight[i] = 1 - (sum(np.isnan(tmp_p[:,i])) / len(tmp_df))
        for i in range(len(discharge_pd)):
            n_nan = sum(np.isnan(tmp_t[i,:]))
            t_arr[i] = np.nansum(tmp_t[i,:]) / (len(meteo) - n_nan)
            n_nan = sum(np.isnan(tmp_p[i,:]))
            p_arr[i] = np.nansum(tmp_p[i,:]) / (len(meteo) - n_nan)
        # Get the heights for the synthed station
        height_p = (sum(np.multiply(p_weight,height)) / len(meteo)) 
        height_t = (sum(np.multiply(t_weight,height)) / len(meteo))
        # Split up the snow pd for each zone
        snow_col = ['Date','Snow_A','Snow_B','Snow_C','Snow_D','Snow_E','Snow_F','Snow_G','Snow_H','Snow_I','Snow_J','Snow_K',
                    'Snow_L','Snow_M','Snow_N','Snow_O','Snow_P','Snow_Q','Snow_R','Snow_S','Snow_T','Snow_U','Snow_V',
                    'Snow_W','Snow_X','Snow_Y','Snow_Z']
        nzone = len(snow_pd.columns)
        snow_col2 = snow_col[0:nzone]
        
        # Rename the snow dataframe, and multiply with 0.01 
        snow_pd.columns = snow_col2
        snow_pd.iloc[:,1:] = snow_pd.iloc[:,1:] * 0.01
        
        # Create new pandas dataframes for data and synthetic heights
        basin = {'Date' : discharge_pd["DateTime"],
                 'T.mean' : t_arr,
                 'Precipitation' : p_arr * 0.1,
                 'Runoff' : discharge_pd['Discharge']
        }
              
        synth_height = {'Height_T' : height_t,'Height_P' : height_p  
        }
        
        tmp_df = pd.DataFrame(basin, columns = ['Date', 'T.mean', 'Precipitation', 'Runoff'])
        
        # Merging the Snow- and Tmp_basin dataframe! 
        
        df = pd.merge(tmp_df, snow_pd, on='Date')
        
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        
        df2 = pd.DataFrame(synth_height, columns = ['Height_T', 'Height_P'],index=[0])
        
        df.to_csv('Data_'+self.basin+'.csv',index=False)
        df2.to_csv('Synth_Station_Heights.csv',index=False)
        return df
    def param_summary(self):
        """
        Use this to get a summary of the optimal Parameters for all available years.
        No inputs are needed for this function.
        """
        os.chdir(self.path)
        x = pd.read_csv(str(self.basin)+'/Parameters_'+str(self.basin)+'.csv',encoding='unicode_escape')
        tmp = list([])
        for i in ['X','Y','Cs','Cp','ddf','R2']:
            a = np.array([x[i].std(),x[i].min(),x[i].mean(),x[i].max(),x[i].median()])
            for i in range(len(a)):
                a[i] = truncate(a[i],3)
            tmp.append(a)
        df = pd.DataFrame(tmp, index =['X', 'Y', 'Cs', 'Cp', 'ddf', 'R2'],
                          columns =['Stddev','Min','Mean','Max','Median'])
        df.to_csv(str(self.basin)+'/Summary_'+str(self.basin)+'.csv')
        return df

def Model(x,Tcrit = 1,Tlapse = -0.65,Plapse = 0.04,sav_len = 61):
    """
    Use this function to model the discharge from a given basin.
    Takes the output from Setup_Model().get_data() as input.
    Additional Parameters:
    Tcrit = Critical Temperature, where Precipitation will be handled as 
            snow. Constant.
    Tlapse = Temperature Lapse Rate in °C/100m. Constant.
    Plapse = Precipitation Lapse Rate, in 1/100 per 100m. Constant.
    sav_len = Window length of the Temperature filter, used to
              determine snowpack state. Constant.
    Output: List of 8, in order:
    [q_model,q_measure,melt_roff,prec_roff,recession,snow_stor,snow_cover(S),temp(T),prec(P)]
    """
    # Unpack the data from Setup_Model class
    data   = x[0]
    basin  = x[1]
    height = x[2]
    params = x[3]
    
    # Basin Info #
    ##############
    # Get the reference heights for the synthetic station
    href_t = float(height['Height_T'])
    href_p = float(height['Height_P'])
    
    # Get the areas for each zone in the basin
    A_tmp = basin.columns[basin.columns.str.startswith('Area')]
    A = basin[A_tmp].to_numpy().astype('float')
    A = A[np.logical_not(np.isnan(A))]
    
    # Get the mean height for each zone in the basin
    zone_tmp = basin.columns[basin.columns.str.startswith('Elev')]
    zone_height = basin[zone_tmp].to_numpy().astype('float')
    zone_height = zone_height[np.logical_not(np.isnan(zone_height))]
    
    # Get the number of elevation zones in the basin
    nzones = len(A)
    
    # Get the number of days in the simulation period
    ndays = len(data)
    
    # Data #
    ########
    # Measured discharge data
    q_measured = data['Runoff'].to_numpy()
    # Temperature array
    T = data['T.mean'].to_numpy()
    # Precipitation array
    P = data['Precipitation'].to_numpy()
    # Snow-covered area
    sca = np.full((ndays,nzones), np.NaN)
    for j in range(nzones):
        # Changed from +3 to +4 because the runoff column was put before Snow
        sca[:,j] = data.iloc[:,j+4].to_numpy()
    # Modeled discharge
    q = np.full((ndays),np.NaN)
    q[0] = q_measured[0]
    
    # Parameters # 
    ##############
    x      = params['X']
    y      = params['Y']
    Cs     = params['Cs']
    Cp     = params['Cp']
    ddf    = params['ddf']
    Tcrit  = Tcrit
    Tlapse = Tlapse
    Plapse = Plapse
    state  = np.full((nzones), 1)
    
    # Get zero-crossings for Snowpack state
    T_temp = np.full((len(T),nzones),np.NaN)
    T_hat  = np.full((len(T),nzones),np.NaN)
    zero_crossing = list()
    
    # Run Savitzky-Golay Filter on mean temp and get 0-crossings for snowpack state
    for j in range(nzones):
        T_temp[:,j] = T[:] + ((zone_height[j] - href_t) / 100) * Tlapse
        T_hat[:,j] = scipy.signal.savgol_filter(T_temp[:,j], sav_len, 3)
        tmp = np.where(np.diff(np.sign(T_hat[:,j])))[0]
        zero_crossing.append(tmp)
    
    # Output arrays #
    #################
    melt = np.zeros((ndays))
    roff = np.zeros((ndays))
    rec  = np.zeros((ndays))
    stor = np.zeros((ndays,nzones))
    
    # Instance all internal model variables #
    #########################################
    k = 0
    Runoff = 0
    Snowmelt = 0
    Melt_runoff = 0
    Storage = np.zeros((nzones))
    Snowmelt_runoff = np.zeros((ndays,nzones))
    Precipitation_runoff = np.zeros((ndays,nzones))
    
    # Instance iterator
    iterator = it.product(range(len(q_measured)-1),range(nzones))
    
    # MODEL #
    #########
    
    for i,j in iterator:
        #Determining the Recession Coefficient k
        k = x * (q[i] ** -y)
        
        if k > 0.99:
            k = 0.99
        else:
            k = k
            
        # Check for the state of the snow pack (if it is ripe or not)
        if T_hat[i,j] > 0:
            state[j] = 1
        else:
            state[j] = 0
                
        # Adjust the Measured Temperature/Precipitation to the mean height of the Basin/Zone
        Tadj = T[i] + (((zone_height[j] - href_t) / 100) * Tlapse)
        Padj = P[i] + (P[i] * (((zone_height[j] - href_p) / 100) * Plapse))
        
        # Compute the expected Melt Depth. If T < 0, the expected Melt_Depth is set to 0.
        if Tadj > 0:
            Melt_Depth = Tadj * ddf
        else:
            Melt_Depth = 0
            
        # If T > Tcrit, there is Precipitation Runoff happening
        if Tadj > Tcrit and state[j] == 0:
            Snowmelt = Melt_Depth * sca[i,j]
            Runoff = Padj * (1-sca[i,j]) * Cp
            if Melt_Depth < Storage[j]:
                Melt_runoff = Melt_Depth * (1-sca[i,j])
                Storage[j] = Storage[j] - Melt_Depth
            elif Melt_Depth >= Storage[j]:
                Melt_runoff = Storage[j] * (1 - sca[i,j])
                Storage[j] = 0
        elif Tadj > Tcrit and state[j] == 1:
            Snowmelt = Melt_Depth * sca[i,j]
            Runoff = Padj * Cp
            if Melt_Depth < Storage[j]:
                Melt_runoff = Melt_Depth * (1-sca[i,j])
                Storage[j] = Storage[j] - Melt_Depth
            elif Melt_Depth >= Storage[j]:
                Melt_runoff = Storage[j] * (1 - sca[i,j])
                Storage[j] = 0
        # If T < Tcrit, there is no Precipitation Runoff happening, but snowfall.
        # Snowpack state is unimportant
        elif Tadj <= Tcrit:
            Snowmelt = Melt_Depth * sca[i,j]
            Runoff = 0
            Storage[j] = Storage[j] + Padj
            if Melt_Depth < Storage[j]:
                Melt_runoff = Melt_Depth * (1-sca[i,j])
                Storage[j] = Storage[j] - Melt_Depth
            elif Melt_Depth >= Storage[j]:
                Melt_runoff = Storage[j] * (1 - sca[i,j])
                Storage[j] = 0
                
        # Calculate daily runoff Components
        Snowmelt_runoff[i,j] = (Snowmelt + Melt_runoff) * (A[j] * (10000/86400)) * (1-k) * Cs
        Precipitation_runoff[i,j] = Runoff * (A[j] * (10000/86400)) * (1-k)
        stor[i,j] = Storage[j]
        
        # Sum the discharge from all zones
        if j == nzones-1:
            Recession_flow = q[i] * k
            q[i+1] = np.sum(Snowmelt_runoff[i,:]) + np.sum(Precipitation_runoff[i,:]) + Recession_flow
            melt[i] = np.sum(Snowmelt_runoff[i,:])
            roff[i] = np.sum(Precipitation_runoff[i,:])
            rec[i] = Recession_flow
        
       
    print('X :'+str(x)+',Y :'+str(y)+',Cs :'+str(Cs)+',Cp :'+str(Cp)+',ddf :'+str(ddf))
    return q,q_measured,melt,roff,rec,stor,sca,T,P

def Model_free(x,p,Tcrit = 1,Tlapse = -0.65,Plapse = 0.04,sav_len = 61):
    """
    Use this function to model the discharge from a given basin.
    Takes the output from Setup_Model().get_data() as input.
    Additional Parameters:
    p     = list of parameters in Order: x,y,cs,cp,ddf
    Tcrit = Critical Temperature, where Precipitation will be handled as 
            snow. Constant.
    Tlapse = Temperature Lapse Rate in °C/100m. Constant.
    Plapse = Precipitation Lapse Rate, in 1/100 per 100m. Constant.
    sav_len = Window length of the Temperature filter, used to
              determine snowpack state. Constant.
    Output: List of 8, in order:
    [q_model,q_measure,melt_roff,prec_roff,recession,snow_stor,snow_cover(S),temp(T),prec(P)]
    """
    # Unpack the data from Setup_Model class
    data   = x[0]
    basin  = x[1]
    height = x[2]
    params = dict_zip(p)
    
    # Basin Info #
    ##############
    # Get the reference heights for the synthetic station
    href_t = float(height['Height_T'])
    href_p = float(height['Height_P'])
    
    # Get the areas for each zone in the basin
    A_tmp = basin.columns[basin.columns.str.startswith('Area')]
    A = basin[A_tmp].to_numpy().astype('float')
    A = A[np.logical_not(np.isnan(A))]
    
    # Get the mean height for each zone in the basin
    zone_tmp = basin.columns[basin.columns.str.startswith('Elev')]
    zone_height = basin[zone_tmp].to_numpy().astype('float')
    zone_height = zone_height[np.logical_not(np.isnan(zone_height))]
    
    # Get the number of elevation zones in the basin
    nzones = len(A)
    
    # Get the number of days in the simulation period
    ndays = len(data)
    
    # Data #
    ########
    # Measured discharge data
    q_measured = data['Runoff'].to_numpy()
    # Temperature array
    T = data['T.mean'].to_numpy()
    # Precipitation array
    P = data['Precipitation'].to_numpy()
    # Snow-covered area
    sca = np.full((ndays,nzones), np.NaN)
    for j in range(nzones):
        sca[:,j] = data.iloc[:,j+4].to_numpy()
    # Modeled discharge
    q = np.full((ndays),np.NaN)
    q[0] = q_measured[0]
    
    # Parameters # 
    ##############
    x      = params['X']
    y      = params['Y']
    Cs     = params['Cs']
    Cp     = params['Cp']
    ddf    = params['ddf']
    Tcrit  = Tcrit
    Tlapse = Tlapse
    Plapse = Plapse
    state  = np.full((nzones), 1)
    
    # Get zero-crossings for Snowpack state
    T_temp = np.full((len(T),nzones),np.NaN)
    T_hat  = np.full((len(T),nzones),np.NaN)
    zero_crossing = list()
    
    # Run Savitzky-Golay Filter on mean temp and get 0-crossings for snowpack state
    for j in range(nzones):
        T_temp[:,j] = T[:] + ((zone_height[j] - href_t) / 100) * Tlapse
        T_hat[:,j] = scipy.signal.savgol_filter(T_temp[:,j], sav_len, 3)
        tmp = np.where(np.diff(np.sign(T_hat[:,j])))[0]
        zero_crossing.append(tmp)
    
    # Output arrays #
    #################
    melt = np.zeros((ndays))
    roff = np.zeros((ndays))
    rec  = np.zeros((ndays))
    stor = np.zeros((ndays,nzones))
    
    # Instance all internal model variables #
    #########################################
    k = 0
    Runoff = 0
    Snowmelt = 0
    Melt_runoff = 0
    Storage = np.zeros((nzones))
    Snowmelt_runoff = np.zeros((ndays,nzones))
    Precipitation_runoff = np.zeros((ndays,nzones))
    
    # Instance iterator
    iterator = it.product(range(len(q_measured)-1),range(nzones))
    
    # MODEL #
    #########
    
    for i,j in iterator:
        #Determining the Recession Coefficient k
        k = x * (q[i] ** -y)
        
        if k > 0.99:
            k = 0.99
        else:
            k = k
            
        # Check for the state of the snow pack (if it is ripe or not)
        if T_hat[i,j] > 0:
            state[j] = 1
        else:
            state[j] = 0
                
        # Adjust the Measured Temperature/Precipitation to the mean height of the Basin/Zone
        Tadj = T[i] + (((zone_height[j] - href_t) / 100) * Tlapse)
        Padj = P[i] + (P[i] * (((zone_height[j] - href_p) / 100) * Plapse))
        
        # Compute the expected Melt Depth. If T < 0, the expected Melt_Depth is set to 0.
        if Tadj > 0:
            Melt_Depth = Tadj * ddf
        else:
            Melt_Depth = 0
            
        # If T > Tcrit, there is Precipitation Runoff happening
        if Tadj > Tcrit and state[j] == 0:
            Snowmelt = Melt_Depth * sca[i,j]
            Runoff = Padj * (1-sca[i,j]) * Cp
            if Melt_Depth < Storage[j]:
                Melt_runoff = Melt_Depth * (1-sca[i,j])
                Storage[j] = Storage[j] - Melt_Depth
            elif Melt_Depth >= Storage[j]:
                Melt_runoff = Storage[j] * (1 - sca[i,j])
                Storage[j] = 0
        elif Tadj > Tcrit and state[j] == 1:
            Snowmelt = Melt_Depth * sca[i,j]
            Runoff = Padj * Cp
            if Melt_Depth < Storage[j]:
                Melt_runoff = Melt_Depth * (1-sca[i,j])
                Storage[j] = Storage[j] - Melt_Depth
            elif Melt_Depth >= Storage[j]:
                Melt_runoff = Storage[j] * (1 - sca[i,j])
                Storage[j] = 0
        # If T < Tcrit, there is no Precipitation Runoff happening, but snowfall.
        # Snowpack state is unimportant
        elif Tadj <= Tcrit:
            Snowmelt = Melt_Depth * sca[i,j]
            Runoff = 0
            Storage[j] = Storage[j] + Padj
            if Melt_Depth < Storage[j]:
                Melt_runoff = Melt_Depth * (1-sca[i,j])
                Storage[j] = Storage[j] - Melt_Depth
            elif Melt_Depth >= Storage[j]:
                Melt_runoff = Storage[j] * (1 - sca[i,j])
                Storage[j] = 0
                
        # Calculate daily runoff Components
        Snowmelt_runoff[i,j] = (Snowmelt + Melt_runoff) * (A[j] * (10000/86400)) * (1-k) * Cs
        Precipitation_runoff[i,j] = Runoff * (A[j] * (10000/86400)) * (1-k)
        stor[i,j] = Storage[j]
        
        # Sum the discharge from all zones
        if j == nzones-1:
            Recession_flow = q[i] * k
            q[i+1] = np.sum(Snowmelt_runoff[i,:]) + np.sum(Precipitation_runoff[i,:]) + Recession_flow
            melt[i] = np.sum(Snowmelt_runoff[i,:])
            roff[i] = np.sum(Precipitation_runoff[i,:])
            rec[i] = Recession_flow
        
       
    print('X :'+str(x)+',Y :'+str(y)+',Cs :'+str(Cs)+',Cp :'+str(Cp)+',ddf :'+str(ddf))
    return q,q_measured,melt,roff,rec,stor,sca,T,P


def Optimizer(x,prefix = 'Parameters_',Cp_default = 0.5, ddf_default = 0.5,Tcrit = 1,Tlapse = -0.65,Plapse = 
              0.04,sav_len = 61, start_day = '-09-01', end_day = '-08-31', multi = True, verbose = True):
    """
    Use this function to get the Optimal Parameters for a given basin for a range of years.
    Takes the output from Setup_Model().get_data_opti() as input.
    Additional Parameters:
    prefix = prefix for the resulting parameter .csv file, as string.
    Cp_default = first guess for Cp, Constant.
    ddf_default = first guess for the degree-day-factor, Constant.
    For Tcrit, Tlapse, Plapse, sav_len see Model() help.
    Output: dataframe of optimal parameters x,y,cs,cp,ddf and max. r squared for a range of years
            in a given basin.
    """
    data      = x[0]
    basin     = x[1]
    height    = x[2]
    year_iter = x[3]
    basin_nr  = x[4]
    
    # Create Pandas Dataframe to store the optimal Parameters
    opti_df = pd.DataFrame(columns=['Year', 'X', 'Y', 'Cs', 'Cp', 'ddf', 'R2'])
    
    # Basin Info #
    ##############
    # Get the reference heights for the synthetic station
    href_t = float(height['Height_T'])
    href_p = float(height['Height_P'])
    
    # Get the areas for each zone in the basin
    A_tmp = basin.columns[basin.columns.str.startswith('Area')]
    A = basin[A_tmp].to_numpy().astype('float')
    A = A[np.logical_not(np.isnan(A))]
    
    # Get the mean height for each zone in the basin
    zone_tmp = basin.columns[basin.columns.str.startswith('Elev')]
    zone_height = basin[zone_tmp].to_numpy().astype('float')
    zone_height = zone_height[np.logical_not(np.isnan(zone_height))]
    
    # get the xy parameters with the xy_param function
    
    for i in range(len(year_iter)):
        if multi == True:
            start_date = str(year_iter[i]-1)+start_day
            end_date = str(year_iter[i])+end_day
        else:
            start_date = str(year_iter[i])+start_day
            end_date = str(year_iter[i])+end_day
        year = int(year_iter[i])
        
        if verbose == True:
            print(str(year))
        
        # Subset dataframe to single hydrological year
        tmp_frame = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
        
        ans = xy_param(tmp_frame)
        
        # Get the number of elevation zones in the basin
        nzones = len(A)

        # Get the number of days in the simulation period
        ndays = len(tmp_frame)

        # Data #
        ########
        # Measured discharge data
        q_measured = tmp_frame['Runoff'].to_numpy()
        # Temperature array
        T = tmp_frame['T.mean'].to_numpy()
        # Precipitation array
        P = tmp_frame['Precipitation'].to_numpy()
        # Snow-covered area
        sca = np.full((ndays,nzones), np.NaN)
        for j in range(nzones):
            sca[:,j] = tmp_frame.iloc[:,j+4].to_numpy()
        # Modeled discharge
        q = np.full((ndays),np.NaN)
        q[0] = q_measured[0]
        
        # Parameters #
        ##############
        x      = float(abs(ans[0]))
        y      = float(abs(ans[1]))
        Cs     = np.arange(0.05,1.01,0.02)
        Cp     = float(Cp_default)
        ddf    = float(ddf_default)
        Tcrit  = float(Tcrit)
        Tlapse = float(Tlapse)
        Plapse = float(Plapse)
        state  = np.full((nzones), 1)
        
        # Set the Parameter to be optimized
        Param  = Cs
        # Create param array to store 
        param_array = np.full((len(q), len(Param)), np.nan)
        param_array[0,:] = q_measured[0]
        
        # Get zero-crossings for Snowpack state
        T_temp = np.full((len(T),nzones),np.NaN)
        T_hat  = np.full((len(T),nzones),np.NaN)
        zero_crossing = list()

        # Run Savitzky-Golay Filter on mean temp and get 0-crossings for snowpack state
        for j in range(nzones):   
            T_temp[:,j] = T[:] + ((zone_height[j] - href_t) / 100) * Tlapse
            T_hat[:,j] = scipy.signal.savgol_filter(T_temp[:,j], sav_len, 3)
            tmp = np.where(np.diff(np.sign(T_hat[:,j])))[0]
            zero_crossing.append(tmp)
        
        # Create and iterator
        iterator = it.product(range(len(Param)),range(len(q_measured)-1),range(nzones))
        
        # Instance internal variables
        k           = 0
        Runoff      = 0
        Snowmelt    = 0
        Melt_runoff = 0
        Storage     = np.full((nzones),np.nan)
        Zone_runoff = np.full((ndays,nzones),np.nan)
        
        for h,i,j in iterator:
            #Determining the Recession Coefficient k
            k = x * (q[i] ** -y)
            
            if k >= 0.99:
                k = 0.99
            else:
                k = k
            
            # Check for the state of the snow pack (if it is ripe or not)
            if T_hat[i,j] > 0:
                state[j] = 1
            else:
                state[j] = 0
            
            # Adjust the Measured Temperature/Precipitation to the mean height of the Basin/Zone
            Tadj = T[i] + ((zone_height[j] - href_t) / 100) * Tlapse
            Padj = P[i] + (P[i] * (((zone_height[j] - href_p) / 100) * Plapse))
            
            # Compute the expected Melt Depth. If T < 0, the expected Melt_Depth is set to 0.
            if Tadj > 0:
                Melt_Depth = Tadj * ddf
            else:
                Melt_Depth = 0
            
            # If T > Tcrit, there is Precipitation Runoff happening
            if Tadj > Tcrit and state[j] == 0:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj * (1-sca[i,j])
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                else:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            #The same thing for state = 1
            elif Tadj > Tcrit and state[j] == 1:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                elif Melt_Depth >= Storage[j]:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # If T < Tcrit, there is no Precipitation Runoff happening, but snowfall.
            # Snowpack state is unimportant
            elif Tadj <= Tcrit:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = 0
                Storage[j] = Storage[j] + Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                else:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # Calculate daily runoff Components
            Zone_runoff[i,j] = ((Melt_runoff + Snowmelt) *  Cs[h] + Runoff * Cp) * (A[j] * (10000/86400)) * (1-k)

            if j == nzones-1:
                q[i+1] = np.nansum(Zone_runoff[i,:]) + (q[i] * k)
                param_array[i+1,h] = q[i+1]
                
        # Get index with maximum R²
        index = r_squared(param_array,q_measured)
        # Replace array with optimal parameter
        Cs = float(Param[index[0]])
        # Set next array to be optimized as parameter
        Cp = np.arange(0.01,1.01,0.02)
        Param = Cp
       
        param_array = np.full((len(q), len(Param)), np.nan)
        param_array[0,:] = q_measured[0]
        iterator = it.product(range(len(Param)),range(len(q_measured)-1),range(nzones))
        
        for h,i,j in iterator:
            #Determining the Recession Coefficient k
            k = x * (q[i] ** -y)
            
            if k >= 0.99:
                k = 0.99
            else:
                k = k
            
            # Check for the state of the snow pack (if it is ripe or not)
            if T_hat[i,j] > 0:
                state[j] = 1
            else:
                state[j] = 0
            
            # Adjust the Measured Temperature/Precipitation to the mean height of the Basin/Zone
            Tadj = T[i] + ((zone_height[j] - href_t) / 100) * Tlapse
            Padj = P[i] + (P[i] * (((zone_height[j] - href_p) / 100) * Plapse))
            
            # Compute the expected Melt Depth. If T < 0, the expected Melt_Depth is set to 0.
            if Tadj > 0:
                Melt_Depth = Tadj * ddf
            else:
                Melt_Depth = 0
            
            # If T > Tcrit, there is Precipitation Runoff happening
            if Tadj > Tcrit and state[j] == 0:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj * (1-sca[i,j])
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                else:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            #The same thing for state = 1
            elif Tadj > Tcrit and state[j] == 1:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                elif Melt_Depth >= Storage[j]:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # If T < Tcrit, there is no Precipitation Runoff happening, but snowfall.
            # Snowpack state is unimportant
            elif Tadj <= Tcrit:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = 0
                Storage[j] = Storage[j] + Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                else:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # Calculate daily runoff Components
            Zone_runoff[i,j] = ((Melt_runoff + Snowmelt) *  Cs + Runoff * Cp[h]) * (A[j] * (10000/86400)) * (1-k)

            if j == nzones-1:
                q[i+1] = np.nansum(Zone_runoff[i,:]) + (q[i] * k)
                param_array[i+1,h] = q[i+1]
                
        # Get index with maximum R²
        index = r_squared(param_array,q_measured)
        # Replace array with optimal parameter
        Cp = float(Param[index[0]])
        # Set next array to be optimized as parameter
        ddf = np.arange(0.01,1.01,0.02)
        Param = ddf
        
        param_array = np.full((len(q), len(Param)), np.nan)
        param_array[0,:] = q_measured[0]
        iterator = it.product(range(len(Param)),range(len(q_measured)-1),range(nzones))

        for h,i,j in iterator:
            #Determining the Recession Coefficient k
            k = x * (q[i] ** -y)
            
            if k >= 0.99:
                k = 0.99
            else:
                k = k
            
            # Check for the state of the snow pack (if it is ripe or not)
            if T_hat[i,j] > 0:
                state[j] = 1
            else:
                state[j] = 0
            
            # Adjust the Measured Temperature/Precipitation to the mean height of the Basin/Zone
            Tadj = T[i] + ((zone_height[j] - href_t) / 100) * Tlapse
            Padj = P[i] + (P[i] * (((zone_height[j] - href_p) / 100) * Plapse))
            
            # Compute the expected Melt Depth. If T < 0, the expected Melt_Depth is set to 0.
            if Tadj > 0:
                Melt_Depth = Tadj * ddf[h]
            else:
                Melt_Depth = 0
            
            # If T > Tcrit, there is Precipitation Runoff happening
            if Tadj > Tcrit and state[j] == 0:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj * (1-sca[i,j])
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                else:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            #The same thing for state = 1
            elif Tadj > Tcrit and state[j] == 1:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                elif Melt_Depth >= Storage[j]:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # If T < Tcrit, there is no Precipitation Runoff happening, but snowfall.
            # Snowpack state is unimportant
            elif Tadj <= Tcrit:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = 0
                Storage[j] = Storage[j] + Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                else:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # Calculate daily runoff Components
            Zone_runoff[i,j] = ((Melt_runoff + Snowmelt) *  Cs + Runoff * Cp) * (A[j] * (10000/86400)) * (1-k)

            if j == nzones-1:
                q[i+1] = np.nansum(Zone_runoff[i,:]) + (q[i] * k)
                param_array[i+1,h] = q[i+1]

        # Get index with maximum R²
        index = r_squared(param_array,q_measured)
        # Replace array with optimal parameter
        ddf = float(Param[index[0]])
        R2  = float(index[1])
        
        # Save the optimized parameters in a pandas df
        optimal = np.array([int(year),x,y,Cs,Cp,ddf,R2])
        opti_df.loc[len(opti_df)] = optimal
        
        if len(opti_df) == len(year_iter):
            opti_df = opti_df.set_index('Year')
            opti_df.to_csv(str(basin_nr)+'/'+str(prefix)+str(basin_nr)+'.csv')
        
    return opti_df

def Lag_Time_Finder(x):
    """
    Function to find appropriate time lag for Catchment
    """
    #q,q_measured,melt,roff,rec,stor,sca,T,P
    q = x[0]
    q_measured = x[1]
    snowmelt = x[2]
    precip = x[3]
    rec = x[4]
    initial = q[0]
    lags = np.arange(6,24.5,0.5)
    
    param_array = np.full((len(q), len(lags)), np.nan)

    for j in range(len(lags)):
        
        if lags[j] <= 18:
  
            param_array[0,j]  = q[0]

            factor = (0.25 + (0.25/6) * lags[j])

            for i in range(len(q)-1):
                param_array[i+1,j] = (factor*(snowmelt[i]+precip[i] + rec[i]) + (1-factor) * (snowmelt[i+1] + precip[i+1] + rec[i+1]))
        
        else:
           
            param_array[0,j] = q[0]
            param_array[1,j] = q[1]
        
            factor = (0 + (0.25/6) * lags[j])
            
            for i in range(len(q)-2):
                param_array[i+2,j] = (factor * (snowmelt[i] + precip[i] + rec[i]) + (1-factor) * (snowmelt[i+1] + precip[i+1] + rec[i+1]))
        
              
    index = r_squared(param_array,q_measured)
    
    lag = lags[index[0]]
    r2 = index[1]
    array = index[2]
   
    return lag[0],r2,array

def Iterator_Model(x,Tcrit = 1,Tlapse = -0.65,Plapse = 0.04,sav_len = 61):
    """
    Use this function to model the discharge from a given basin.
    Takes the output from Setup_Model().get_data() as input.
    Additional Parameters:
    Tcrit = Critical Temperature, where Precipitation will be handled as 
            snow. Constant.
    Tlapse = Temperature Lapse Rate in °C/100m. Constant.
    Plapse = Precipitation Lapse Rate, in 1/100 per 100m. Constant.
    sav_len = Window length of the Temperature filter, used to
              determine snowpack state. Constant.
    Output: List of 8, in order:
    [q_model,q_measure,melt_roff,prec_roff,recession,snow_stor,snow_cover(S),temp(T),prec(P)]
    """
    # Unpack the data from Setup_Model class
    data   = x[0]
    basin  = x[1]
    height = x[2]
    
    # Basin Info #
    ##############
    # Get the reference heights for the synthetic station
    href_t = float(height['Height_T'])
    href_p = float(height['Height_P'])
    
    # Get the areas for each zone in the basin
    A_tmp = basin.columns[basin.columns.str.startswith('Area')]
    A = basin[A_tmp].to_numpy().astype('float')
    A = A[np.logical_not(np.isnan(A))]
    
    # Get the mean height for each zone in the basin
    zone_tmp = basin.columns[basin.columns.str.startswith('Elev')]
    zone_height = basin[zone_tmp].to_numpy().astype('float')
    zone_height = zone_height[np.logical_not(np.isnan(zone_height))]
    
    # Get the number of elevation zones in the basin
    nzones = len(A)
    
    # Get the number of days in the simulation period
    ndays = len(data)
    
    # Data #
    ########
    # Measured discharge data
    q_measured = data['Runoff'].to_numpy()
    # Temperature array
    T = data['T.mean'].to_numpy()
    # Precipitation array
    P = data['Precipitation'].to_numpy()
    # Snow-covered area
    sca = np.full((ndays,nzones), np.NaN)
    for j in range(nzones):
        # Changed from +3 to +4 because the runoff column was put before Snow
        sca[:,j] = data.iloc[:,j+4].to_numpy()
    # Modeled discharge
    q = np.full((ndays),np.NaN)
    q[0] = q_measured[0]
    
    ans = xy_param(data)
    
    # Parameters # 
    ##############
    x      = float(abs(ans[0]))
    y      = float(abs(ans[1]))
    Cs     = np.arange(0.05,1,0.05)
    Cp     = np.arange(0.05,1,0.05)
    ddf    = np.arange(0.05,1,0.05)
    Tcrit  = Tcrit
    Tlapse = Tlapse
    Plapse = Plapse
    state  = np.full((nzones), 1)
    
    # Get zero-crossings for Snowpack state
    T_temp = np.full((len(T),nzones),np.NaN)
    T_hat  = np.full((len(T),nzones),np.NaN)
    zero_crossing = list()
    
    # Run Savitzky-Golay Filter on mean temp and get 0-crossings for snowpack state
    for j in range(nzones):
        T_temp[:,j] = T[:] + ((zone_height[j] - href_t) / 100) * Tlapse
        T_hat[:,j] = scipy.signal.savgol_filter(T_temp[:,j], sav_len, 3)
        tmp = np.where(np.diff(np.sign(T_hat[:,j])))[0]
        zero_crossing.append(tmp)
    
    # Output arrays #
    #################
    melt = np.zeros((ndays))
    roff = np.zeros((ndays))
    rec  = np.zeros((ndays))
    stor = np.zeros((ndays,nzones))
  
    # Instance iterator
    iterator = [x for x in it.product(range(len(q_measured)-1),range(nzones))]
    param_iter = it.product(Cs,Cp,ddf)
    
    #cs_list = []
    #cp_list = []
    #ddf_list = []
    #r2_list = []
    
    d = []
    
    # MODEL #
    #########
    
    for Cs,Cp,ddf in param_iter:
        
        q = np.full((ndays),np.NaN)
        q[0] = q_measured[0]
        
        k = 0
        Runoff = 0
        Snowmelt = 0
        Melt_runoff = 0
        Storage = np.zeros((nzones))
        Snowmelt_runoff = np.zeros((ndays,nzones))
        Precipitation_runoff = np.zeros((ndays,nzones))
        
        print(Cs,Cp,ddf)
    
        for i,j in iterator:

            #Determining the Recession Coefficient k
            k = x * (q[i] ** -y)

            if k > 0.99:
                k = 0.99
            else:
                k = k

            # Check for the state of the snow pack (if it is ripe or not)
            if T_hat[i,j] > 0:
                state[j] = 1
            else:
                state[j] = 0

            # Adjust the Measured Temperature/Precipitation to the mean height of the Basin/Zone
            Tadj = T[i] + (((zone_height[j] - href_t) / 100) * Tlapse)
            Padj = P[i] + (P[i] * (((zone_height[j] - href_p) / 100) * Plapse))

            # Compute the expected Melt Depth. If T < 0, the expected Melt_Depth is set to 0.
            if Tadj > 0:
                Melt_Depth = Tadj * ddf
            else:
                Melt_Depth = 0

            # If T > Tcrit, there is Precipitation Runoff happening
            if Tadj > Tcrit and state[j] == 0:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj * (1-sca[i,j]) * Cp
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                elif Melt_Depth >= Storage[j]:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            elif Tadj > Tcrit and state[j] == 1:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = Padj * Cp
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                elif Melt_Depth >= Storage[j]:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0
            # If T < Tcrit, there is no Precipitation Runoff happening, but snowfall.
            # Snowpack state is unimportant
            elif Tadj <= Tcrit:
                Snowmelt = Melt_Depth * sca[i,j]
                Runoff = 0
                Storage[j] = Storage[j] + Padj
                if Melt_Depth < Storage[j]:
                    Melt_runoff = Melt_Depth * (1-sca[i,j])
                    Storage[j] = Storage[j] - Melt_Depth
                elif Melt_Depth >= Storage[j]:
                    Melt_runoff = Storage[j] * (1 - sca[i,j])
                    Storage[j] = 0

            # Calculate daily runoff Components
            Snowmelt_runoff[i,j] = (Snowmelt + Melt_runoff) * (A[j] * (10000/86400)) * (1-k) * Cs
            Precipitation_runoff[i,j] = Runoff * (A[j] * (10000/86400)) * (1-k)
            stor[i,j] = Storage[j]

            # Sum the discharge from all zones
            if j == nzones-1:
                Recession_flow = q[i] * k
                q[i+1] = np.sum(Snowmelt_runoff[i,:]) + np.sum(Precipitation_runoff[i,:]) + Recession_flow
        
        r2 = r_squared_help(q,q_measured)
        dv = dv_help(q,q_measured)
        
        tmp = ({
            'x':x,
            'y':y,
            'Cs':Cs,
            'Cp':Cp,
            'ddf':ddf,
            'Tcrit':Tcrit,
            'Tlapse':Tlapse,
            'NSE':r2,
            'Dv':dv
        })
        d.append(tmp)
        #r2_list.append(r2)
        #cs_list.append(Cs)
        #cp_list.append(Cp)
        #ddf_list.append(ddf)

    
    
    
    return d