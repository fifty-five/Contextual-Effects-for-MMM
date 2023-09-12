# Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt



# Functions to work with the dates
datestring_to_datetime = lambda s: datetime.strptime(s, '%Y/%m/%d')
datetime_to_int = lambda dt: int(datetime.strftime(dt, '%Y%m%d')) # to integer with format YYYYMMDD
datestring_to_int = lambda s: datetime_to_int(datestring_to_datetime(s))



def punctual_event_effect(
    event_name, 
    delta_days_peak_effect, 
    delta_time_steps_before_peak, 
    delta_time_steps_after_peak, 
    curve_std_idx,
    time_series_start, 
    time_series_end, 
    timestep_frequency,
    events_table_path, 
    verbose, 
    figsize=(15, 4)
):
    
    """
    Inputs:
        event_name: str -> Name of the events, wrt the table containig all the events names and dates
        delta_days_peak_effect: int -> Ex: a value of -2 would mean that the maximum effect of the event 
            over the KPI would arrive during the time step of two DAYS before the date of the event
        delta_time_steps_before_peak: int -> Standard duration (in time steps) of the effect of the event before the peak arrives
        delta_time_steps_after_peak: int -> Standard duration (in time steps) of the effect of the event after the peak arrives
        curve_std_idx: int -> {-1 (or <= -1): "exponential", 0: "piecewise_affine", 1 (or >= 1): "bell_curve"}
        time_series_start: str -> Date of the beginning of the whole time series, format = '%Y/%m/%d'
        time_series_end: str -> Date of the end of the whole time series, format = '%Y/%m/%d'
        timestep_frequency: str -> "D" for daily, "W-MON" for weekly, "MS" for montly
        events_table_path: str -> path to the events table with names and dates
        verbose: int -> 0, 1 or 2 ; 1 if user wants to get the plots, 2 for plot + scatter
        
    Outputs:
        pandas Series of the effect of the event through time
        
    Remark:
        We suppose that the total effect of a event (considering before, during and after peak) 
        cannot be longer than a year
        
    """
    
    
    
    # Initialize the dates and the series
    initial_date_range = pd.date_range(
        start = time_series_start,
        end = time_series_end, 
        freq = timestep_frequency
    )
    
    dt_start = initial_date_range[0].date()
    dt_end = initial_date_range[-1].date()
    
    time_delta_one_year = relativedelta(years = 1)
    
    # We look one year prior and after the initial_date_range to take into account potential effects of the events
    # that can occure right before or right after the initial_date_range
    dt_start_one_year_prior = dt_start - time_delta_one_year
    dt_end_one_year_after = dt_end + time_delta_one_year
    
    # We look two year prior and after the initial_date_range just to evoid bugs when assigning the effect of 
    # the first and last instance of the event during the script
    dt_start_two_years_prior = dt_start_one_year_prior - time_delta_one_year
    dt_end_two_years_after = dt_end_one_year_after + time_delta_one_year
    
    
    date_range_four_years_larger = pd.date_range(
        start = dt_start_two_years_prior, 
        end = dt_end_two_years_after, 
        freq = timestep_frequency
    )
    
    # Initially, the series used will be indexed using date_range_four_years_larger to make the calculations, 
    # we will then crop this series into the variable final_series to get the values of interest
    series = pd.Series(
        index = date_range_four_years_larger, 
        data = 0.
    )
    
    
    # Get the dataframe with all the events dates
    events_df = pd.read_csv(events_table_path)
    
    # Get all the events dates that correspond to event_name, and that happen
    # between one year before time_series_start and one year after time_series_end, 
    # because we want to take into account potential effects of the events that can occure 
    # right before or right after the date range of interest
    events_through_date_range = \
        events_df[(events_df["event_name"] == event_name) & 
                    (events_df["ds"].apply(datestring_to_int) >= datetime_to_int(dt_start_one_year_prior)) &
                    (events_df["ds"].apply(datestring_to_int) <= datetime_to_int(dt_end_one_year_after))]
        
    for i in range(len(events_through_date_range)):
        
        dt_event = datestring_to_datetime(events_through_date_range.iloc[i]["ds"])
        
        if timestep_frequency == 'MS':
            
            dt_event_peak = dt_event + timedelta(days = delta_days_peak_effect) # + because negative means before
            dt_peak_effect_timestep = dt_event_peak - timedelta(days = dt_event.day - 1)
            
            dt_before_peak_start = dt_peak_effect_timestep - relativedelta(months = delta_time_steps_before_peak)
            dt_before_peak_end = dt_peak_effect_timestep - relativedelta(months = 1)
            dt_after_peak_start = dt_peak_effect_timestep + relativedelta(months = 1)
            dt_after_peak_end = dt_peak_effect_timestep + relativedelta(months = delta_time_steps_after_peak)
            
        elif timestep_frequency == 'W-MON':
            
            dt_event_peak = dt_event + timedelta(days = delta_days_peak_effect) # + because negative means before
            dt_peak_effect_timestep = dt_event_peak - timedelta(days = dt_event_peak.weekday())
            
            dt_before_peak_start = dt_peak_effect_timestep - timedelta(weeks = delta_time_steps_before_peak)
            dt_before_peak_end = dt_peak_effect_timestep - timedelta(weeks = 1)
            dt_after_peak_start = dt_peak_effect_timestep + timedelta(weeks = 1)
            dt_after_peak_end = dt_peak_effect_timestep + timedelta(weeks = delta_time_steps_after_peak)
            
        elif timestep_frequency == 'D':
            
            dt_event_peak = dt_event + timedelta(days = delta_days_peak_effect) # + because negative means before
            dt_peak_effect_timestep = dt_event_peak
            
            dt_before_peak_start = dt_peak_effect_timestep - timedelta(days = delta_time_steps_before_peak)
            dt_before_peak_end = dt_peak_effect_timestep - timedelta(days = 1)
            dt_after_peak_start = dt_peak_effect_timestep + timedelta(days = 1)
            dt_after_peak_end = dt_peak_effect_timestep + timedelta(days = delta_time_steps_after_peak)
            
        else:
            raise ValueError("timestep_frequency is not accepted, must select one of the following: " + \
                             "['D', 'W-MON', 'MS']")
        
        
        if curve_std_idx <= -1: # "exponential"
            
            values_before_peak = [np.exp(4*(x-0)/delta_time_steps_before_peak)\
                                  for x in np.arange(-delta_time_steps_before_peak, 0)]
            values_after_peak = [np.exp(4*(x-0)/delta_time_steps_after_peak)\
                                 for x in np.arange(-1, -(delta_time_steps_after_peak+1), -1)]
        
        elif curve_std_idx == 0: # "piecewise_affine"
            
            values_before_peak = np.arange(1, delta_time_steps_before_peak+1, 1)/(delta_time_steps_before_peak+1)
            values_after_peak = np.arange(delta_time_steps_after_peak, 0, -1)/(delta_time_steps_after_peak+1)
            
        elif curve_std_idx >= 1: # "bell_curve"
            
            values_before_peak = [np.exp(-3.0*((0-x)**3/((delta_time_steps_before_peak+1)**3)))\
                                  for x in np.arange(-delta_time_steps_before_peak, 0)]
            values_after_peak = [np.exp(-3.0*((x-0)**3/((delta_time_steps_after_peak+1)**3)))\
                                 for x in np.arange(1, delta_time_steps_after_peak+1)]
        
        else:
            raise ValueError("curve_std_idx is not accepted, must select one of the following: " + \
                             "{-1 (or <= -1): 'exponential', 0: 'piecewise_affine', 1 (or >= 1): 'bell_curve'}")
        
        
        # Add the values to the series
        series.loc[dt_peak_effect_timestep] = 1 # Maximum effect = 1
        series.loc[dt_before_peak_start:dt_before_peak_end] = values_before_peak # Effect strictely before the peak
        series.loc[dt_after_peak_start:dt_after_peak_end] = values_after_peak # Effect strictely after the peak
        
    
    # Crop the final series to the dates of interest
    final_series = series.loc[dt_start:dt_end]
        
    # Plot
    if verbose >= 1:
        plt.figure(figsize=figsize)
        plt.plot(final_series)
        if verbose >= 2:
            plt.scatter(final_series.index, final_series.values)
        plt.grid()
        plt.title("Effect of {}".format(event_name))
        plt.xlabel("Time")
        plt.ylabel("Effect (between 0 and 1)")
        plt.show()
        
    return final_series
    

    
def generate_custom_events_series(
    events_series_base_params, 
    delta_days_peak_effect_supp_target, 
    delta_time_steps_before_peak_supp_target,
    delta_time_steps_after_peak_supp_target,
    curve_std_idx_supp_target,
    time_series_start, 
    time_series_end, 
    timestep_frequency, 
    events_table_path, 
    verbose, 
    figsize=(15, 4)
):
    
    """
    Function that generate the events effects stored throught pandas Series and pandas DataFrames
    
    Inputs:
        events_series_base_params: dict -> {keys = events names, values = {keys = params names, values = base values}}
        delta_days_peak_effect_supp_target: int -> A positive value (+k) means that the peak date 
            of the effect of the events, for this target, will arrive k days later than the base value 
            given for the event. The opposite for negative values
        delta_time_steps_before_peak_supp_target: int -> A positive value (+k) means that the duration 
            of the effect of the events before their peak, for this target, will last k time_steps longer 
            than the base value given for the event. The opposite for negative values.
        delta_time_steps_after_peak_supp_target: int -> A positive value (+k) means that the duration 
            of the effect of the events after their peak, for this target, will last k time_steps longer 
            than the base value given for the event. The opposite for negative values.
        curve_std_idx_supp_target: int -> A positive value (+k) means that the shape of the curve 
            around the peak of the events, for this target, will tend towards a "larger shape" 
            (ie with a larger natural standard deviation). The opposite for negative values.
        time_series_start: str -> Date of the beginning of the whole time series, format = '%Y/%m/%d'
        time_series_end: str -> Date of the end of the whole time series, format = '%Y/%m/%d'
        timestep_frequency: str -> "D" for daily, "W-MON" for weekly, "MS" for montly
        events_table_path: str -> path to the events table with names and dates
        verbose: int -> 0, 1 or 2, depending on the precision wanted for the printing / plots.
        
    Outputs:
        Dictionary of pandas Series of the events
        Dictionary of pandas DataFrames of the events
    
    """
    
    
    global_base_params_events = {
        "time_series_start": time_series_start,
        "time_series_end": time_series_end,
        "timestep_frequency": timestep_frequency, 
        "events_table_path": events_table_path,
        "verbose": verbose, 
        "figsize": figsize
    }
    
    series_events_dic = {}
    
    for event in events_series_base_params.keys():
        
        series_events_dic[event] = punctual_event_effect(
            event_name = event, 
            delta_days_peak_effect = events_series_base_params[event]["delta_days_peak_effect"] + delta_days_peak_effect_supp_target, 
            delta_time_steps_before_peak = events_series_base_params[event]["delta_time_steps_before_peak"] + delta_time_steps_before_peak_supp_target, 
            delta_time_steps_after_peak = events_series_base_params[event]["delta_time_steps_after_peak"] + delta_time_steps_after_peak_supp_target, 
            curve_std_idx = events_series_base_params[event]["curve_std_idx"] + curve_std_idx_supp_target,
            **global_base_params_events
        )

        
    dfs_events_dic = {
        event: pd.DataFrame({'ds': series_events_dic[event].index, event: series_events_dic[event].values}) 
        for event in series_events_dic.keys()
    }
    
    return series_events_dic, dfs_events_dic