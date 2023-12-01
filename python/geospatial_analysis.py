import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas


pd.options.display.max_rows = 10

df_divisions = geopandas.read_file("dataset\pd_divisions_datasd")
df_beats = geopandas.read_file("dataset\pd_beats_datasd")
df_stops = pd.read_csv('final_combined_dataset_v3.csv')


def construct_beats(input_beats):
    '''
    Construct the dataframe for beats.
    This function uses global dfs.
    
    :param input_beats: beats dataframe.
    :type input_beats: pandas dataframe.
    :returns: pandas dataframe
    '''
    
    #Construct the dataframe for beats. This is used to merge and get the divisions.
    beats_dataframe = input_beats[['serv', 'div']]
    beats_dataframe = beats_dataframe.groupby('serv').first()
    beats_dataframe = beats_dataframe[['div']]
    
    #clean the indices and datatypes
    beats_dataframe = beats_dataframe.reset_index()
    beats_dataframe['serv'] = beats_dataframe['serv'].astype(int)
    
    return beats_dataframe


construct_beats(df_beats)


def divsums(count, beats, divisions):
    '''
    This function returns the geopandas dataframe which has both geometric specifications from beats and count per location obtained from count.
    
    :type count: pandas dataframe.
    :param count: pandas dataframe, which contains the data which is used to corelate with beats.
    :type beats: pandas dataframe.
    :param beats: location dataframe.
    :type divisions: geopandas dataframe
    :param divisions: filename for source text 
    :returns: geopandas dataframe
    '''
    # clean the count dataframe
    count_clean = count.reset_index()
    count_clean['service_area'] = count_clean['service_area'].astype(int)
   
    # Joining the count to the beats dataframe
    count_wrt_serv = beats.join(count_clean.set_index('service_area'), on='serv').dropna()
    count_wrt_serv[['arrested']] = count_wrt_serv[['arrested']].fillna(0)
    
    # Joining the above dataframe with division dataframe
    count_wrt_serv = count_wrt_serv.groupby('div').sum()['arrested']
    return divisions.join(count_wrt_serv, on='div_num')


df_stops_serive_area = df_stops[['service_area', 'arrested']]
print(df_stops_serive_area)


def plot_map(input_geopandasdf):
    '''
    This function plots the geopandas dataframe. This dataframe contains information of count per division in geopandas dataframe.
    
    :type input_geopandasdf: geopandas dataframe
    :param input_geopandasdf: geopandas dataframee containing count inforamtion per division and geographical specifications of the division.
    '''
    ax = input_geopandasdf.plot(column='arrested', figsize=(25, 9), cmap='Blues', legend=True)
    input_geopandasdf.apply(lambda x: ax.annotate(text=x.div_name.capitalize(), xy=x.geometry.centroid.coords[0], ha='center'),axis=1)
    ax.set_axis_off()


def stop_data_for_plotting(stops, beats, divisions):
    '''
    This function cleans the dataframe data. returns a geopandas dataframe for stop count which can be plotted.
    
    :type stops: pandas dataframe.
    :param stops: filename for source text.
    :param input_beats: beats dataframe.
    :type input_beats: pandas dataframe.
    :type divisions: geopandas dataframe
    :param divisions: filename for source text 
    :returns: geopandas dataframe for stop count which can be plotted. 
    '''
    # Cleanup data
    df_stops_serive_area = stops[['service_area', 'arrested']]
    
    # Build counts
    # Amount of stops per service area
    stop_count = df_stops_serive_area.groupby('service_area').count()
    # Build beats
    beats_dataframe = construct_beats(beats)
    
    # Build divsums
    divsums_stop_dataframe = divsums(stop_count, beats_dataframe, divisions)
    return divsums_stop_dataframe


stop_data_for_plotting(df_stops, df_beats, df_divisions)

stop_data_div = stop_data_for_plotting(df_stops, df_beats, df_divisions)
plot_map(stop_data_div)


def arrest_data_for_plotting(stops, beats, divisions):
    '''
    This function cleans the dataframe data. returns a geopandas dataframe for arrest count which can be plotted.
    
    :type stops: pandas dataframe.
    :param stops: filename for source text.
    :param input_beats: beats dataframe.
    :type input_beats: pandas dataframe.
    :type divisions: geopandas dataframe
    :param divisions: filename for source text 
    :returns: geopandas dataframe for arrest count which can be plotted. 
    '''
    # Cleanup data
    df_stops_serive_area = stops[['service_area', 'arrested']]
    
    # Build counts
    # Amount of stops per service area
    stop_count = df_stops_serive_area.groupby('service_area').count()
    
    arrested_final_count = df_stops_serive_area.loc[df_stops_serive_area['arrested'] == 1].groupby('service_area').count()
    probability_of_arrest = (arrested_final_count/stop_count).fillna(0)
  
    # Build beats
    beats_dataframe = construct_beats(beats)
    
    # Build divsums
    divsums_arrest_dataframe = divsums(probability_of_arrest, beats_dataframe, divisions)
    return divsums_arrest_dataframe


arrest_data_div = arrest_data_for_plotting(df_stops, df_beats, df_divisions)
plot_map(arrest_data_div)


