import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("final_combined_dataset_v3.csv")
drop_cols = ['stop_id', 'search_details_id']
df = df.drop(columns = drop_cols)

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for relevant numerical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None
    """
    # Selecting relevant numerical columns for correlation heatmap
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Creating a correlation matrix
    correlation_matrix = df[numerical_columns].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

plot_correlation_heatmap(df)

def map_column_values(df, column_name, mapping):
    """
    Maps values in a DataFrame column based on a specified mapping.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to be mapped.
    - mapping (dict): A dictionary specifying the mapping of old values to new values.

    Returns:
    - DataFrame: The modified DataFrame.
    """
    df[column_name] = df[column_name].map(mapping)
    return df

search_details_type_mapping = {
    0: 'ActionTaken',
    1: 'ActionTakenOther',
    2: 'SearchBasis',
    3: 'SearchBasisOther',
    4: 'SearchType'
}

# Map the values in the 'search_details_type' column
df = map_column_values(df, 'search_details_type', search_details_type_mapping)

subject_race_mapping_switched = {
    0: 'Other Asian',
    1: 'Black',
    2: 'Chinese',
    3: 'Cambodian',
    4: 'Filipino',
    5: 'Guamanian',
    6: 'Hispanic',
    7: 'Indian',
    8: 'Japanese',
    9: 'Korean',
    10: 'Laotian',
    11: 'Other',
    11: 'Pacific Islander',
    12: 'Samoan',
    14: 'Hawaiian',
    15: 'Viet',
    16: 'White',
    17: 'Unknown',
    18: 'Asian indian'
}
# Map the values in the 'subject_race_mapping_switched' column
df = map_column_values(df, 'subject_race', subject_race_mapping_switched)

sex_map={
    0:'MALE',
    1: 'FEMALE',
    2: 'UNKNOWN'
}
# Map the values in the 'sex_map' column
df = map_column_values(df, 'subject_sex', sex_map)

resident_map={
    0:'NON-RESIDENT',
    1: 'RESIDENT'
}
# Map the values in the 'resident_map' column
df = map_column_values(df, 'sd_resident', resident_map)

def plot_boxplot_numerical_columns(df):
    """
    Plots a boxplot for relevant numerical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None
    """
    # Selecting relevant numerical columns for boxplot
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Plotting boxplot
    plt.figure(figsize=(15, 20))
    sns.boxplot(data=df[numerical_columns])
    plt.title('Boxplot for Numerical Columns')
    plt.show()

plot_boxplot_numerical_columns(df)

def plot_custom_chart(df, chart_type='barplot', x=None, y=None, hue=None, palette='viridis', title='', xlabel='', ylabel='', rotation=45, legend_loc='upper right', bbox_to_anchor=(1.2, 1)):
    """
    Plots various custom charts based on the specified parameters.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - chart_type (str, optional): Type of chart to be generated.
      Options: 'barplot', 'countplot', 'pie', 'multi_countplot', 'histplot'. Default is 'barplot'.
    - x (str, optional): The column on the x-axis.
    - y (str, optional): The column on the y-axis.
    - hue (str, optional): The column to differentiate by color.
    - palette (str or list, optional): Color palette for the plot. Default is 'viridis'.
    - title (str, optional): The title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - rotation (int, optional): Rotation angle for x-axis labels. Default is 45.
    - legend_loc (str, optional): Location of the legend. Default is 'upper right'.
    - bbox_to_anchor (tuple, optional): Adjustments to the legend box. Default is (1.2, 1).

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))

    if chart_type == 'barplot':
        sns.barplot(x=x, y=y, data=df, palette=palette)
    elif chart_type == 'countplot':
        sns.countplot(x=x, data=df, palette=palette)
    elif chart_type == 'pie':
        plt.pie(y, labels=x, autopct='%1.1f%%', startangle=140, colors=sns.color_palette(palette))
    elif chart_type == 'multi_countplot':
        sns.countplot(x=x, hue=hue, data=df, palette=palette)
        plt.legend(title=hue, loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
    elif chart_type == 'histplot':
        sns.histplot(df[x], bins=20, kde=True, color=palette)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha='right')

    plt.show()

# Check the distribution of stop causes
stop_cause_counts = df['search_details_type'].value_counts()

# Barplot for stop causes distribution
plot_custom_chart(df, chart_type='barplot', x=stop_cause_counts.index, y=stop_cause_counts.values, palette='viridis', title='Distribution of Stop Causes', xlabel='Stop Cause', ylabel='Count')

# Countplot for stop causes by service area
plot_custom_chart(df, chart_type='countplot', x='service_area', hue='stop_cause', palette='Set2', title='Distribution of Stop Causes by Service Area', xlabel='Service Area', ylabel='Count', legend_loc='upper left')

# Pie chart for top stop causes
top_stop_causes = stop_cause_counts.head(5)
plot_custom_chart(df, chart_type='pie', x=top_stop_causes.index, y=top_stop_causes.values, palette='pastel', title='Top Stop Causes')

# Barplot for distribution of stops across service areas
service_area_counts = df['service_area'].value_counts()
plot_custom_chart(df, chart_type='barplot', x=service_area_counts.index, y=service_area_counts.values, palette='viridis', title='Distribution of Stops Across Service Areas', xlabel='Service Areas', ylabel='Count')

# Barplot for top service areas with the most stops
top_service_areas = service_area_counts.head(10)  # Adjust the number of top service areas as needed
plot_custom_chart(df, chart_type='barplot', x=top_service_areas.index, y=top_service_areas.values, palette='viridis', title='Top Service Areas with the Most Stops', xlabel='Service Areas', ylabel='Count')

# Barplot for subject race distribution
subject_race_counts = df['subject_race'].value_counts()
plot_custom_chart(df, chart_type='barplot', x=subject_race_counts.index, y=subject_race_counts.values, palette='viridis', title='Subject Race Distribution', xlabel='Subject Race', ylabel='Count')

# Barplot for subject sex distribution
subject_sex_counts = df['subject_sex'].value_counts()
plot_custom_chart(df, chart_type='barplot', x=subject_sex_counts.index, y=subject_sex_counts.values, palette='viridis', title='Subject Sex Distribution', xlabel='Subject Sex', ylabel='Count')

# Analyze trends in service areas over time
df['date_stop'] = pd.to_datetime(df['date_stop'], format ='%d-%m-%Y')
df['year_month'] = df['date_stop'].dt.to_period('M')

# Multi-countplot for service areas distribution over time
plot_custom_chart(df, chart_type='multi_countplot', x='year_month', hue='service_area', palette='viridis', title='Service Areas Distribution Over Time', xlabel='Year-Month', ylabel='Count', legend_loc='upper right', bbox_to_anchor=(1.2, 1))
print(df.columns)

def plot_count_subplots(df, x, hue, title, ylabel, legend_title, layout_position, palette='viridis'):
    """
    Generates count subplots for different columns based on the specified parameters.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - x (str): The column on the x-axis.
    - hue (str): The column to differentiate by color.
    - title (str): The title of the subplot.
    - ylabel (str): Label for the y-axis.
    - legend_title (str): Title for the legend.
    - layout_position (tuple): Position of the subplot in the layout.
    - palette (str or list, optional): Color palette for the plot. Default is 'viridis'.

    Returns:
    - None
    """
    plt.subplot(*layout_position)
    sns.countplot(x=x, hue=hue, data=df, palette=palette)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.legend(title=legend_title, loc='upper right')

def set_up_plot_layout(rows, cols, figsize=(35, 12)):
    """
    Sets up the plot layout based on the specified number of rows and columns.

    Parameters:
    - rows (int): Number of rows in the layout.
    - cols (int): Number of columns in the layout.
    - figsize (tuple, optional): Size of the figure. Default is (35, 12).

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    plt.tight_layout()

set_up_plot_layout(2, 2)

# Plot count subplots for different columns
plot_count_subplots(df, 'subject_race', 'arrested', 'Arrests by Racial Group', 'Count', 'Arrested', (2, 2, 1))
plot_count_subplots(df, 'subject_race', 'searched', 'Searches by Racial Group', 'Count', 'Searched', (2, 2, 2))
plot_count_subplots(df, 'subject_race', 'stop_cause', 'Stop Causes by Racial Group', 'Count', 'Stop Cause', (2, 2, 3))
plot_count_subplots(df, 'subject_race', 'sd_resident', 'Residency Status by Racial Group', 'Count', 'Resident', (2, 2, 4))

# Show the plot
plt.show()

# Example usage for histogram plot
plot_custom_chart(df, chart_type='histplot', x='subject_age', palette='skyblue', title='Distribution of Subject Ages', xlabel='Subject Age', ylabel='Count')

# Create age bins
age_bins = [0, 18, 25, 35, 45, 55, 65, 75, 100]
age_labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']

df['age_group'] = pd.cut(df['subject_age'], bins=age_bins, labels=age_labels, right=False)

# Example usage for histogram plot
plot_custom_chart(df, chart_type='countplot', x='age_group', palette='viridis', title='Distribution of Subject Ages in Age Bins', xlabel='Age Group', ylabel='Count')