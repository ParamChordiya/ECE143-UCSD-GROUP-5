# importing all required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import geopandas
import collections

# Data Cleaning and label encoding - preprocessing


def unique_value_feature(df, column):
    """
    Print the unique values and their counts for a given column in a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - column (str): The column name for which unique values are to be analyzed.

    Returns:
    None

    Example:
    unique_value_feature(my_dataframe, 'example_column')
    """
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    assert isinstance(column, str), "Input 'column' must be a string."

    unique_values = df[column].value_counts()
    print(f"Feature: {column}\n{unique_values}\n{'=' * 30}\n")


def clean_features(df, replace_mapping, feature):
    """
    Clean and standardize a specified feature column in the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - replace_mapping (dict): A dictionary specifying the values to be replaced.
    - feature (str): The feature/column name to be cleaned and standardized.

    Returns:
    pandas.DataFrame: The DataFrame with the specified feature column cleaned and standardized.
    """
    # Input parameter assertions
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    assert isinstance(
        replace_mapping, dict
    ), "Input 'replace_mapping' must be a dictionary."
    assert isinstance(feature, str), "Input 'feature' must be a string."

    for old_value, new_value in replace_mapping.items():
        df.loc[df[feature].isin([old_value]), feature] = new_value

    return df


def drop_rows_by_feature_value(df, feature, value):
    """
    Drop rows from a DataFrame where the specified feature has the specified value.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - feature (str): The feature/column name based on which rows will be dropped.
    - value: The value of the feature for which rows will be dropped.

    Returns:
    pandas.DataFrame: The DataFrame with specified rows dropped.
    """
    # Input parameter assertions
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    assert isinstance(feature, str), "Input 'feature' must be a string."

    return df[df[feature] != value]


def get_unique_values_as_list(df, feature):
    """
    Get all unique values of a specified feature in a DataFrame as a list.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - feature (str): The feature/column name for which unique values are to be retrieved.

    Returns:
    list: A list of unique values for the specified feature.
    """
    # Input parameter assertions
    assert isinstance(df, pd.DataFrame), "Input 'df' must be a pandas DataFrame."
    assert isinstance(feature, str), "Input 'feature' must be a string."

    unique_values = df[feature].unique().tolist()
    return unique_values


df = pd.read_csv("dataset\final_combined_dataset_v2.csv")
df


print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())


df = df.drop_duplicates()

unique_value_feature(df, "stop_cause")


stop_cause_mapping = {
    "NOT MARKED": "Not Marked",
    "not marked": "Not Marked",
    "Suspect Info": "Suspect Info (I.S., Bulletin, Log)",
    "&Equipment Violation": "Equipment Violation",
    "Personal Observ/Knowledge": "Personal Knowledge/Informant",
    "&Moving Violation": "Moving Violation",
    "&Radio Call/Citizen Contact": "Radio Call/Citizen Contact",
    "no cause listed": "Not Listed",
    "none listed": "Not Listed",
    "Not Marked": "Not Listed",
    "not noted": "Not Listed",
    "not listed": "Not Listed",
    "not marked  not marked": "Not Listed",
    "NOT SPECIFIED": "Not Listed",
    "No Cause Specified on a Card": "Not Listed",
    "UNI, &County, H&&S Code": "Muni, County, H&S Code",
    "MUNI, County, H&S Code": "Muni, County, H&S Code",
    "Not Listed": "Not Listed/Other",
    "Other": "Not Listed/Other",
}
df = clean_features(df, stop_cause_mapping, "stop_cause")

unique_value_feature(df, "stop_cause")

unique_value_feature(df, "subject_race")

columns_to_drop = ["date_time"]
df = df.drop(columns=columns_to_drop)

unique_value_feature(df, "sd_resident")

sd_resident_mapping = {
    "y": "Y",
    "n": "N",
}
df = clean_features(df, sd_resident_mapping, "sd_resident")


unique_value_feature(df, "sd_resident")

df = drop_rows_by_feature_value(df, "sd_resident", " ")


unique_value_feature(df, "sd_resident")

unique_value_feature(df, "arrested")

df = drop_rows_by_feature_value(df, "arrested", " ")
df = clean_features(df, sd_resident_mapping, "arrested")
unique_value_feature(df, "arrested")


unique_value_feature(df, "searched")


df = drop_rows_by_feature_value(df, "searched", " ")
df = clean_features(df, sd_resident_mapping, "searched")
unique_value_feature(df, "searched")


unique_subject_age_values = get_unique_values_as_list(df, "subject_age")
print(unique_subject_age_values)


print(df.shape)

ages_to_drop = [
    "0",
    "5",
    "230",
    "2_",
    "2",
    "211",
    "8",
    "234",
    "185",
    "13",
    "12",
    "9",
    "153",
    "7",
    "6",
    "4",
    "221",
    "5_",
    "1",
    "4_",
    "255",
    "224",
    "3",
    "N",
    "204",
    "223",
    "228",
    "222",
    "213",
    "No Age",
    "125",
    "243",
    "399",
    "f26",
    "3_",
    "11",
    "233",
    "180",
    "173",
    "100",
    "119",
    "163",
    "212",
    "220",
    "145",
    "120",
    "226",
    "143",
]

for i in ages_to_drop:
    df = drop_rows_by_feature_value(df, "subject_age", i)


print(df.shape)

unique_subject_age_values = get_unique_values_as_list(df, "subject_age")
print(unique_subject_age_values)

df = clean_features(df, {"Unknown": 0}, "service_area")


yes_no_mapping = {
    "Y": 1,
    "N": 0,
}
df = clean_features(df, yes_no_mapping, "sd_resident")

df = clean_features(df, yes_no_mapping, "arrested")
df = clean_features(df, yes_no_mapping, "searched")


label_encoder = LabelEncoder()


df["subject_race"] = label_encoder.fit_transform(df["subject_race"])

label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("Label Mapping:")
print(label_mapping)


df["search_details_type"] = label_encoder.fit_transform(df["search_details_type"])
label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("Label Mapping:")
print(label_mapping)
df["stop_cause"] = label_encoder.fit_transform(df["stop_cause"])
label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("Label Mapping:")
print(label_mapping)
print(df.describe().T)
df.to_csv("final_combined_dataset_v3.csv", index=False)


# Exploratory Data Analysis

df = pd.read_csv("final_combined_dataset_v3.csv")
drop_cols = ["stop_id", "search_details_id"]
df = df.drop(columns=drop_cols)


def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for relevant numerical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None
    """
    # Selecting relevant numerical columns for correlation heatmap
    numerical_columns = df.select_dtypes(include=["number"]).columns

    # Creating a correlation matrix
    correlation_matrix = df[numerical_columns].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
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
    0: "ActionTaken",
    1: "ActionTakenOther",
    2: "SearchBasis",
    3: "SearchBasisOther",
    4: "SearchType",
}

# Map the values in the 'search_details_type' column
df = map_column_values(df, "search_details_type", search_details_type_mapping)

subject_race_mapping_switched = {
    0: "Other Asian",
    1: "Black",
    2: "Chinese",
    3: "Cambodian",
    4: "Filipino",
    5: "Guamanian",
    6: "Hispanic",
    7: "Indian",
    8: "Japanese",
    9: "Korean",
    10: "Laotian",
    11: "Other",
    11: "Pacific Islander",
    12: "Samoan",
    14: "Hawaiian",
    15: "Viet",
    16: "White",
    17: "Unknown",
    18: "Asian indian",
}
# Map the values in the 'subject_race_mapping_switched' column
df = map_column_values(df, "subject_race", subject_race_mapping_switched)

sex_map = {0: "MALE", 1: "FEMALE", 2: "UNKNOWN"}
# Map the values in the 'sex_map' column
df = map_column_values(df, "subject_sex", sex_map)

resident_map = {0: "NON-RESIDENT", 1: "RESIDENT"}
# Map the values in the 'resident_map' column
df = map_column_values(df, "sd_resident", resident_map)


def plot_boxplot_numerical_columns(df):
    """
    Plots a boxplot for relevant numerical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None
    """
    # Selecting relevant numerical columns for boxplot
    numerical_columns = df.select_dtypes(include=["number"]).columns

    # Plotting boxplot
    plt.figure(figsize=(15, 20))
    sns.boxplot(data=df[numerical_columns])
    plt.title("Boxplot for Numerical Columns")
    plt.show()


plot_boxplot_numerical_columns(df)


def plot_custom_chart(
    df,
    chart_type="barplot",
    x=None,
    y=None,
    hue=None,
    palette="viridis",
    title="",
    xlabel="",
    ylabel="",
    rotation=45,
    legend_loc="upper right",
    bbox_to_anchor=(1.2, 1),
):
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

    if chart_type == "barplot":
        sns.barplot(x=x, y=y, data=df, palette=palette)
    elif chart_type == "countplot":
        sns.countplot(x=x, data=df, palette=palette)
    elif chart_type == "pie":
        plt.pie(
            y,
            labels=x,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette(palette),
        )
    elif chart_type == "multi_countplot":
        sns.countplot(x=x, hue=hue, data=df, palette=palette)
        plt.legend(title=hue, loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
    elif chart_type == "histplot":
        sns.histplot(df[x], bins=20, kde=True, color=palette)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")

    plt.show()


# Check the distribution of stop causes
stop_cause_counts = df["search_details_type"].value_counts()

# Barplot for stop causes distribution
plot_custom_chart(
    df,
    chart_type="barplot",
    x=stop_cause_counts.index,
    y=stop_cause_counts.values,
    palette="viridis",
    title="Distribution of Stop Causes",
    xlabel="Stop Cause",
    ylabel="Count",
)

# Countplot for stop causes by service area
plot_custom_chart(
    df,
    chart_type="countplot",
    x="service_area",
    hue="stop_cause",
    palette="Set2",
    title="Distribution of Stop Causes by Service Area",
    xlabel="Service Area",
    ylabel="Count",
    legend_loc="upper left",
)

# Pie chart for top stop causes
top_stop_causes = stop_cause_counts.head(5)
plot_custom_chart(
    df,
    chart_type="pie",
    x=top_stop_causes.index,
    y=top_stop_causes.values,
    palette="pastel",
    title="Top Stop Causes",
)

# Barplot for distribution of stops across service areas
service_area_counts = df["service_area"].value_counts()
plot_custom_chart(
    df,
    chart_type="barplot",
    x=service_area_counts.index,
    y=service_area_counts.values,
    palette="viridis",
    title="Distribution of Stops Across Service Areas",
    xlabel="Service Areas",
    ylabel="Count",
)

# Barplot for top service areas with the most stops
top_service_areas = service_area_counts.head(
    10
)  # Adjust the number of top service areas as needed
plot_custom_chart(
    df,
    chart_type="barplot",
    x=top_service_areas.index,
    y=top_service_areas.values,
    palette="viridis",
    title="Top Service Areas with the Most Stops",
    xlabel="Service Areas",
    ylabel="Count",
)

# Barplot for subject race distribution
subject_race_counts = df["subject_race"].value_counts()
plot_custom_chart(
    df,
    chart_type="barplot",
    x=subject_race_counts.index,
    y=subject_race_counts.values,
    palette="viridis",
    title="Subject Race Distribution",
    xlabel="Subject Race",
    ylabel="Count",
)

# Barplot for subject sex distribution
subject_sex_counts = df["subject_sex"].value_counts()
plot_custom_chart(
    df,
    chart_type="barplot",
    x=subject_sex_counts.index,
    y=subject_sex_counts.values,
    palette="viridis",
    title="Subject Sex Distribution",
    xlabel="Subject Sex",
    ylabel="Count",
)

# Analyze trends in service areas over time
df["date_stop"] = pd.to_datetime(df["date_stop"], format="%d-%m-%Y")
df["year_month"] = df["date_stop"].dt.to_period("M")

# Multi-countplot for service areas distribution over time
plot_custom_chart(
    df,
    chart_type="multi_countplot",
    x="year_month",
    hue="service_area",
    palette="viridis",
    title="Service Areas Distribution Over Time",
    xlabel="Year-Month",
    ylabel="Count",
    legend_loc="upper right",
    bbox_to_anchor=(1.2, 1),
)
print(df.columns)


def plot_count_subplots(
    df, x, hue, title, ylabel, legend_title, layout_position, palette="viridis"
):
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
    plt.legend(title=legend_title, loc="upper right")


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
plot_count_subplots(
    df,
    "subject_race",
    "arrested",
    "Arrests by Racial Group",
    "Count",
    "Arrested",
    (2, 2, 1),
)
plot_count_subplots(
    df,
    "subject_race",
    "searched",
    "Searches by Racial Group",
    "Count",
    "Searched",
    (2, 2, 2),
)
plot_count_subplots(
    df,
    "subject_race",
    "stop_cause",
    "Stop Causes by Racial Group",
    "Count",
    "Stop Cause",
    (2, 2, 3),
)
plot_count_subplots(
    df,
    "subject_race",
    "sd_resident",
    "Residency Status by Racial Group",
    "Count",
    "Resident",
    (2, 2, 4),
)

# Show the plot
plt.show()

# Example usage for histogram plot
plot_custom_chart(
    df,
    chart_type="histplot",
    x="subject_age",
    palette="skyblue",
    title="Distribution of Subject Ages",
    xlabel="Subject Age",
    ylabel="Count",
)

# Create age bins
age_bins = [0, 18, 25, 35, 45, 55, 65, 75, 100]
age_labels = ["0-18", "19-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]

df["age_group"] = pd.cut(
    df["subject_age"], bins=age_bins, labels=age_labels, right=False
)

# Example usage for histogram plot
plot_custom_chart(
    df,
    chart_type="countplot",
    x="age_group",
    palette="viridis",
    title="Distribution of Subject Ages in Age Bins",
    xlabel="Age Group",
    ylabel="Count",
)

# Temporal Analysis

df = pd.read_csv("final_combined_dataset_v3.csv")
print(df.columns)


def split_and_convert_time_columns(df, time_column="time_stop"):
    """
    Split the 'time_stop' column into 'hour' and 'minute' columns,
    and convert them to numeric type.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'time_stop' column.
    - time_column (str): The name of the column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with 'hour' and 'minute' columns added and converted.
    """
    # Split the 'time_stop' column into 'hour' and 'minute' columns
    df[["hour", "minute"]] = df[time_column].str.split(":", expand=True)

    # Convert the 'hour' and 'minute' columns to numeric type
    df[["hour", "minute"]] = df[["hour", "minute"]].apply(pd.to_numeric)

    return df


def extract_date_columns(df, date_column="date_stop"):
    """
    Convert the specified date column to datetime format
    and extract year, month, and day into separate columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the date column.
    - date_column (str): The name of the date column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with new year, month, and day columns added.
    """
    # Convert the specified date column to datetime format
    df[date_column] = pd.to_datetime(df[date_column], format="%d-%m-%Y")

    # Extract year, month, and day into separate columns
    df["year_stop"] = df[date_column].dt.year
    df["month_stop"] = df[date_column].dt.month
    df["day_stop"] = df[date_column].dt.day

    return df


def plot_data_bar(data, x, title, plot_type="count", trendline=False, color="red"):
    """
    Plots data based on the specified parameters.

    Parameters:
    - data (DataFrame): The DataFrame containing the data to be plotted.
    - x (str): The column on the x-axis.
    - title (str): The title of the plot.
    - plot_type (str, optional): Type of plot to be generated. Options: 'count', 'bar', 'regplot'. Default is 'count'.
    - trendline (bool, optional): If True, a trendline will be added to the plot. Default is False.
    - color (str, optional): The color of the trendline if added. Default is 'red'.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))

    if plot_type == "count":
        sns.countplot(x=x, data=data)
    elif plot_type == "bar":
        data[x] = data[x].replace({0: "No", 1: "Yes"})
        data_grouped = (
            data.groupby("year_stop")[x].value_counts(normalize=True).unstack()
        )
        data_grouped.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.xlabel("Year")
        plt.ylabel("Proportion of Stops")
    elif plot_type == "regplot":
        sns.regplot(
            x=data[x].value_counts().index,
            y=data[x].value_counts().values,
            scatter=True,
            color=color,
        )

    if trendline:
        sns.regplot(
            x=data[x].value_counts().index,
            y=data[x].value_counts().values,
            scatter=False,
            color=color,
        )

    plt.title(title)
    plt.show()


df = split_and_convert_time_columns(df)
df = extract_date_columns(df)


# Plotting the number of stops over time
plot_data_bar(df, "year_stop", "Number of Stops Over the Years", plot_type="count")

# Plotting arrests over time
plot_data_bar(df, "arrested", "Arrests Over the Years", plot_type="bar")

# Plotting searches over time
plot_data_bar(df, "searched", "Searches Over the Years", plot_type="bar")

# Plotting the distribution of stops by month with a trendline
plot_data_bar(
    df,
    "month_stop",
    "Distribution of Stops by Month",
    plot_type="count",
    trendline=True,
)


def process_date_column(
    df, date_column="date_stop", new_column="day_of_week", abbreviated=False
):
    """
    Converts a date column to datetime format and extracts day names into a new column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - date_column (str, optional): The name of the date column. Default is 'date_stop'.
    - new_column (str, optional): The name of the new column to store day names. Default is 'day_of_week'.
    - abbreviated (bool, optional): If True, uses abbreviated day names (e.g., Mon, Tue). Default is False.

    Returns:
    - DataFrame: The modified DataFrame.
    """
    df[date_column] = pd.to_datetime(df[date_column])

    if abbreviated:
        df[new_column] = df[date_column].dt.strftime("%a")
    else:
        df[new_column] = df[date_column].dt.day_name()

    return df


df = process_date_column(df)
# or with abbreviated day names
# processed_df = process_date_column(df, abbreviated=True)

print(np.unique(df["day_of_week"]))


def plot_data(
    data, x, title, plot_type="count", trendline=False, color="red", order=None
):
    """
    Plots data based on the specified parameters.

    Parameters:
    - data (DataFrame): The DataFrame containing the data to be plotted.
    - x (str): The column on the x-axis.
    - title (str): The title of the plot.
    - plot_type (str, optional): Type of plot to be generated.
      Options: 'countplot', 'heatmap', 'lineplot', 'barplot', 'violinplot', 'scatterplot'. Default is 'countplot'.
    - trendline (bool, optional): If True, a trendline will be added to the plot. Default is False.
    - color (str, optional): The color of the trendline if added. Default is 'red'.
    - order (list, optional): Order of categories for categorical plots. Default is None.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))

    if plot_type == "countplot":
        sns.countplot(x=x, data=data, order=order)
    elif plot_type == "heatmap":
        heatmap_data = data.groupby(["day_of_week", "hour"]).size().unstack()
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt="g", linewidths=0.5)
        plt.xlabel("Hour")
        plt.ylabel("Day of the Week")
    elif plot_type == "lineplot":
        monthly_stops = data.groupby(["year_stop", "month_stop"]).size()
        monthly_stops.plot(marker="o")
        plt.xticks(rotation=45)
        plt.xlabel("Year-Month")
        plt.ylabel("Number of Stops")
    elif plot_type == "barplot":
        searches_by_day = data.groupby("day_of_week")["searched"].mean()
        searches_by_day.sort_values().plot(kind="bar")
        plt.xlabel("Day of the Week")
        plt.ylabel("Proportion of Searches")
    elif plot_type == "violinplot":
        sns.violinplot(
            x="day_of_week", y="hour", data=data, inner="quartile", palette="viridis"
        )
        plt.xlabel("Day of the Week")
        plt.ylabel("Hour of the Day")
    elif plot_type == "scatterplot":
        monthly_searches = data.groupby(["year_stop", "month_stop"])["searched"].mean()
        monthly_searches.plot(marker="o", color=color)
        plt.xticks(rotation=45)
        plt.xlabel("Year-Month")
        plt.ylabel("Proportion of Searches")

    if trendline:
        sns.regplot(
            x=data[x].value_counts().index,
            y=data[x].value_counts().values,
            scatter=False,
            color=color,
        )

    plt.title(title)
    plt.show()


# Countplot of stops by day of the week
plot_data(
    df,
    "day_of_week",
    "Stops by Day of the Week",
    plot_type="countplot",
    order=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
# Heatmap of stops by hour and day
plot_data(df, "", "Heatmap of Stops by Hour and Day of the Week", plot_type="heatmap")
# Trends in stops over months
plot_data(df, "", "Trends in Stops Over Months", plot_type="lineplot")
# Violin plot of stops by day and hour
plot_data(df, "", "Violin Plot of Stops by Day and Hour", plot_type="violinplot")


# Geospatial Analysis
pd.options.display.max_rows = 10

df_divisions = geopandas.read_file("dataset\pd_divisions_datasd")
df_beats = geopandas.read_file("dataset\pd_beats_datasd")
df_stops = pd.read_csv("final_combined_dataset_v3.csv")


def construct_beats(input_beats):
    """
    Construct the dataframe for beats.
    This function uses global dfs.

    :param input_beats: beats dataframe.
    :type input_beats: pandas dataframe.
    :returns: pandas dataframe
    """

    # Construct the dataframe for beats. This is used to merge and get the divisions.
    beats_dataframe = input_beats[["serv", "div"]]
    beats_dataframe = beats_dataframe.groupby("serv").first()
    beats_dataframe = beats_dataframe[["div"]]

    # clean the indices and datatypes
    beats_dataframe = beats_dataframe.reset_index()
    beats_dataframe["serv"] = beats_dataframe["serv"].astype(int)

    return beats_dataframe


construct_beats(df_beats)


def divsums(count, beats, divisions):
    """
    This function returns the geopandas dataframe which has both geometric specifications from beats and count per location obtained from count.

    :type count: pandas dataframe.
    :param count: pandas dataframe, which contains the data which is used to corelate with beats.
    :type beats: pandas dataframe.
    :param beats: location dataframe.
    :type divisions: geopandas dataframe
    :param divisions: filename for source text
    :returns: geopandas dataframe
    """
    # clean the count dataframe
    count_clean = count.reset_index()
    count_clean["service_area"] = count_clean["service_area"].astype(int)

    # Joining the count to the beats dataframe
    count_wrt_serv = beats.join(
        count_clean.set_index("service_area"), on="serv"
    ).dropna()
    count_wrt_serv[["arrested"]] = count_wrt_serv[["arrested"]].fillna(0)

    # Joining the above dataframe with division dataframe
    count_wrt_serv = count_wrt_serv.groupby("div").sum()["arrested"]
    return divisions.join(count_wrt_serv, on="div_num")


df_stops_serive_area = df_stops[["service_area", "arrested"]]
print(df_stops_serive_area)


def plot_map(input_geopandasdf):
    """
    This function plots the geopandas dataframe. This dataframe contains information of count per division in geopandas dataframe.

    :type input_geopandasdf: geopandas dataframe
    :param input_geopandasdf: geopandas dataframee containing count inforamtion per division and geographical specifications of the division.
    """
    ax = input_geopandasdf.plot(
        column="arrested", figsize=(25, 9), cmap="Blues", legend=True
    )
    input_geopandasdf.apply(
        lambda x: ax.annotate(
            text=x.div_name.capitalize(), xy=x.geometry.centroid.coords[0], ha="center"
        ),
        axis=1,
    )
    ax.set_axis_off()


def stop_data_for_plotting(stops, beats, divisions):
    """
    This function cleans the dataframe data. returns a geopandas dataframe for stop count which can be plotted.

    :type stops: pandas dataframe.
    :param stops: filename for source text.
    :param input_beats: beats dataframe.
    :type input_beats: pandas dataframe.
    :type divisions: geopandas dataframe
    :param divisions: filename for source text
    :returns: geopandas dataframe for stop count which can be plotted.
    """
    # Cleanup data
    df_stops_serive_area = stops[["service_area", "arrested"]]

    # Build counts
    # Amount of stops per service area
    stop_count = df_stops_serive_area.groupby("service_area").count()
    # Build beats
    beats_dataframe = construct_beats(beats)

    # Build divsums
    divsums_stop_dataframe = divsums(stop_count, beats_dataframe, divisions)
    return divsums_stop_dataframe


stop_data_for_plotting(df_stops, df_beats, df_divisions)

stop_data_div = stop_data_for_plotting(df_stops, df_beats, df_divisions)
plot_map(stop_data_div)


def arrest_data_for_plotting(stops, beats, divisions):
    """
    This function cleans the dataframe data. returns a geopandas dataframe for arrest count which can be plotted.

    :type stops: pandas dataframe.
    :param stops: filename for source text.
    :param input_beats: beats dataframe.
    :type input_beats: pandas dataframe.
    :type divisions: geopandas dataframe
    :param divisions: filename for source text
    :returns: geopandas dataframe for arrest count which can be plotted.
    """
    # Cleanup data
    df_stops_serive_area = stops[["service_area", "arrested"]]

    # Build counts
    # Amount of stops per service area
    stop_count = df_stops_serive_area.groupby("service_area").count()

    arrested_final_count = (
        df_stops_serive_area.loc[df_stops_serive_area["arrested"] == 1]
        .groupby("service_area")
        .count()
    )
    probability_of_arrest = (arrested_final_count / stop_count).fillna(0)

    # Build beats
    beats_dataframe = construct_beats(beats)

    # Build divsums
    divsums_arrest_dataframe = divsums(
        probability_of_arrest, beats_dataframe, divisions
    )
    return divsums_arrest_dataframe


arrest_data_div = arrest_data_for_plotting(df_stops, df_beats, df_divisions)
plot_map(arrest_data_div)


# Age vs Reason action
data_2014 = pd.read_csv("../dataset/vehicle_stops_2014_datasd.csv")
data_2015 = pd.read_csv("../dataset/vehicle_stops_2015_datasd.csv")
data_2016 = pd.read_csv("../dataset/vehicle_stops_2016_datasd.csv")
data_2017 = pd.read_csv("../dataset/vehicle_stops_2017_datasd.csv")
data_combined = pd.read_csv("../dataset/final_combined_dataset_v2.csv")


def create_race_dictionary(csv_path):
    """
    Creates a race dictionary based on the specified CSV file.

    Parameters:
    - csv_path (str): The path to the CSV file containing race codes and descriptions.

    Returns:
    - race_dict (defaultdict): The race dictionary.
    """
    race_csv = pd.read_csv(csv_path)
    race_dict = collections.defaultdict(str)

    for i in range(len(race_csv)):
        race_dict[race_csv["Race Code"][i]] = race_csv["Description"][i]

    return race_dict


race_dictionary = create_race_dictionary("../dataset/vehicle_stops_race_codes.csv")

### Total unique violations
print("Total unique violations:-", len(data_combined["stop_cause"].unique()))


# ### Stop Causes

data_combined["stop_cause"].value_counts()

# ### Stop by Race

race_list = list(data_combined["subject_race"].value_counts())

data_combined[data_combined["subject_race"] == "U"]["subject_age"]


def filter_age_range(data, age_range=(1, 120)):
    """
    Filters a DataFrame based on a specified age range.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - age_range (tuple, optional): The age range to keep. Default is (1, 120).

    Returns:
    - filtered_data (DataFrame): The filtered DataFrame.
    """
    age_filter = data["subject_age"].between(*age_range)
    filtered_data = data[age_filter]

    return filtered_data


filtered_data_combined = filter_age_range(data_combined)

data_combined["subject_age"] = data_combined["subject_age"].astype(int)


# ### Pullover probability vs age for every Race


def plot_race_kde(data, race_dict, race_limited=None, age_range=(15, 75)):
    """
    Plots Kernel Density Estimates (KDE) for age distribution based on specified races.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - race_dict (dict): The dictionary mapping race codes to descriptions.
    - race_limited (list, optional): List of race codes to include. Default is None.
    - age_range (tuple, optional): The age range to visualize. Default is (15, 75).

    Returns:
    - None
    """
    if race_limited is None:
        race_limited = data["subject_race"].unique()

    fig, ax = plt.subplots()
    ax.set_xlim(*age_range)

    for race in race_limited:
        s = data[data["subject_race"] == race]["subject_age"]
        s.plot.kde(ax=ax, label=race_dict.get(race, race))

    ax.legend()
    plt.show()


plot_race_kde(
    data_combined,
    race_dictionary,
    race_limited=["B", "C", "I", "W", "Z"],
    age_range=(15, 75),
)


# ### Pullover probability vs age for Asian Indians and Chinese


def plot_race_kde(data, race_dict, race_limited=None, age_range=(15, 75)):
    """
    Plots Kernel Density Estimates (KDE) for age distribution based on specified races.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - race_dict (dict): The dictionary mapping race codes to descriptions.
    - race_limited (list, optional): List of race codes to include. Default is None.
    - age_range (tuple, optional): The age range to visualize. Default is (15, 75).

    Returns:
    - None
    """
    if race_limited is None:
        race_limited = data["subject_race"].unique()

    fig, ax = plt.subplots()
    ax.set_xlim(*age_range)

    for race in race_limited:
        s = data[data["subject_race"] == race]["subject_age"]
        s.plot.kde(ax=ax, label=race_dict.get(race, race))

    ax.legend()
    plt.show()


plot_race_kde(
    data_combined, race_dictionary, race_limited=["C", "Z"], age_range=(15, 75)
)


# ### What does this graph tells us?
# #### Observe that the peak values for the two graphs are different. The chines peak comes around the age of 26/17 whereas the Indian Peak comes around the age of 32 33. We can infer that an Indian guy will have better probability of possessing a car in his 30's as compared to his 20's which makes sense, as more pullover per race at any age indicates more cars per race at any age
#
#
# ### What is the significance of this information?
# ###


def plot_stop_cause_kde(data, stop_causes, age_range=(15, 75)):
    """
    Plots Kernel Density Estimates (KDE) for age distribution based on specified stop causes.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - stop_causes (list): List of stop causes to include.
    - age_range (tuple, optional): The age range to visualize. Default is (15, 75).

    Returns:
    - None
    """
    fig, ax = plt.subplots()
    ax.set_xlim(*age_range)

    for cause in data["stop_cause"].unique():
        if cause not in stop_causes:
            continue
        s = data[data["stop_cause"] == cause]["subject_age"]
        s.plot.kde(ax=ax, label=cause)

    ax.legend()
    plt.show()


stop_reasons_to_plot = [
    "Moving Violation",
    "Equipment Violation",
    "Radio Call/Citizen Contact",
]
plot_stop_cause_kde(data_combined, stop_reasons_to_plot, age_range=(15, 75))
