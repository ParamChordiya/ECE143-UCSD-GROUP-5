import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


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
    assert isinstance(replace_mapping, dict), "Input 'replace_mapping' must be a dictionary."
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


df= pd.read_csv("dataset\final_combined_dataset_v2.csv")
df


print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())


df = df.drop_duplicates()

unique_value_feature(df,'stop_cause')


stop_cause_mapping = {
    'NOT MARKED': 'Not Marked',
    'not marked': 'Not Marked',
    'Suspect Info': 'Suspect Info (I.S., Bulletin, Log)',
    '&Equipment Violation': 'Equipment Violation',
    'Personal Observ/Knowledge': 'Personal Knowledge/Informant',
    '&Moving Violation': 'Moving Violation',
    '&Radio Call/Citizen Contact': 'Radio Call/Citizen Contact',
    'no cause listed': 'Not Listed',
    'none listed': 'Not Listed',
    'Not Marked': 'Not Listed',
    'not noted': 'Not Listed',
    'not listed': 'Not Listed',
    'not marked  not marked': 'Not Listed',
    'NOT SPECIFIED': 'Not Listed',
    'No Cause Specified on a Card': 'Not Listed',
    'UNI, &County, H&&S Code': 'Muni, County, H&S Code',
    'MUNI, County, H&S Code': 'Muni, County, H&S Code',
    'Not Listed' : 'Not Listed/Other', 
    'Other': 'Not Listed/Other'
}
df = clean_features(df, stop_cause_mapping,'stop_cause')

unique_value_feature(df,'stop_cause')

unique_value_feature(df,'subject_race')

columns_to_drop = ['date_time']
df = df.drop(columns=columns_to_drop)

unique_value_feature(df,'sd_resident')

sd_resident_mapping = {
    "y":'Y',
    'n' : 'N',
}
df = clean_features(df, sd_resident_mapping,'sd_resident')


unique_value_feature(df,'sd_resident')

df = drop_rows_by_feature_value(df, 'sd_resident', ' ')


unique_value_feature(df,'sd_resident')

unique_value_feature(df,'arrested')

df = drop_rows_by_feature_value(df, 'arrested', ' ')
df = clean_features(df, sd_resident_mapping,'arrested')
unique_value_feature(df,'arrested')


unique_value_feature(df,'searched')


df = drop_rows_by_feature_value(df, 'searched', ' ')
df = clean_features(df, sd_resident_mapping,'searched')
unique_value_feature(df,'searched')


unique_subject_age_values = get_unique_values_as_list(df, 'subject_age')
print(unique_subject_age_values)


print(df.shape)

ages_to_drop =['0', '5', '230','2_', '2', '211', '8', '234', '185', '13', '12', '9', '153', '7', '6', '4', '221', '5_', '1', '4_', 
               '255', '224', '3', 'N', '204', '223', '228', '222',  '213', 'No Age', '125', '243', '399', 'f26', '3_', '11', '233', 
               '180', '173', '100', '119', '163', '212', '220', '145', '120', '226', '143']

for i in ages_to_drop:
    df = drop_rows_by_feature_value(df, 'subject_age', i)


print(df.shape)

unique_subject_age_values = get_unique_values_as_list(df, 'subject_age')
print(unique_subject_age_values)

df = clean_features(df, {'Unknown': 0},'service_area')


yes_no_mapping = {
    "Y": 1,
    'N' : 0,
}
df = clean_features(df, yes_no_mapping,'sd_resident')

df = clean_features(df, yes_no_mapping,'arrested')
df = clean_features(df, yes_no_mapping,'searched')


label_encoder = LabelEncoder()


df['subject_race'] = label_encoder.fit_transform(df['subject_race'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:")
print(label_mapping)


df['search_details_type'] = label_encoder.fit_transform(df['search_details_type'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:")
print(label_mapping)

df['stop_cause'] = label_encoder.fit_transform(df['stop_cause'])


label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:")
print(label_mapping)


print(df.describe().T)


df.to_csv('final_combined_dataset_v3.csv', index=False)


