import pandas as pd
import os
from datetime import datetime
import re
import numpy as np
# import kagglehub

missing_month_pattern = r'^\d{4}\.00\.\d{2}(?:\.[a-z])?(?:\.R)?$' # regex that identifies missing month value in 'Date' column
missing_day_pattern = r'^\d{4}\.\d{2}\.00(?:\.[a-z])?(?:\.R)?$' # regex that identifies missing day value in 'Date' column

def count_date_matching_pattern(df, pattern, case_column='Case Number'):
    """
    This is mostly LLM generated code which I modified
    
    Count the number of case numbers with unknown months (mm = 00)
    in the format yyyy.mm.dd(.x)(.R)
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    case_column (str): Name of the column containing case numbers
    
    Returns:
    int: Count of unknown months
    """
    count = 0
    matching_cases = []
    
    for case in df[case_column].dropna():
        case_str = str(case).strip()
        
        # Pattern for yyyy.00.dd format with optional suffixes
        # This looks for month = 00 in the date pattern
        
        
        if re.match(pattern, case_str):
            count += 1
            matching_cases.append(case_str)
    
    return count, matching_cases

def parse_case_number_to_datetime(case_number):
    """
    Convert case number string to datetime object based on various formats.
    
    Parameters:
    case_number (str): The case number string to parse
    
    Returns:
    datetime or NaT: Parsed datetime or NaT if parsing fails. In cases where the month but not day is known, it returns the 1st day of the month but where either month or year is unkown, NaT is returned.
    """
    if pd.isna(case_number) or case_number is None:
        return pd.NaT
    
    # Convert to string and strip whitespace
    case_str = str(case_number).strip()
    
    # Handle empty strings
    if case_str == '':
        return pd.NaT
    
    # Pattern 1: yyyy.mm.dd format with optional .x and .R suffixes
    # Examples: 2018.05.13, 2018.05.13.b, 2018.05.13.R, 2018.05.13.b.R
    pattern1 = r'^(\d{4})\.(\d{2})\.(\d{2})(?:\.([a-z]))?(?:\.(R))?$'
    match = re.match(pattern1, case_str)
    
    if match:
        year, month, day, letter, reported = match.groups()
        year = int(year)
        month = int(month)
        day = int(day)
        
        # Handle unknown month or day (00)
        if month == 0:
            return pd.NaT # unknown month is only 300/6302, <5% so discard
        if day == 0:
            day = 1    # keeping known months but unknown day seems more important since 742/6302, >10% are affected. this won't negatively affect analysis anyway
            
        try:
            return datetime(year, month, day)
        except ValueError:
            return pd.NaT
    
    # Pattern 2: 0000.yyyy (BC years)
    pattern2 = r'^0000\.(\d{4})$'
    match = re.match(pattern2, case_str)
    
    if match:
        year = int(match.group(1))
        return pd.NaT  # BC dates can't be directly represented in pandas datetime
    
    # Pattern 3: ND-nnnn or ND.nnnn (Undated incidents)
    pattern3 = r'^ND[.-](\d{4})$'
    match = re.match(pattern3, case_str)
    
    if match:
        return pd.NaT

    # If no patterns match, return NaT
    return pd.NaT


# # download dataset from kaggle
# path = kagglehub.dataset_download("felipeesc/shark-attack-dataset")
# print("Path to dataset files:", path)

# dataset already downloaded so i saved the path to it in a file path.txt
print("Loading kaggle shark attack dataset from: felipeesc/shark-attack-dataset...")
with open("path.txt", "r") as f:
    path = f.read()
attacks_df = pd.read_csv(path+'/attacks.csv', encoding_errors='ignore')

print()
print("Original data:")
print(attacks_df.head())

print()
print("Original number of columns before cleaning:", len(attacks_df))

print("\n\n\nSTARTING CLEANING...\n\n\n")

# drop duplicate or redundant columns. keep only one of the needed ones
print("Dropping redundant columns...")
attacks_df.drop(axis=1, labels=['Case Number.1','Case Number.2', 'Year'], inplace=True)
attacks_df.drop(axis=1, labels=['Unnamed: 22', 'Unnamed: 23'], inplace=True)
attacks_df.drop(axis=1, labels=['href', 'href formula'], inplace=True) # drop them, keep 'pdf'

print("Dropping sparse rows(>70% null values)...")
attacks_df.dropna(axis=0,thresh=7, inplace=True) # drop values with at least threshold 7 i.e. less than 40% of fields are non-na values


print("Standardising data types...")
# remove leading and trailing spaces
attacks_df = attacks_df.map(lambda x: x.strip() if isinstance(x, str) else x)

# convert age to number rather than string
attacks_df['Age'] = pd.to_numeric(attacks_df['Age'], errors='coerce', downcast='integer')

print("Missing values in 'Date' column:")
print("Unknown days:", count_date_matching_pattern(attacks_df, missing_day_pattern)[0])
print("Unknown months:", count_date_matching_pattern(attacks_df, missing_month_pattern)[0])
print("Cleaning up missing date data...")
attacks_df.rename(columns={"Date":"Information on date of occurence"}, inplace=True)
attacks_df['Date'] = attacks_df['Case Number'].apply(parse_case_number_to_datetime) # deals with unstandardised date data by creating standardised new Data (datetime) column
attacks_df.drop(axis=1, labels=['Case Number'], inplace=True) # drop Case Number value since it is no longer useful
print(len(attacks_df[attacks_df['Date'].notna()]), "non-NaT dates of", len(attacks_df), "total")

# replacing non-standardised values with standardised ones
print("Standardising data values...")

column = 'Type'
# in 'Type' column, Boat, Boating and Boatomg are the same thing and should be treated as such. Invalid should be represented as NaN
attacks_df.loc[(attacks_df[column] =='Boat'), column] = 'Boating' 
attacks_df.loc[(attacks_df[column] =='Boatomg'), column] = 'Boating'
attacks_df.loc[(attacks_df[column] =='Invalid'), column] = np.NaN

column = 'Country'
attacks_df.loc[(attacks_df[column] =='CEYLON'), column] = 'SRI LANKA'
attacks_df.loc[(attacks_df[column] =='CEYLON (SRI LANKA)'), column] = 'SRI LANKA'
attacks_df.loc[(attacks_df[column] =='Fiji'), column] = 'FIJI'
attacks_df.loc[(attacks_df[column] =='MALDIVE ISLANDS'), column] = 'MALDIVES'
attacks_df.loc[(attacks_df[column] =='ST. MAARTIN'), column] = 'ST. MARTIN'
attacks_df.loc[(attacks_df[column] =='Seychelles'), column] = 'SEYCHELLES'
attacks_df.loc[(attacks_df[column] =='Sierra Leone'), column] = "SIERRA LEONE"
attacks_df.loc[(attacks_df[column] =='UNITED ARAB EMIRATES (UAE)'), column] = 'UNITED ARAB EMIRATES'
attacks_df.loc[(attacks_df[column] =='REUNION'), column] = 'REUNION ISLAND'
attacks_df[column] = attacks_df[column].str.upper() # capitalise all countries

column = 'Sex '
attacks_df.loc[(attacks_df[column] =='lli'), column] = np.NaN
attacks_df.loc[(attacks_df[column] =='.'), column] = np.NaN
attacks_df.loc[(attacks_df[column] =='N'), column] = 'M'

column = 'Fatal (Y/N)'
attacks_df.loc[(attacks_df[column] =='y'), column] = "Y"
attacks_df.loc[(attacks_df[column] =='UNKNOWN'), column] = np.NaN
attacks_df.loc[(attacks_df[column] =='M'), column] = 'N'
attacks_df.loc[(attacks_df[column] =='2017'), column] = np.NaN

print("\n\n\nALL DONE!")
print()
print("Final dataset:")
print(attacks_df.head())