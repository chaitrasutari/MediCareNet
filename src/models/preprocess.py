#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
import argparse


def main(file_path, output_path):
    # In[3]:


    # Load the cleaned dataset
    ## file_path = "../data/processed/cleaned_data.csv"
    df = pd.read_csv(file_path)

    # Show basic info and first few rows to confirm structure
    df.info(), df.head()


    # In[4]:


    # 1. Remove target leakage rows based on discharge_disposition_id
    leak_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(leak_ids)]


    # In[5]:


    # 2. Cap outliers for 'num_medications' at 95th percentile
    cap = df['num_medications'].quantile(0.95)
    df['num_medications'] = df['num_medications'].clip(upper=cap)


    # In[6]:


    # 3. Create binned feature 'number_inpatient_bin'
    df['number_inpatient_bin'] = pd.cut(
        df['number_inpatient'], 
        bins=[-1, 0, 2, 100], 
        labels=["0", "1-2", "3+"]
    )


    # In[7]:


    # 3. Create binned feature 'number_inpatient_bin'
    df['number_inpatient_bin'] = pd.cut(
        df['number_inpatient'], 
        bins=[-1, 0, 2, 100], 
        labels=["0", "1-2", "3+"]
    )


    # In[8]:


    # 4. Create composite feature 'total_visits'
    df['total_visits'] = df['number_inpatient'] + df['number_outpatient'] + df['number_emergency']


    # In[9]:


    df.head()


    # In[10]:


    # 5. Group rare categories (<1%) as 'Other' for categorical variables
    # Example for a generic categorical column 'some_cat_column'
    def group_rare_categories(series, threshold=0.01):
        freq = series.value_counts(normalize=True)
        rare_labels = freq[freq < threshold].index
        return series.apply(lambda x: 'Other' if x in rare_labels else x)


    # In[12]:


    print(df.columns)


    # In[13]:


    # Apply for relevant categorical columns (replace 'cat_cols' with your columns)
    cat_cols = ['race', 'diag_1_category', 'diag_2_category', 'diag_3_category']

    for col in cat_cols:
        if col in df.columns:
            df[col] = group_rare_categories(df[col])


    # In[14]:


    # 6. Create interaction features
    df['age_inpatient_interaction'] = df['age'].astype(str) + '_' + df['number_inpatient_bin'].astype(str)
    df['diabetes_insulin_combo'] = df['diabetesMed'].astype(str) + '_' + df['insulin'].astype(str)


    # In[20]:


    print(df.columns)


    # In[21]:


    print(df.head())


    # In[24]:


    print(df.columns.tolist())


    # In[25]:


    # List of numerical columns to scale
    num_cols = [
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'time_in_hospital',
        'number_inpatient',
        'number_outpatient',
        'number_emergency',
        'total_visits'
    ]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform only if all columns exist to avoid errors
    existing_num_cols = [col for col in num_cols if col in df.columns]

    df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])


    # In[27]:


    ## output_path = Path("../data/processed/feature_engineered_data.csv")


    # In[28]:


    output_path = Path(output_path)  # convert string to Path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to: {output_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw/cleaned CSV input")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    args = parser.parse_args()
    main(args.input, args.output)