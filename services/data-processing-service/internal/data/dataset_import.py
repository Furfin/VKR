import pandas as pd
import numpy as np
import dateparser
from pandas.api.types import (
    is_numeric_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype
)

COLUMNS_KEY = "columns"
SHAPE_KEY = "columns"

datetime_datatype = "datetime"
numeric_datatype = "numeric"
cathegorical_datatype = "cat"
text_datatype = "text"

def parse_date(date):
    parsed = dateparser.parse(date)
    if parsed == None:
        raise ValueError
    return parsed

def clean_dataframe(df):
    df = df.replace([
        '', ' ', 'NA', 'N/A', 'null', 'NaN', 'None', 
        '?', '-', '--', 'n/a', 'nan', 'NULL', np.inf, -np.inf
    ], np.nan)

    return df.dropna(how='any')

def is_valid_datetime(series: pd.Series) -> bool:
    if not pd.api.types.is_datetime64_any_dtype(series):
        return False
    
    min_year = series.dt.year.min()
    max_year = series.dt.year.max()
    return 1900 < min_year < 2100 and 1900 < max_year < 2100 and not series.name.lower().endswith(('_id', 'id'))

def detect_column_types(df: pd.DataFrame,data, categorical_threshold=0.1):
    type_info = {}
    
    for col in df.columns:
        if df[col].isna().all():
            type_info[col] = {'type': 'empty'}
            continue
        
        if is_valid_datetime(df[col]):
            type_info[col] = {'type': datetime_datatype}
            continue
            
        if is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
            type_info[col] = {'type': numeric_datatype}
            continue
            
        if is_string_dtype(df[col]):
            try:
               if not col.lower().endswith(('_id', 'id')) and df[col].head(100).apply(parse_date).all():
                   df[col] = df[col].apply(parse_date)
                   type_info[col] = {'type': datetime_datatype}
                   continue
            except Exception as error:
                print(error)
                pass
                
        if is_numeric_dtype(df[col]):
            type_info[col] = {'type': numeric_datatype}
            continue
            

        if is_string_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            if unique_ratio < categorical_threshold:
                type_info[col] = {'type': cathegorical_datatype, 
                                 'unique_values': df[col].nunique()}
            else:
                type_info[col] = {'type': text_datatype, 
                                 'dtype': str(df[col].dtype)}
            continue

        type_info[col] = {'type': 'unknown', 'dtype': str(df[col].dtype)}
    data[SHAPE_KEY] = df.shape
    data[COLUMNS_KEY] = type_info
    return (data, df)