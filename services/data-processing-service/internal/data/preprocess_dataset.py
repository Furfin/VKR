NUM_STRAT_ROBUST = "RobustScaler"
NUM_STRAT_MINMAX = "MinMaxScaler" 
NUM_STRAT_STANDART = "StandardScaler"
NUM_STRAT_LOG = "LogTransform"
NUM_STRAT_NORMAL = "Normalize"
NUM_STRAT_Q3 = "3Sigm"
NUM_STRAT_NORMALM = "NormalizeMedian"
NUM_STRAT_L2 = "L2"
NUM_STRAT_NONE = "None"

CAT_STRAT_FREQ = "Frequency Encoding"
CAT_STRAT_ONEHOT = "One-Hot Encoding"
CAT_STRAT_LABEL = "Label Encoding"
CAT_STRAT_BINARY = "Binary Encoding"

TEXT_STRAT_TFIDF = "TF-IDF"
TEXT_STRAT_BOW = "Bag of words"
TEXT_STRAT_NGRAM = "NGrams"
TEXT_STRAT_LEXICAL = "Lexical"

DATETIME_STRAT_SPLIT = "Split datetime"
DATETIME_STRAT_TIMESTAMP = "Timestamp"
DATETIME_STRAT_WEEKDAY = "Extract weekday"

import pandas as pd
from typing import Callable, Union
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, LabelEncoder, OneHotEncoder
from category_encoders import BinaryEncoder
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import dateparser
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def parse_date(date):
    parsed = dateparser.parse(str(date))
    if parsed == None:
        raise ValueError
    return parsed

# ЧИСЛОВЫЕ

def none(column: pd.Series) -> pd.Series:
    return column.astype("int64")

def scale_standard(column: pd.Series) -> pd.Series:
    """Стандартизация (Z-score нормализация)"""
    scaler = StandardScaler()
    return pd.Series(
        scaler.fit_transform(column.values.reshape(-1, 1)).flatten(),
        index=column.index,
        name=column.name
    )

def scale_minmax(column: pd.Series) -> pd.Series:
    """Масштабирование в диапазон [0, 1]"""
    scaler = MinMaxScaler()
    return pd.Series(
        scaler.fit_transform(column.values.reshape(-1, 1)).flatten(),
        index=column.index,
        name=column.name
    )

def scale_robust(column: pd.Series) -> pd.Series:
    """Масштабирование устойчивое к выбросам"""
    scaler = RobustScaler()
    return pd.Series(
        scaler.fit_transform(column.values.reshape(-1, 1)).flatten(),
        index=column.index,
        name=column.name
    )

def transform_log(column: pd.Series) -> pd.Series:
    """Логарифмическое преобразование"""
    return pd.Series(
        np.log1p(column),
        index=column.index,
        name=column.name
    )

def transform_quantile(column: pd.Series) -> pd.Series:
    """Квантильное преобразование к нормальному распределению"""
    qt = QuantileTransformer(output_distribution='normal')
    return pd.Series(
        qt.fit_transform(column.values.reshape(-1, 1)).flatten(),
        index=column.index,
        name=column.name
    )

def clip_outliers(column: pd.Series, n_sigmas: float = 3) -> pd.Series:
    """Обрезка выбросов по правилу n сигм"""
    mean, std = column.mean(), column.std()
    lower = mean - n_sigmas * std
    upper = mean + n_sigmas * std
    return column.clip(lower, upper)

def normalize_median(column: pd.Series) -> pd.Series:
    """Нормализация относительно медианы"""
    return column / column.median()

def scale_unit_length(column: pd.Series) -> pd.Series:
    """Нормализация к единичной длине вектора (L2 норма)"""
    return column / np.linalg.norm(column)

# КАТЕГОРИАЛЬНЫЕ

def encode_onehot(column: pd.Series) -> pd.DataFrame:
    """One-Hot Encoding"""
    return pd.get_dummies(column, dtype=int)

def encode_label(column: pd.Series) -> pd.Series:
    """Label Encoding (целочисленные метки)"""
    encoder = LabelEncoder()
    return pd.Series(
        encoder.fit_transform(column),
        index=column.index,
        name=column.name
    )

def encode_frequency(column: pd.Series) -> pd.Series:
    """Частота встречаемости категорий"""
    freq = column.value_counts(normalize=True)
    return pd.Series(
        column.map(freq),
        index=column.index,
        name=f"{column.name}_freq"
    )

def encode_binary(column: pd.Series) -> pd.DataFrame:
    """Binary Encoding (для высокой кардинальности)"""
    encoder = BinaryEncoder()
    return encoder.fit_transform(column)

# ТЕКСТ

def preprocess_text(text):
    """Базовый препроцессинг текста"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

def text_to_bow(column: pd.Series, max_features: int = 1000) -> pd.DataFrame:
    """Bag-of-Words представление"""
    vectorizer = CountVectorizer(
        max_features=max_features,
        tokenizer=preprocess_text
    )
    bow = vectorizer.fit_transform(column)
    return pd.DataFrame(
        bow.toarray(),
        columns=[f"bow_{f}" for f in vectorizer.get_feature_names_out()],
        index=column.index
    )

def text_to_tfidf(column: pd.Series, max_features: int = 1000) -> pd.DataFrame:
    """TF-IDF представление"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        tokenizer=preprocess_text
    )
    tfidf = vectorizer.fit_transform(column)
    return pd.DataFrame(
        tfidf.toarray(),
        columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()],
        index=column.index
    )

def text_to_ngrams(column: pd.Series, ngram_range: tuple = (1, 2), max_features: int = 1000) -> pd.DataFrame:
    """N-gram представление"""
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        tokenizer=preprocess_text
    )
    ngrams = vectorizer.fit_transform(column)
    return pd.DataFrame(
        ngrams.toarray(),
        columns=[f"ngram_{f}" for f in vectorizer.get_feature_names_out()],
        index=column.index
    )

def text_to_lexical(column: pd.Series) -> pd.DataFrame:
    """Лексические характеристики текста"""
    def extract_features(text):
        tokens = preprocess_text(text)
        return {
            'char_count': len(text),
            'word_count': len(tokens),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens))/len(tokens) if tokens else 0
        }
    
    return pd.DataFrame(
        column.apply(extract_features).tolist(),
        index=column.index
    )

# ВРЕМЕННЫЕ МЕТКИ

def split_datetime(column: pd.Series) -> pd.DataFrame:
    column = column.apply(parse_date)
    return pd.DataFrame({
        'year': column.dt.year,
        'month': column.dt.month,
        'day': column.dt.day,
        'hour': column.dt.hour,
        'minute': column.dt.minute
    }, index=column.index)

def datetime_to_timestamp(column: pd.Series) -> pd.Series:
    column = column.apply(parse_date)
    return column.astype('int64') // 10**9  # Наносекунды -> секунды

def extract_weekday(column: pd.Series) -> pd.Series:
    column = column.apply(parse_date)
    return column.dt.weekday  # 0-6 (пн-вс)

strats = {
    NUM_STRAT_ROBUST: scale_robust,
    NUM_STRAT_MINMAX: scale_minmax,
    NUM_STRAT_STANDART: scale_standard,
    NUM_STRAT_LOG: transform_log,
    NUM_STRAT_NORMAL: transform_quantile,
    NUM_STRAT_Q3: clip_outliers,
    NUM_STRAT_NORMALM: normalize_median,
    NUM_STRAT_L2: scale_unit_length,
    NUM_STRAT_NONE:none,
    
    CAT_STRAT_ONEHOT:encode_onehot,
    CAT_STRAT_LABEL:encode_label,
    CAT_STRAT_BINARY: encode_binary,
    CAT_STRAT_FREQ: encode_frequency,
    
    TEXT_STRAT_TFIDF: text_to_tfidf,
    TEXT_STRAT_BOW: text_to_bow,
    TEXT_STRAT_NGRAM: text_to_ngrams,
    TEXT_STRAT_LEXICAL: text_to_lexical,
    
    DATETIME_STRAT_SPLIT: split_datetime,
    DATETIME_STRAT_TIMESTAMP: datetime_to_timestamp,
    DATETIME_STRAT_WEEKDAY: extract_weekday,
}

def apply_strat(
    df: pd.DataFrame,
    col: str,
    transformer: Callable[[pd.Series], Union[pd.Series, pd.DataFrame]],
) -> Union[pd.DataFrame, None]:

    transformed_data = transformer(df[col])
    df.drop(col, axis=1, inplace=True)
    return pd.concat([df,transformed_data], axis=1, join='inner')