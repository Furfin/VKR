NUM_AUG_SMOTE = "Smote"
NUM_AUG_NOISE = "Numeric Noise"
NUM_AUG_MIXUP = "MixUp"

CAT_AUG_FREQW = "Freqweight"

TEXT_AUG_NOISE = "Text Noise"

DATETIME_AUG_SHIFT = "Timeseries shift"
DATETIME_AUG_NOISE = "Datetime Noise"
DATETIME_AUG_SCALE = "Scaling"


from nlpaug.augmenter.word import SynonymAug, RandomWordAug
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from typing import Callable
from datetime import timedelta

# ТЕКСТ

def augment_text_noise(text_series: pd.Series, n_samples: int, noise_type: str = "swap") -> pd.Series:
    """Добавление шума: перестановка слов."""
    aug = RandomWordAug(action=noise_type)
    new_texts = [" ".join(aug.augment(text)) for text in text_series.sample(n_samples, replace=True)]
    return pd.concat([text_series, pd.Series(new_texts)], ignore_index=True)

# ЧИСЛОВЫЕ

def augment_numeric_smote(data_series: pd.Series, n_samples: int, k_neighbors: int = 5) -> pd.Series:
    """Аугментация через SMOTE (синтез новых примеров на основе k ближайших соседей)."""
    X = data_series.values.reshape(-1, 1)
    y = np.random.randint(0, 2, len(X))  # Замените на реальные метки для классификации
    sm = SMOTE(k_neighbors=k_neighbors)
    X_res, _ = sm.fit_resample(X, y)
    new_data = X_res[-n_samples:, 0]
    return pd.concat([data_series, pd.Series(new_data)], ignore_index=True)

def augment_numeric_noise(data_series: pd.Series, n_samples: int, noise_scale: float = 0.1) -> pd.Series:
    """Добавление гауссова шума к существующим данным."""
    noise = np.random.normal(0, data_series.std() * noise_scale, n_samples)
    new_data = data_series.sample(n_samples, replace=True).values + noise
    return pd.concat([data_series, pd.Series(new_data)], ignore_index=True)

def augment_numeric_mixup(data_series: pd.Series, n_samples: int) -> pd.Series:
    """Линейная интерполяция между случайными парами примеров (MixUp)."""
    new_data = []
    for _ in range(n_samples):
        x1, x2 = data_series.sample(2).values
        alpha = np.random.beta(0.4, 0.4)  # Равномерное смешивание
        new_data.append(alpha * x1 + (1 - alpha) * x2)
    return pd.concat([data_series, pd.Series(new_data)], ignore_index=True)

# КАТЕГОРИИ

def augment_categorical_freqweight(cat_series: pd.Series, n_samples: int) -> pd.Series:
    """Аугментация через частотное взвешивание (усиление редких категорий)."""
    freq = cat_series.value_counts(normalize=True)
    new_data = np.random.choice(freq.index, size=n_samples, p=freq.values)
    return pd.concat([cat_series, pd.Series(new_data)], ignore_index=True)

# ВРЕМЕННЫЕ МЕТКИ

def augment_timeseries_shift(ts_series: pd.Series, n_samples: int, max_shift_days: int = 3) -> pd.Series:
    new_dates = []
    for _ in range(n_samples):
        base_date = ts_series.sample(1).iloc[0]
        shift = timedelta(days=np.random.randint(1, max_shift_days+1))
        new_dates.append(base_date + shift)
    return pd.concat([ts_series, pd.Series(new_dates)], ignore_index=True)

def augment_timeseries_noise(ts_series: pd.Series, n_samples: int, noise_scale: float = 0.1) -> pd.Series:
    new_dates = []
    for _ in range(n_samples):
        base_date = ts_series.sample(1).iloc[0]
        # Конвертируем в timestamp (секунды с эпохи)
        ts = base_date.timestamp()
        noise = np.random.normal(0, np.std([d.timestamp() for d in ts_series]) * noise_scale)
        new_date = pd.to_datetime(ts + noise, unit='s')
        new_dates.append(new_date)
    return pd.concat([ts_series, pd.Series(new_dates)], ignore_index=True)

def augment_timeseries_scaling(ts_series: pd.Series, n_samples: int, scale_range: tuple = (0.9, 1.1)) -> pd.Series:
    new_sequences = []
    for _ in range(n_samples):
        # Выбираем случайную последовательность
        sample_size = min(10, len(ts_series))
        base_dates = ts_series.sample(sample_size).sort_values().reset_index(drop=True)
        
        # Масштабируем интервалы
        scale = np.random.uniform(*scale_range)
        scaled_dates = [base_dates.iloc[0]]
        for i in range(1, len(base_dates)):
            delta = (base_dates.iloc[i] - base_dates.iloc[i-1]) * scale
            scaled_dates.append(scaled_dates[-1] + delta)
            
        new_sequences.extend(scaled_dates)
    
    return pd.concat([ts_series, pd.Series(new_sequences)], ignore_index=True)

augmentations = {
    NUM_AUG_SMOTE: augment_numeric_smote,
    NUM_AUG_NOISE: augment_numeric_noise,
    NUM_AUG_MIXUP: augment_numeric_mixup,

    CAT_AUG_FREQW: augment_categorical_freqweight,

    TEXT_AUG_NOISE: augment_text_noise,

    DATETIME_AUG_SHIFT: augment_timeseries_shift,
    DATETIME_AUG_NOISE: augment_timeseries_noise,
    DATETIME_AUG_SCALE: augment_timeseries_scaling
}

def augment_data(series: pd.Series, n_samples:int, augmentator: Callable[[pd.Series, int], pd.Series]) -> pd.Series:
    return augmentator(series, n_samples)