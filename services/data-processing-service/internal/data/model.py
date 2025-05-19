import pandas as pd
from typing import Dict, Any, List
from pkg.minio.client import MinioClient
from internal.web.api.constants import *
from internal.data.dataset_import import *
from internal.data.preprocess_dataset import *
from internal.data.dataset_augmentation import *
from nltk.tokenize import word_tokenize
from io import BytesIO

class Dataset:    
    def __init__(self,minio_client: MinioClient, filename: str,target: str):
        self.filename = filename
        self.minio_client = minio_client
        self.df = None
        self.data: Dict[str, Any] = {"aug_num":0, "target": target, "status":"new"}
        
    async def initialize(self):
        self.df = clean_dataframe(await self.minio_client.download_fileobj_sample(DATASETS_BUCKET, self.filename))
        self.data, self.df = detect_column_types(self.df,self.data, 0.3)
        self.set_preprocessing_strats()
        if self.data[COLUMNS_KEY][self.data["target"]]["type"] != cathegorical_datatype and \
        self.data[COLUMNS_KEY][self.data["target"]]["type"] != numeric_datatype:
            raise ValueError(f'Target must be numeric or cathegorical:' + str(self.data[COLUMNS_KEY][self.data["target"]]["type"]))
    
    def set_preprocessing_strats(self):
        columns = self.data[COLUMNS_KEY].keys()
        data = self.data[COLUMNS_KEY]
        if self.data["target"] not in columns:
            raise ValueError(f"init dataset: traget column not found")
        for col in columns:
            if data[col]["type"] == numeric_datatype:
                    data[col]["strat"] = NUM_STRAT_NONE
                    data[col]["aug"] = NUM_AUG_NOISE
            if data[col]["type"] == cathegorical_datatype:
                    if self.df[col].nunique() > 10:
                        data[col]["strat"] = CAT_STRAT_LABEL
                    else:
                         data[col]["strat"] = CAT_STRAT_ONEHOT
                    data[col]["aug"] = CAT_AUG_FREQW
            if data[col]["type"] == text_datatype:
                data[col]["lang"] = "en"
                sample_text = self.df[col].iloc[0]
                if len(word_tokenize(sample_text)) > 5:
                    data[col]["strat"] = TEXT_STRAT_TFIDF
                else:
                    data[col]["strat"] = TEXT_STRAT_BOW
                data[col]["aug"] = TEXT_AUG_NOISE
            if data[col]["type"] == datetime_datatype:
                data[col]["strat"] = DATETIME_STRAT_TIMESTAMP
                data[col]["aug"] = DATETIME_AUG_NOISE
    
    def set_strats(self, column_data: Dict[str, Any]):
        data = self.data[COLUMNS_KEY]
        for col in column_data.keys():
            if column_data[col] not in list(strats.keys()):
                raise ValueError(f"set strats for preprocessing: unknown strat '{column_data[col]}'")
            data[col]["strat"] = column_data[col]
        self.data[COLUMNS_KEY] = data
    
    def drop_column(self, column: str):
        self.df.drop(column, axis=1, inplace=True)
        del self.data[COLUMNS_KEY][column]
    
    def set_augments(self, column_data: Dict[str, Any]):
        data = self.data[COLUMNS_KEY]
        for col in column_data.keys():
            if column_data[col] not in list(augmentations.keys()):
                raise ValueError(f"set augmentations: unknown augmentation '{column_data[col]}'")
            data[col]["aug"] = column_data[col]
        self.data[COLUMNS_KEY] = data
    
    def set_data(self, data: Dict[str, Any]):
        self.data = data["data"]
        return self.data
        
    def get_data(self) -> Dict[str, Any]:
        return self.data
    
    async def get_preprocessed_dataset(self) -> pd.DataFrame:
        df = self.augment_data(await self.minio_client.download_fileobj_sample(DATASETS_BUCKET, self.filename))
        df.drop(self.data["target"], axis=1, inplace=True)
        columns = self.data[COLUMNS_KEY].keys()
        data = self.data[COLUMNS_KEY]
        for col in columns:
            if col == self.data["target"]:
                continue
            df = apply_strat(df, col, strats[data[col]["strat"]])
        
        return df
    
    async def get_preprocessed_target(self) -> pd.DataFrame:
        df = augment_data((await self.minio_client.download_fileobj_sample(DATASETS_BUCKET, self.filename))[self.data["target"]],  self.data["aug_num"],  augmentations[self.data[COLUMNS_KEY][self.data["target"]]["aug"]])
        data = self.data[COLUMNS_KEY]
        df.name = self.data["target"]
        df = apply_strat(df.to_frame(), self.data["target"], strats[data[self.data["target"]]["strat"]])
        
        return df

    async def upload_preprocessed_target(self):
        df = pd.read_csv(await self.minio_client.download_fileobj(DATASETS_BUCKET, self.filename))[self.data["target"]]
        df.name = self.data["target"]
        if self.data[COLUMNS_KEY][self.data["target"]]["type"] == cathegorical_datatype:
            pass
        else:
            df = apply_strat(df.to_frame(), self.data["target"], strats[NUM_STRAT_NONE])
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        content_type = "text/csv"
        await self.minio_client.upload_fileobj(
            bucket_name=DATASETS_BUCKET,
            object_name="preprocessed_target_"+self.filename,
            file=buffer,
            length=buffer.getbuffer().nbytes,
            content_type=content_type
        )
    
    async def upload_preprocessed_dataset(self):
        df = self.augment_data(pd.read_csv(await self.minio_client.download_fileobj(DATASETS_BUCKET, self.filename)), self.data["aug_num"])
        df.drop(self.data["target"], axis=1, inplace=True)
        columns = self.data[COLUMNS_KEY].keys()
        data = self.data[COLUMNS_KEY]
        for col in columns:
            if col == self.data["target"]:
                continue
            df = apply_strat(df, col, strats[data[col]["strat"]])
        
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        content_type = "text/csv"
        await self.minio_client.upload_fileobj(
            bucket_name=DATASETS_BUCKET,
            object_name="preprocessed_"+self.filename,
            file=buffer,
            length=buffer.getbuffer().nbytes,
            content_type=content_type
        )
    
    
    def set_aug_num(self, n_samples: int):
       self.data["aug_num"] = n_samples
    
    def augment_data(self, df: pd.DataFrame, n_samples: int = 0) -> pd.DataFrame:
        new_df = []
        columns = self.data[COLUMNS_KEY].keys()
        data = self.data[COLUMNS_KEY]
        for col in columns:
            if n_samples == 0:
                new_df.append(df[col])
            else:
                series = augment_data(df[col], n_samples, augmentations[data[col]["aug"]])
                series.name = col
                new_df.append(series)
        return pd.concat(new_df, axis=1, join='inner')
    
    def cast_to_cat(self, col: str) -> pd.Series:
        self.data[COLUMNS_KEY][col] = {'type': cathegorical_datatype}
        self.set_preprocessing_strats()