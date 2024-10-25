from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor 

Section('data').params(
    train_dataset=Param(str, 'train dataset path', required=True), 
    label_column=Param(str, 'label column name', required=True),  
    train_length=Param(int, 'train length', required=True),
    model_name=Param(str, 'model name', required=True),
    sample_ratio=Param(float, 'sample ratio', required=True),
    gpu_num = Param(int, 'number of gpus', required=True)
)

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

@param('data.train_dataset')
@param('data.label_column')
@param('data.train_length')
@param('data.model_name')
@param('data.sample_ratio')
@param('data.gpu_num')
def main(*_, index, logdir, train_dataset, label_column, train_length, model_name, sample_ratio, gpu_num):
    make_config()

    data = pd.read_csv(train_dataset)
    data_train = data.iloc[:train_length]

    # data_length = len(data_train)

    mask = np.zeros(train_length, dtype=bool)
    mask[np.random.choice(train_length, int(train_length * sample_ratio), replace=False)] = True

    predictor = TabularPredictor(label = label_column)

    # predictor.fit(train_data = data_train[mask], hyperparameters = {'LR': {'max_iter': 100}})

    if model_name in ["LogisticRegression", "RandomForest", "XGBoost", "NN", "LightGBM", "CatBoost"]:
        predictor.fit(train_data = data_train[mask], 
                                hyperparameters = {'LR': {'max_iter': 100}}, ag_args_fit={'num_gpus': gpu_num})
    else:
        raise ValueError("Invalid model")


    # preds = predictor.predict_proba(data_train).iloc[:, 0]
    
    preds = predictor.predict(data)
    preds_binary = [0 if pred == predictor.class_labels[0] else 1 for pred in preds]

    # print(preds)
    # print(preds_binary)
    # print(predictor.class_labels)
    print(mask.shape, preds.shape)

    return {
        'masks': mask,
        'predictions': preds_binary
    }