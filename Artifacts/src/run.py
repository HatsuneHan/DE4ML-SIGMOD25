import argparse
import os

import pandas as pd
import time

from de4ml import DE4ML

def initParser():
  parser = argparse.ArgumentParser(description='Clean the dataset')
  parser.add_argument('-r', '--root_dir', type=str, help='The root directory', required=True)
  parser.add_argument('-d', '--dataset', type=str, help='The dataset name', required=True)
  parser.add_argument('-c', '--cr_method', type=str,  help='CR method', required=True)
  parser.add_argument('-m', '--model_name', type=str,  help='Model Name', required=True)
  parser.add_argument('-ab_e', '--abnormal_epoch', type=int,  help='Training epochs to identify abnormal tuples', default=5)
  parser.add_argument('-dnum', '--datamodels_num', type=int,  help='Number of models used in regression in datamodels', default=100)
  parser.add_argument('-mode', '--mode', type=str,  help='Algorithm Mode', required=True)
  parser.add_argument('-init_repair', '--init_repair', type=bool,  help='whether init or use existing data in repair', default=False)
  parser.add_argument('-init_enhance', '--init_enhance', type=bool,  help='whether init or use existing data in enhance', default=False)
  parser.add_argument('-sr', '--sample_ratio', type=float,  help='sample ratio of datamodels', default=0.5)
  parser.add_argument('-gpu', '--gpu_num', type=int, help='number of gpus', default=1)

  parser.add_argument('-a', '--attacker', type=str,  help='attacker', default="bim")

  parser.add_argument('-B_inf', '--B_inf', type=float,  help='allowed acc lowerbound for data enhancing (not affect repairing)', default=0.5)

  return parser

if __name__ == "__main__":
  args = initParser().parse_args()

  root_dir = args.root_dir
  dataset_name = args.dataset
  dataset_label_dict = {"adult": "income",  "default": "default.payment.next.month", 
                        "Bank": "y", "german": "class", "marketing": "Income",
                        "default_balanced": "default.payment.next.month",
                        "Bank_balanced": "y", "adult_balanced": "income",
                        "mushroom": "label", "EEG": "eyeDetection"}

  base_train_dir = os.path.join(root_dir, dataset_name, 'train')
  base_test_dir = os.path.join(root_dir, dataset_name, 'test')

  data_train_dirty = pd.read_csv(os.path.join(base_train_dir, dataset_name + '_dirty.csv'), na_values=['?'])
  data_train_clean = pd.read_csv(os.path.join(base_train_dir, dataset_name + '_clean.csv'), na_values=['?'])

  data_test_clean = pd.read_csv(os.path.join(base_test_dir, dataset_name + '_clean.csv'), na_values=['?'])

  X_train_dirty = data_train_dirty.drop(dataset_label_dict[dataset_name], axis=1)
  X_train_clean = data_train_clean.drop(dataset_label_dict[dataset_name], axis=1)

  y_train_dirty = data_train_dirty[dataset_label_dict[dataset_name]]
  y_train_clean = data_train_clean[dataset_label_dict[dataset_name]]
  
  X_test = data_test_clean.drop(dataset_label_dict[dataset_name], axis=1)
  y_test = data_test_clean[dataset_label_dict[dataset_name]]

  if os.path.exists(os.path.join(base_test_dir, dataset_name + '_' + args.attacker + '.csv')):
    data_test_adv = pd.read_csv(os.path.join(base_test_dir, dataset_name + '_' + args.attacker + '.csv'), na_values=['?'])
    X_test_adv = data_test_adv.drop(dataset_label_dict[dataset_name], axis=1)
    y_test_adv = data_test_adv[dataset_label_dict[dataset_name]]

    y_test_adv = y_test_adv.to_frame()
  else:
    X_test_adv = None
    y_test_adv = None

  importance_vector = pd.read_csv(os.path.join(root_dir, dataset_name, 'feature_importance.csv'))

  de4ml_params = {"cr_method": args.cr_method,
                  "abnormal_epoch": args.abnormal_epoch,
                  "datamodels_num": args.datamodels_num,
                  "mode": args.mode,
                  "init_repair": args.init_repair,
                  "init_enhance": args.init_enhance,
                  "model_name": args.model_name,
                  "sample_ratio": args.sample_ratio,
                  "gpu_num": args.gpu_num,
                  "B_inf": args.B_inf,
                  "attacker": args.attacker}

  dataset_params = {"name": dataset_name,
                    "label_column": dataset_label_dict[dataset_name],
                    "dataset_rootdir": root_dir}

  de4ml = DE4ML(dataset_params, de4ml_params)
  new_row = pd.DataFrame({'feature': [dataset_label_dict[dataset_name]], 'importance': [999]})
  importance_vector = pd.concat([importance_vector, new_row], ignore_index=True)
  start_time = time.time()
  de4ml.fit(X_train_dirty, y_train_dirty.to_frame(), X_test, y_test.to_frame(), X_train_clean, y_train_clean.to_frame(), X_test_adv, y_test_adv, importance_vector)
  end_time = time.time()
  print("Fit time: ", end_time - start_time)