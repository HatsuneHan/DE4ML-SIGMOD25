import copy
import os
import random
import time
import shutil
import json
import numpy as np
import pandas as pd
from utils.base import Cleaning
from autogluon.tabular import TabularPredictor 
from sklearn.metrics import log_loss, accuracy_score
import yaml
import torch as ch
from adversarial.utils.preprocessing import get_columns_type, preprocess_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import adversarial.utils.deepfool as util_deepfool
import adversarial.utils.carlini as util_carlini
import adversarial.utils.fgsm as util_fgsm
import adversarial.utils.bim as util_bim
import adversarial.utils.mim as util_mim
import adversarial.utils.pgd as util_pgd
import adversarial.utils.boundary as util_boundary
import adversarial.utils.hopskipjump as util_hopskipjump
from sklearn.cluster import KMeans

from adversarial.utils.save import process_datapoints, process_result

class DE4ML:

  def __init__(self, dataset_params: dict, model_params: dict):
    self.dataset_name = dataset_params['name']
    self.label_column = dataset_params['label_column']
    self.root_dir = dataset_params['dataset_rootdir']

    self.cr_method = model_params['cr_method']
    self.abnormal_epoch = model_params['abnormal_epoch']
    self.datamodels_num = model_params['datamodels_num']

    self.mode = model_params['mode']
    self.init_repair = model_params['init_repair']
    self.init_enhance = model_params['init_enhance']

    self.model_name = model_params['model_name']
    self.sample_ratio = model_params['sample_ratio']

    self.gpu_num = model_params['gpu_num']
    
    self.attacker = model_params['attacker']
    self.B_inf = model_params['B_inf']
    

  def fit(self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,  
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        X_train_gt: pd.DataFrame,
        y_train_gt: pd.DataFrame,
        X_test_adv: pd.DataFrame,
        y_test_adv: pd.DataFrame,
        importance_vector: pd.DataFrame):
      
    if self.mode != "repair" and self.mode != "enhance" and self.mode != "all":
      raise ValueError("mode must be either 'repair' or 'enhance' or 'all'")
    
    data_train = pd.concat([X_train, y_train], axis=1)

    existing_repair_path = os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data', self.dataset_name + '_repair_' + str(len(importance_vector)) + '.csv')

    if self.mode == "repair" or (self.mode == "all" and not os.path.exists(existing_repair_path)):

      if self.init_repair:
        repair_tmp_data_path = os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data')
        if os.path.exists(repair_tmp_data_path):
          shutil.rmtree(repair_tmp_data_path)
        os.makedirs(repair_tmp_data_path)

        repair_datamodels_path = os.path.join(self.root_dir, self.dataset_name, 'repair', 'datamodels')
        if os.path.exists(repair_datamodels_path):
          shutil.rmtree(repair_datamodels_path)
        os.makedirs(repair_datamodels_path)

        repair_models_path = os.path.join(self.root_dir, self.dataset_name, 'repair', 'models')
        if os.path.exists(repair_models_path):
          shutil.rmtree(repair_models_path)
        os.makedirs(repair_models_path)

      # Probing dirty values
      start_time = time.time()
      cr_clean = Cleaning(self.root_dir, self.dataset_name, self.label_column, verbose=True)
      # cr_clean = Cleaning(self.root_dir, self.dataset_name, self.label_column, verbose=True, labeling_budget=5)
      
      if self.cr_method == 'rahabaran':
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rahabaran.csv')):
          data_train_repair = cr_clean.mixRahaBaran()
          data_train_repair.to_csv(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rahabaran.csv'), index=False)

          data_train_repair = pd.read_csv(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rahabaran.csv'))

          X_train_repair = data_train_repair.drop(self.label_column, axis=1)
          y_train_repair = data_train_repair[self.label_column].to_frame()
        else:
          data_train_repair = pd.read_csv(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rahabaran.csv'))

          X_train_repair = data_train_repair.drop(self.label_column, axis=1)
          y_train_repair = data_train_repair[self.label_column].to_frame()

      elif self.cr_method == 'rock':
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rock.csv')):
          data_train_clean = pd.concat([X_train_gt, y_train_gt], axis=1)
          data_train_dirty = pd.concat([X_train, y_train], axis=1)

          if not os.path.exists(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train')):
            os.makedirs(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train'))
          
          os.chmod(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train'), 0o777)

          data_train_clean.to_csv(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train', self.dataset_name + '_clean.csv'), index=False)
          data_train_dirty.to_csv(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train', self.dataset_name + '_dirty.csv'), index=False)

          os.chmod(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train', self.dataset_name + '_clean.csv'), 0o777)
          os.chmod(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train', self.dataset_name + '_dirty.csv'), 0o777)

          # give the chmod 777 permission to the dir and files
          response = cr_clean.rock()
          print(response)

          data_train_repair = pd.read_csv(os.path.join('/tmp/de4ml-rock', self.dataset_name, 'train', self.dataset_name + '_rock.csv'))
          data_train_repair.to_csv(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rock.csv'), index=False)

          X_train_repair = data_train_repair.drop(self.label_column, axis=1)
          y_train_repair = data_train_repair[self.label_column].to_frame()
          
        else:
          data_train_repair = pd.read_csv(os.path.join(self.root_dir, self.dataset_name, 'train', self.dataset_name + '_rock.csv'))

          X_train_repair = data_train_repair.drop(self.label_column, axis=1)
          y_train_repair = data_train_repair[self.label_column].to_frame()
      
      end_time = time.time()
      print("Time for probing dirty values: ", end_time - start_time)


      #### evaluate dirty
      accuracy_dirty = evaluate(X_train, y_train, X_test, y_test, self.label_column, self.model_name, self.gpu_num, 'repair', 'initial/dirty', self.root_dir, self.dataset_name)

      # #### evaluate repair
      accuracy_repair = evaluate(X_train_repair, y_train_repair, X_test, y_test, self.label_column, self.model_name, self.gpu_num, 'repair', 'initial/full_' + self.cr_method, self.root_dir, self.dataset_name)

      #### evaluate ground truth
      accuracy_gt = evaluate(X_train_gt, y_train_gt, X_test, y_test, self.label_column, self.model_name, self.gpu_num, 'repair', 'initial/gt', self.root_dir, self.dataset_name)

      print("Accuracy Dirty: ", accuracy_dirty)
      print("Accuracy Repair: ", accuracy_repair)
      print("Accuracy GT: ", accuracy_gt)


      # Indentify AIT in DET Algorithm
      data_train_update = copy.deepcopy(data_train)

      result_dict = {}
      result_dict['dirty'] = accuracy_dirty
      result_dict['cr_method'] = accuracy_repair
      result_dict['ground_truth'] = accuracy_gt

      S_cov = set()
      Delta_D_V = set()

      if data_train.shape == data_train_repair.shape:
          for row in range(data_train.shape[0]):
            for col in data_train.columns:
              if data_train.at[row, col] != data_train_repair.at[row, col]:
                S_cov.add((row, col))

      num_feat = len(importance_vector)

      for iteration in range(0, num_feat+1):  

        # step 1: get abnormal tuples
        tuple_loss = [0] * len(data_train_update)
        # data_train = pd.concat([X_train_update, y_train], axis=1)

        # check if directory exists
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data')):
          os.makedirs(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data'))
          
        data_train_update.to_csv(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data', self.dataset_name + '_repair_' + str(iteration) + '.csv'), index=False)
        
        if iteration == num_feat:
          break

        for i in range(1, self.abnormal_epoch + 1):
          save_model_dir_path = os.path.join(self.root_dir, self.dataset_name, 'repair', 'models', "AT", "iteration-" + str(iteration), "epoch-" + str(i))
          if not os.path.exists(save_model_dir_path):
            os.makedirs(save_model_dir_path)

          if not os.listdir(save_model_dir_path):
            predictor = TabularPredictor(label = self.label_column, path = save_model_dir_path, verbosity = False)
            
            if self.model_name == "LogisticRegression":
              predictor.fit(train_data = data_train_update, 
                                      hyperparameters = {'LR': {'max_iter': i}}, ag_args_fit={'num_gpus': self.gpu_num})
            elif self.model_name == "XGBoost":
              # seems no parameter for max_iter
              predictor.fit(train_data = data_train_update, 
                                      hyperparameters = {'LR': {'max_iter': i}}, ag_args_fit={'num_gpus': self.gpu_num})
            elif self.model_name == "NN":
              # nn fits the dataset in several epochs, so it is not suitable for this case
              predictor.fit(train_data = data_train_update, 
                                      hyperparameters = {'LR': {'max_iter': i}}, ag_args_fit={'num_gpus': self.gpu_num})
            elif self.model_name == "CatBoost":
              # seems no parameter for max_iter
              predictor.fit(train_data = data_train_update, 
                                      hyperparameters = {'LR': {'max_iter': i}}, ag_args_fit={'num_gpus': self.gpu_num})
            
            else:
              raise ValueError("model_name must be one of the following: 'LogisticRegression', 'XGBoost', 'NN', 'CatBoost'")
          else:
            predictor = TabularPredictor.load(save_model_dir_path)

          X_train_update = data_train_update.drop(self.label_column, axis=1)
          y_train_update = data_train_update[self.label_column].to_frame()
          
          y_pred_proba = predictor.predict_proba(X_train_update)
          
          for idx in range(len(y_pred_proba)):
            true_label = [y_train_update.loc[idx]]
            predicted_prob = [y_pred_proba.loc[idx].values]
            predicted_prob = predicted_prob / np.sum(predicted_prob)
            loss = log_loss(true_label, predicted_prob, labels=predictor.class_labels)
            tuple_loss[idx] += loss

        tuple_loss = [loss / self.abnormal_epoch for loss in tuple_loss]

        loss_df = pd.DataFrame({
          'index': list(range(len(tuple_loss))),
          'loss': tuple_loss
        })

        loss_df = loss_df.sort_values(by='loss', ascending=False).reset_index(drop=True)
        top_half_loss_df = loss_df.head(int(0.5*len(tuple_loss)))

        AT = top_half_loss_df['index'].tolist()

        # step 2: get D_dt, corrupted tuples 
        D_cot = list({row for row, col in S_cov})
        # print("length of D_cot: ", len(D_cot))
        
        # step 3: get Dhat_dt, corrupted tuples with imperceptible features
        imper_featA = importance_vector.iloc[-1]['feature']

        Dhat_cot = []

        for idx in D_cot:
          if (idx, imper_featA) in S_cov:
            Dhat_cot.append(idx)

        # final step: get AIT, abnormal tuples with imperceptible features
        AIT = list(set(AT) & set(Dhat_cot))
        
        # Train DataModels:
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'repair', 'datamodels', "iteration-" + str(iteration), 'reg_results', 'datamodels.pt')):
          make_spec_file(self.datamodels_num, len(data_train_update), len(data_train_update), 
                        self.dataset_name, self.root_dir, 'repair')
          make_config_file(self.root_dir, self.dataset_name, int(self.datamodels_num * 0.9), int(self.datamodels_num * 0.1), int(self.datamodels_num * 0.05), 42, iteration, 'repair')
          
          script_path = make_datamodels_script(self.root_dir, self.dataset_name, self.datamodels_num, 
                                               len(data_train_update), self.label_column, iteration, 'repair', 
                                               self.model_name, self.sample_ratio, self.gpu_num)
          os.chmod(script_path, 0o755)
          os.system("zsh " + script_path)

        model_data = ch.load(os.path.join(self.root_dir, self.dataset_name, 'repair', 'datamodels', "iteration-" + str(iteration), 'reg_results', 'datamodels.pt'))
        weight = (model_data['weight'].numpy()).T

        # remove unnecessary data
        for root, dirs, files in os.walk(os.path.join(self.root_dir, self.dataset_name, 'repair', 'datamodels', "iteration-" + str(iteration))):
          for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith('reg_results/datamodels.pt') and not file_path.endswith('result_dict.json'):
                os.remove(file_path)
            
          for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  
                shutil.rmtree(dir_path)

        # identify influential tuples
        indices_set = set()
        index_count = {}
        cnt_weight = 0 
        cnt_none = 0

        for theta in weight:
          cnt_weight += 1
          lower_quartile = np.percentile(theta, 50)
          upper_quartile = np.percentile(theta, 100)

          min_indices = np.where(theta <= lower_quartile)[0]
          # max_indices = np.where(theta >= upper_quartile)[0]
          max_indices = []

          indices = list(set(min_indices) | set(max_indices))

          if lower_quartile == upper_quartile:
            cnt_none += 1
            continue

          for index in indices:
            if index in index_count:
              index_count[int(index)] += 1
            else:
              index_count[int(index)] = 1

        # sort index_count in descending order
        index_count = dict(sorted(index_count.items(), key=lambda item: item[1], reverse=True))
        
        # save index_count
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data', 'index_count')):
          os.makedirs(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data', 'index_count'))

        with open(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data', 'index_count', 'index_count_' + str(iteration) + '.json'), 'w') as f:
          json.dump(index_count, f)
        
        expected_cnt = int((len(data_train_update) - cnt_none) * 0.5)

        IT = [key for key, value in index_count.items() if value >= expected_cnt]

        # # Obtain Critical Values: AIT intersects IT
        FS = list(set(AIT) & set(IT))

        S_cv = set()

        for idx in FS:
          data_train_update.loc[idx, imper_featA] = data_train_repair.loc[idx, imper_featA] # value perturbation
          S_cv.add((idx, imper_featA))

        S_cov = S_cov - S_cv
        Delta_D_V = Delta_D_V.union(S_cv)
        
        X_train_update = data_train_update.drop(self.label_column, axis=1)
        y_train_update = data_train_update[self.label_column].to_frame()

        #### evaluate refined scope of repair
        accuracy_update = evaluate(X_train_update, y_train_update, X_test, y_test, self.label_column, self.model_name, self.gpu_num, 'repair', 'iteration_' + str(iteration), self.root_dir, self.dataset_name)
        # result_dict['repair_' + str(iteration+1)] = accuracy_update
        if iteration == num_feat - 1:
          result_dict['deaat'] = accuracy_update

        importance_vector = importance_vector[:-1]

        # save result dict
        with open(os.path.join(self.root_dir, self.dataset_name, 'repair', 'tmp_data', 'result_dict.json'), 'w') as f:
          json.dump(result_dict, f)

      print(result_dict)
              
    if self.mode == "enhance" or self.mode == "all":
      
      if self.init_enhance:
        enhance_models_path = os.path.join(self.root_dir, self.dataset_name, 'enhance', 'models')
        if os.path.exists(enhance_models_path):
          shutil.rmtree(enhance_models_path)
        os.makedirs(enhance_models_path)
        
        enhance_tmp_data_path = os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data')
        if os.path.exists(enhance_tmp_data_path):
          shutil.rmtree(enhance_tmp_data_path)
        os.makedirs(enhance_tmp_data_path)

        enhance_datamodels_path = os.path.join(self.root_dir, self.dataset_name, 'enhance', 'datamodels')
        if os.path.exists(enhance_datamodels_path):
          shutil.rmtree(enhance_datamodels_path)
        os.makedirs(enhance_datamodels_path)

      # DEP algorithm
      data_dep = None
      if self.mode == "enhance":
        data_dep = pd.concat([X_train_gt, y_train_gt], axis=1) 
      elif self.mode == "all":
        data_dep = pd.read_csv(existing_repair_path)
        
      X_dep = data_dep.drop(self.label_column, axis=1)
      y_dep = data_dep[self.label_column].to_frame()

      X_dep_initial = X_dep.copy()
      y_dep_initial = y_dep.copy()

      # split the dataset into training and validation set
      val_ratio = 0.25
      val_attack_num = 1

      np.random.seed(42)
      val_indices = np.random.choice(data_dep.index, size=int(val_ratio * len(data_dep)), replace=False)
      
      train_indices = data_dep.drop(val_indices).index

      dataset_batch_size = None
      if self.dataset_name == 'adult_balanced':
        dataset_batch_size = 128
      elif self.dataset_name == 'Bank_balanced':
        dataset_batch_size = 64
      elif self.dataset_name == 'german':
        dataset_batch_size = 16
      else:
        dataset_batch_size = 32

      S_at_subset_list = []
      adv_cnt = 0
      total_S_at_size = 0

      while True:
        # choose subset indices of X_dep and y_dep
        np.random.seed(42+adv_cnt)
        sample_indices = np.random.choice(data_dep.index, size=int(0.5 * len(data_dep)), replace=False)

        X_dep_partial = X_dep.copy().loc[sample_indices]
        y_dep_partial = y_dep.copy().loc[sample_indices]

        # adversarial toolbox only supports several models, here we use lr as a surrogate model for all other models
        S_at_subset = adversarial_attack(
          X = X_dep_partial, y = y_dep_partial, label_column = self.label_column, 
          model_name = "lr", adv_name = self.attacker, norm = 'inf',
          num_instances = 'all', batch_size = dataset_batch_size, seed = 42, 
          val_indices = None, attack_full = True, same_shape = False)
        
        S_at_subset_list.append(S_at_subset)
        adv_cnt += 1
        total_S_at_size += len(S_at_subset)
        
        print("adv_cnt: ", adv_cnt)
        print("total_S_at_size: ", total_S_at_size)

        if total_S_at_size >= 0.95 * len(X_dep):
          break

      S_at_multi_adv = pd.concat(S_at_subset_list, axis=0).drop_duplicates().reset_index(drop=True)
      S_at = S_at_multi_adv.copy()

      # we do this because sometimes the attacker cannot properly handle label [1,2] and ['1','2'], for this case (label is number) we all convert the label to int
      if data_dep[self.label_column].dtypes == 'int64':
        S_at[self.label_column] = S_at[self.label_column].apply(lambda x: int(x))
        S_at[self.label_column].astype(data_dep[self.label_column].dtypes)
      
      full_S_at = S_at.copy()
      enhance_result_dict = {}

      # initial performance
      data_initial = None
      if self.mode == "enhance":
        data_initial = pd.concat([X_train_gt, y_train_gt], axis=1) 
      elif self.mode == "all":
        data_initial = pd.read_csv(existing_repair_path)

      initial_predictor = TabularPredictor(label = self.label_column, verbosity = False)
      initial_predictor = train_predictor(self.model_name, initial_predictor, data_initial, self.gpu_num)

      unenhanced_y_pred_clean = initial_predictor.predict(X_test)
      unenhanced_acc = accuracy_score(y_test, unenhanced_y_pred_clean)
      unenhanced_y_pred_adv = initial_predictor.predict(X_test_adv)
      unenhanced_rob = accuracy_score(y_test_adv, unenhanced_y_pred_adv)

      enhance_result_dict['unenhanced'] = "rob: " + str(unenhanced_rob)
      
      # DEP algorithm
      acc_curr = acc_pre = rob_curr = rob_pre = rob_pre_2 = 0
      remove_indices = []
      last_remove_indices = []
      dep_iteration = 0

      while not S_at.empty and dep_iteration <= 20:
        
        data_train_dep = data_dep.loc[train_indices]
        data_val_dep = data_dep.loc[val_indices]

        # store train to get datamodels
        save_data = pd.concat([data_train_dep, S_at], axis=0).reset_index(drop=True)
        save_data.to_csv(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', self.dataset_name + '_enhance_' + str(dep_iteration) + '.csv'), index=False)

        save_model_dir_path = os.path.join(self.root_dir, self.dataset_name, 'enhance', 'models', "iteration-" + str(dep_iteration))
        if not os.path.exists(save_model_dir_path):
          os.makedirs(save_model_dir_path)

        if not os.listdir(save_model_dir_path):
          finetuned_predictor = TabularPredictor(label = self.label_column, path = save_model_dir_path, verbosity = False)
          finetuned_predictor = train_predictor(self.model_name, finetuned_predictor, data_train_dep, self.gpu_num)
        else:
          finetuned_predictor = TabularPredictor.load(save_model_dir_path)
        
        PVD = list()

        for _ in range(0, val_attack_num):
          df_val_attack = adversarial_attack(
            X = X_dep_initial, y = y_dep_initial, label_column = self.label_column, 
            model_name = "lr", adv_name = self.attacker, norm = 'inf',
            num_instances = 'all', batch_size = dataset_batch_size, seed = 42, 
            val_indices = val_indices, attack_full = False, same_shape = True)
          
          if data_dep[self.label_column].dtypes == 'int64':
            df_val_attack[self.label_column] = df_val_attack[self.label_column].apply(lambda x: int(x))
            df_val_attack[self.label_column].astype(data_dep[self.label_column].dtypes)
          
          PVD.append(df_val_attack)

        acc_curr, rob_curr, S_mt_indices = dep_evaluate(PVD, data_val_dep, S_at, finetuned_predictor, self.label_column)
        S_at_remained_indices = list(set(S_at.index) - set(S_mt_indices))
        
        with open(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', 'result_dict.json'), 'w') as f:
          json.dump(enhance_result_dict, f)

        if (acc_curr < self.B_inf or (rob_curr <= rob_pre and rob_pre <= rob_pre_2)) and dep_iteration > 2:
          # revoke the change of last iter because it does harm
          data_dep = data_dep.iloc[:-len(remove_indices)]
          data_dep = data_dep.iloc[:-len(last_remove_indices)]
          break

        # train datamodels for data enhancing
        # if True:
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'datamodels', "iteration-" + str(dep_iteration), 'reg_results', 'datamodels.pt')):
          make_spec_file(self.datamodels_num, len(data_train_dep), len(save_data),
                        self.dataset_name, self.root_dir, 'enhance')
          

          make_config_file(self.root_dir, self.dataset_name, int(self.datamodels_num * 0.9), int(self.datamodels_num * 0.1), int(self.datamodels_num * 0.05), 42, dep_iteration, 'enhance')
          
          script_path = make_datamodels_script(self.root_dir, self.dataset_name, self.datamodels_num, 
                                              len(data_train_dep), self.label_column, dep_iteration, 'enhance', 
                                              self.model_name, self.sample_ratio, self.gpu_num)
          
          os.chmod(script_path, 0o755)
          # os.system("zsh " + script_path)
          os.system("bash " + script_path)

        model_enhance_data = ch.load(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'datamodels', "iteration-" + str(dep_iteration), 'reg_results', 'datamodels.pt'))

        # remove unnecessary data
        for root, dirs, files in os.walk(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'datamodels', "iteration-" + str(dep_iteration))):
          for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith('reg_results/datamodels.pt') and not file_path.endswith('result_dict.json'):
              os.remove(file_path)
            
          for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  
                shutil.rmtree(dir_path)


        weight = (model_enhance_data['weight'].numpy()).T
        weight_S_at = weight[len(data_train_dep):]

        # K-Means
        
        if int(len(S_mt_indices) * 0.2) == 0:
          if rob_curr < rob_pre:
            data_dep = data_dep.iloc[:-len(remove_indices)]
          break

        np.random.seed(42)
        kmeans_indices = np.random.choice(S_mt_indices, size=int(len(S_mt_indices) * 0.2), replace=False)
        initial_centroids = weight_S_at[kmeans_indices]
        
        kmeans = KMeans(n_clusters=len(initial_centroids), init=initial_centroids, random_state=42, n_init=1)
        kmeans.fit(weight)

        # get the cluster of each element in weight_S_at
        weight_S_at_clusters = kmeans.predict(weight_S_at)

        dict_cluster_size = {}
        dict_mistaken_size = {}
        dict_correct_size = {}
        dict_ratio = {}

        for idx in kmeans.labels_:
          dict_cluster_size[idx] = dict_cluster_size.get(idx, 0) + 1

        for idx in weight_S_at_clusters[S_mt_indices]:
          dict_mistaken_size[idx] = dict_mistaken_size.get(idx, 0) + 1

        for idx in weight_S_at_clusters[S_at_remained_indices]:
          dict_correct_size[idx] = dict_correct_size.get(idx, 0) + 1

        for idx in dict_cluster_size.keys():
          dict_ratio[int(idx)] = ((dict_mistaken_size.get(idx, 0) - dict_correct_size.get(idx, 0)) / dict_cluster_size[idx], dict_cluster_size[idx])

        # sort dict_ratio descending, if item[1][0] the same, sort by item[1][1] descending
        dict_ratio = dict(sorted(dict_ratio.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True))

        # save dict_ratio
        if not os.path.exists(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', 'dict_ratio')):
          os.makedirs(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', 'dict_ratio'))

        with open(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', 'dict_ratio', 'dict_ratio_' + str(dep_iteration) + '.json'), 'w') as f:
          json.dump(dict_ratio, f)
        
        S_ct = pd.DataFrame()

        last_remove_indices = remove_indices
        # get the adversarial examples in the cluster with the highest ratio
        remove_indices = []

        dict_idx = 0

        while True:
          for idx in range(len(weight_S_at_clusters)):
            if weight_S_at_clusters[idx] == list(dict_ratio.keys())[dict_idx] and idx in S_mt_indices:
              row_to_concat = S_at.loc[[idx]]  
              S_ct = pd.concat([S_ct, row_to_concat], axis=0, ignore_index=True) 
              remove_indices.append(idx)

          # choose more than one cluster if the ratio is 1.0 to accelerate the convergence
          if dict_idx < len(dict_ratio.keys()) - 1 and \
          dict_ratio[list(dict_ratio.keys())[dict_idx+1]][0] == dict_ratio[list(dict_ratio.keys())[dict_idx]][0] and \
          dict_ratio[list(dict_ratio.keys())[dict_idx+1]][0] == 1.0:
            dict_idx += 1
            continue
          else:
            break

        if len(remove_indices) == 0:
          break

        data_dep = pd.concat([data_dep, S_ct], axis=0).reset_index(drop=True)
        X_dep = data_dep.drop(self.label_column, axis=1)
        y_dep = data_dep[self.label_column].to_frame()

        train_indices = data_dep.drop(val_indices).index

        S_at = S_at.drop(remove_indices).reset_index(drop=True)

        rob_pre_2 = rob_pre
        rob_pre = rob_curr
        acc_pre = acc_curr
        
        # evaluate 
        enhanced_predictor = TabularPredictor(label = self.label_column, verbosity = False)
        enhanced_predictor = train_predictor(self.model_name, enhanced_predictor, data_dep, self.gpu_num)

        enhanced_y_pred_clean = enhanced_predictor.predict(X_test)
        enhanced_acc = accuracy_score(y_test, enhanced_y_pred_clean)
        enhanced_y_pred_adv = enhanced_predictor.predict(X_test_adv)
        enhanced_rob = accuracy_score(y_test_adv, enhanced_y_pred_adv)
        
        with open(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', 'result_dict.json'), 'w') as f:
          json.dump(enhance_result_dict, f)

        dep_iteration += 1

      # data_dep is the final enhanced dataset

      # evaluate 
      enhanced_predictor = TabularPredictor(label = self.label_column, verbosity = False)
      enhanced_predictor = train_predictor(self.model_name, enhanced_predictor, data_dep, self.gpu_num)

      enhanced_y_pred_clean = enhanced_predictor.predict(X_test)
      enhanced_acc = accuracy_score(y_test, enhanced_y_pred_clean)
      enhanced_y_pred_adv = enhanced_predictor.predict(X_test_adv)
      enhanced_rob = accuracy_score(y_test_adv, enhanced_y_pred_adv)

      enhance_result_dict['deaap'] = "rob: " + str(enhanced_rob)
      
      with open(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', 'result_dict.json'), 'w') as f:
        json.dump(enhance_result_dict, f)

      data_dep.to_csv(os.path.join(self.root_dir, self.dataset_name, 'enhance', 'tmp_data', self.dataset_name + '_final_enhanced.csv'), index=False)
      
      print(enhance_result_dict)

    return


def make_spec_file(num_models, train_length, pred_length, dataset_name, save_dir, task):
  if task != 'repair' and task != 'enhance':
    raise ValueError("task must be either 'repair' or 'enhance'")
  
  spec = {
    "num_models":num_models,
    "schema": {
      "masks":{
        "dtype":"bool_",
        "shape":[train_length]
      },
      "predictions":{
        "dtype":"int64",
        "shape":[pred_length]
      }
    }  
  }
  with open(os.path.join(save_dir, dataset_name, task, dataset_name+'_spec.json'), 'w') as f:
    json.dump(spec, f)


def make_config_file(save_dir, dataset_name, num_train, num_val, batch_size, seed, iteration, task):
  if task != 'repair' and task != 'enhance':
    raise ValueError("task must be either 'repair' or 'enhance'")
  
  if not os.path.exists(os.path.join(save_dir, dataset_name, task, 'datamodels', "iteration-" + str(iteration))):
    os.makedirs(os.path.join(save_dir, dataset_name, task, 'datamodels', "iteration-" + str(iteration)))

  config = {
    "data": {
      "data_path": os.path.join(save_dir, dataset_name, task, 'datamodels', "iteration-" + str(iteration), dataset_name+'_data.beton'),
      "num_train": num_train,
      "num_val": num_val,
      "seed": seed,
      "target_start_ind": 0,
      "target_end_ind": -1,
    },
    "cfg": {
      "k": 10,
      "batch_size": batch_size,
      "lr": 1e-4,
      "eps": 1e-2,
      "out_dir": os.path.join(save_dir, dataset_name, task, 'datamodels', "iteration-" + str(iteration)),
      "num_workers": 8
    },
    "early_stopping":{
      "check_every": 3,
      "eps": 5e-1
      # "eps": 1
    }
  }

  with open(os.path.join(save_dir, dataset_name, task, dataset_name+'_regression_config.yaml'), 'w') as f:
    yaml.dump(config, f, allow_unicode=True)

def make_datamodels_script(save_dir, dataset_name, num_models, train_length, label_column, iteration, task, model_name, sample_ratio, gpu_num):
  if task != 'repair' and task != 'enhance':
    raise ValueError("task must be either 'repair' or 'enhance'")
  
  work_dir = os.path.join(save_dir, dataset_name, task, 'datamodels', "iteration-" + str(iteration))
  spec_path = os.path.join(save_dir, dataset_name, task, dataset_name+'_spec.json')
  dataset_path = os.path.join(save_dir, dataset_name, task, 'tmp_data', dataset_name + '_' + task + '_' + str(iteration) + '.csv')
  config_path = os.path.join(save_dir, dataset_name, task, dataset_name+'_regression_config.yaml') 

  shell_script = f"""#!/bin/zsh
  set -e
  work_dir={work_dir}

  setopt nullglob
  rm -rf "{work_dir}/"*
  
  echo "Logging in {work_dir}"
  python -m datamodels.training.initialize_store \
    --logging.logdir={work_dir} \
    --logging.spec={spec_path} 

  seq 0 {num_models-1} | parallel -k --lb -j8 CUDA_VISIBLE_DEVICES='$(({{%}} % {gpu_num}))' \
  python -m datamodels.training.worker \
      --worker.index={{}} \
      --worker.main_import=train_datamodels \
      --worker.logdir={work_dir} \
      --data.train_dataset={dataset_path} \
      --data.label_column={label_column} \
      --data.train_length={train_length} \
      --data.model_name={model_name} \
      --data.sample_ratio={sample_ratio} \
      --data.gpu_num={gpu_num}

  python -m datamodels.regression.write_dataset \
    --cfg.data_dir {work_dir} \
    --cfg.out_path {work_dir}/reg_data.beton \
    --cfg.y_name predictions \
    --cfg.x_name masks

  python -m datamodels.regression.compute_datamodels \
    -C {config_path} \
    --data.data_path "{work_dir}/reg_data.beton" \
    --cfg.out_dir "{work_dir}/reg_results"

  echo "> regression DONE!"
  echo "> Datamodels stored in: {work_dir}/reg_results/datamodels.pt" 
  """

  with open(dataset_name + "_datamodels_" + task + ".sh", "w") as file:
    file.write(shell_script)

  return dataset_name + "_datamodels_" + task + ".sh"

def evaluate(X_train, y_train, X_test, y_test, label_column, model_name, gpu_num,
             task = None, case = None, root_dir = None, 
             dataset_name = None):
  
  data_train = pd.concat([X_train, y_train], axis=1)

  save_model_dir_path = os.path.join(root_dir, dataset_name, task, 'models', case)

  if not os.path.exists(save_model_dir_path):
    os.makedirs(save_model_dir_path)

  if not os.listdir(save_model_dir_path):
    predictor = TabularPredictor(label = label_column, path = save_model_dir_path, verbosity = False)
    predictor = train_predictor(model_name, predictor, data_train, gpu_num)
  else:
    predictor = TabularPredictor.load(save_model_dir_path)

  y_pred = predictor.predict(X_test)
  if y_test[label_column].dtypes == 'int64':
    y_pred = y_pred.apply(lambda x: int(x))
    y_pred.astype(y_test[label_column].dtypes)

  accuracy = accuracy_score(y_test, y_pred)

  return accuracy

def load_dataset_config(df, label_column):
  target_name = label_column

  df[target_name] = df[target_name].astype(str)

  feature_names = [col for col in df.columns if col != target_name]
  possible_outcomes = list(df[target_name].unique())
  numerical_cols, categorical_cols, columns_type = get_columns_type(df)

  return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

# the attacker needs sklearn model, but we use autogluon model
# so the model in the `adversarial_attack` function is a surrogate model
  
# not support list:
# pgd, fgsm norm = 2, mim
  
# support list:
# deepfool, carlini, fgsm norm = 1, fgsm norm = 'inf', bim
# blackbox: boundary, hopskipjump norm = 2,  hopskipjump norm = inf
# hopskipjump norm = 1 can run but original repo does not use it
def adversarial_attack(X, y, label_column, model_name, adv_name, norm, num_instances, batch_size, seed, val_indices = None, attack_full = True, same_shape = True):
  df = pd.concat([X, y], axis=1)
  dataset_loading_fn = load_dataset_config(df, label_column)
  df_info = preprocess_df(dataset_loading_fn)

  # this test is actually validation set
  df_train = df_val = None

  if val_indices is None:
    df_train, df_val = train_test_split(
    df_info.dummy_df, train_size=0.75, random_state=seed, shuffle=True)
  else:
    df_train = df_info.dummy_df.drop(val_indices)
    df_val = df_info.dummy_df.loc[val_indices]

  X_train = np.array(df_train[df_info.ohe_feature_names])
  y_train = np.array(df_train[df_info.target_name])
  X_val = np.array(df_val[df_info.ohe_feature_names])
  y_val = np.array(df_val[df_info.target_name])

  X_full = np.array(df_info.dummy_df[df_info.ohe_feature_names])
  y_full = np.array(df_info.dummy_df[df_info.target_name])

  # aim is what we want to attack
  X_aim = X_full if attack_full else X_val
  y_aim = y_full if attack_full else y_val

  # input is what we use to train the model
  X_input = X_full if attack_full else X_train
  y_input = y_full if attack_full else y_train

  models = {}
  if model_name == "lr":
    lr_model = LogisticRegression(random_state = seed)
    lr_model.fit(X_input, y_input)

  models['lr'] = lr_model

  if adv_name == "deepfool":

    deepfool_results = util_deepfool.generate_deepfool_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            models_to_run=["lr"],
        )
    result_dfs = process_result(deepfool_results, df_info, label_column)[model_name]
  
  elif adv_name == "carlini":
    
    carlini_l_2_results = util_carlini.generate_carlini_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            norm="l_2", #["l_2", "l_inf"]
            models_to_run=["lr"]
        )

    result_dfs = process_result(carlini_l_2_results, df_info, label_column)[model_name]

  elif adv_name == "fgsm":
    if norm == 2:
      raise ValueError("Invalid adversarial attack method")
    fgsm_results = util_fgsm.generate_fgsm_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            norm=norm, #[int, float, 'inf']
            models_to_run=["lr"],
        )
    result_dfs = process_result(fgsm_results, df_info, label_column)[model_name]
  
  elif adv_name == "bim":
    bim_results = util_bim.generate_bim_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            models_to_run=["lr"],
        )
    result_dfs = process_result(bim_results, df_info, label_column)[model_name]

  elif adv_name == "mim":
    raise ValueError("Invalid adversarial attack method")
    mim_results = util_mim.generate_mim_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            models_to_run=["lr"],
        )
    mim_datapoints = process_datapoints(mim_results)
    result_dfs = process_result(mim_results, df_info, label_column)[model_name]

  elif adv_name == "pgd":
    raise ValueError("Invalid adversarial attack method")
    pgd_results = util_pgd.generate_pgd_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            norm=norm, #[int, float, 'inf']
            models_to_run=["lr"],
        )
    pgd_datapoints = process_datapoints(pgd_results)
    result_dfs = process_result(pgd_results, df_info, label_column)[model_name]

  elif adv_name == "boundary":
    boundary_results = util_boundary.generate_boundary_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            models_to_run=["lr"], # 
        )
    boundary_datapoints = process_datapoints(boundary_results)
    result_dfs = process_result(boundary_results, df_info, label_column)[model_name]

  elif adv_name == "hopskipjump":
    hopskipjump_results = util_hopskipjump.generate_hopskipjump_result(
            df_info,
            models,
            num_instances,
            batch_size,
            X_aim,
            y_aim,
            norm=2,
            models_to_run=["lr"],
    )
    hopskipjump_datapoints = process_datapoints(hopskipjump_results)
    result_dfs = process_result(hopskipjump_results, df_info, label_column)[model_name]
 
  else:
    raise ValueError("Invalid adversarial attack method")

  df_adversarial = None

  if attack_full:
    # attack all the input data
    df_length = len(result_dfs)
    result_dfs.index = df.index[:df_length]

    result_dfs = result_dfs[result_dfs['Predict_Success?'] == "Y"]
    result_dfs = result_dfs[result_dfs['Attack_Success?'] == "Y"]
    result_dfs = result_dfs.reindex(columns=df.columns)

    df_adversarial = copy.deepcopy(df)

  else:
    # attack only the validation set
    df_length = len(result_dfs)
    result_dfs.index = df_val.index[:df_length]

    result_dfs = result_dfs[result_dfs['Predict_Success?'] == "Y"]
    result_dfs = result_dfs[result_dfs['Attack_Success?'] == "Y"]
    result_dfs = result_dfs.reindex(columns=df.columns)

    df_adversarial = copy.deepcopy(df[df.index.isin(df_val.index)])
  
  if same_shape:
    for idx in result_dfs.index:
      df_adversarial.loc[idx] = result_dfs.loc[idx]
    return df_adversarial
  else:
    return result_dfs
      
def dep_evaluate(PVD, df_val, S_at, M, label_column):
  # get acc
  X_val = df_val.drop(label_column, axis=1)
  y_val = df_val[label_column]
  y_val_pred = M.predict(X_val)
  
  if df_val[label_column].dtypes == 'int64':
    y_val_pred = y_val_pred.apply(lambda x: int(x))
    y_val_pred = y_val_pred.astype(df_val[label_column].dtypes)

  acc_curr = accuracy_score(y_val, y_val_pred)

  # get rob
  rob_curr = list()
  for df_val_attack in PVD:
    X_val_attack = df_val_attack.drop(label_column, axis=1)
    y_val_attack = df_val_attack[label_column]
    y_val_attack_pred = M.predict(X_val_attack)

    if df_val_attack[label_column].dtypes == 'int64':
      y_val_attack_pred = y_val_attack_pred.apply(lambda x: int(x))
      y_val_attack_pred = y_val_attack_pred.astype(df_val[label_column].dtypes)

    acc_attack = accuracy_score(y_val_attack, y_val_attack_pred)

    rob_curr.append(acc_attack)

  rob_curr = np.mean(rob_curr)
  
  # get S_mt
  X_at = S_at.drop(label_column, axis=1)
  y_at = S_at[label_column]
  y_at_pred = M.predict(X_at)

  if S_at[label_column].dtypes == 'int64':
    y_at_pred = y_at_pred.apply(lambda x: int(x))
    y_at_pred = y_at_pred.astype(df_val[label_column].dtypes)

  S_mt_indices = y_at.index[y_at != y_at_pred]

  return acc_curr, rob_curr, S_mt_indices

def train_predictor(model_name, predictor, data_train, gpu_num = 1):

  if model_name == "LogisticRegression":
    predictor.fit(train_data = data_train, 
                            hyperparameters = {'LR': {'max_iter': 100}}, ag_args_fit={'num_gpus': gpu_num})
  elif model_name == "XGBoost":
    # seems no parameter for max_iter
    predictor.fit(train_data = data_train, 
                            hyperparameters = {'XGB': {}}, ag_args_fit={'num_gpus': gpu_num})
  elif model_name == "NN":
    predictor.fit(train_data = data_train, 
                            hyperparameters = {'NN_TORCH': {'num_epochs': 5}}, ag_args_fit={'num_gpus': gpu_num})
  elif model_name == "CatBoost":
    # seems no parameter for max_iter
    predictor.fit(train_data = data_train, 
                            hyperparameters = {'CAT': {}}, ag_args_fit={'num_gpus': gpu_num})
  else:
    raise ValueError("model_name must be one of the following: 'LogisticRegression', 'XGBoost', 'NN', 'CatBoost'")
  
  return predictor