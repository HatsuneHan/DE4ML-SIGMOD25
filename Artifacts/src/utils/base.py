import json
import os
import pandas as pd
import IPython.display
import raha
import numpy as np
import requests

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def GetEmbeddings(X, y):

  print("original shape: ")
  print(X.shape, y.shape)
  
  # get numerical and categorical columns
  X_numerical_columns = X.select_dtypes(include=[np.number]).columns
  X_categorical_columns = X.columns.difference(X_numerical_columns)

  # scaler for numerical data in X
  scaler_X = StandardScaler()

  X[X_numerical_columns] = scaler_X.fit_transform(X[X_numerical_columns])

  # onehot encoder for categorical data in X
  encoder_X = OneHotEncoder()

  X_onehot = encoder_X.fit_transform(X[X_categorical_columns])

  feature_names = [f"{col}__{val}" for col, vals in zip(X_categorical_columns, encoder_X.categories_) for val in vals]

  X_onehot_df = pd.DataFrame(X_onehot.toarray(), 
                          columns=feature_names, index=X.index)

  X = X.drop(columns=X_categorical_columns)
  X = pd.concat([X, X_onehot_df], axis=1)

  # one hot encoding
  encoder_y = LabelEncoder()
  y = encoder_y.fit_transform(y)

  # print X shape
  print("transformed shape: ")
  print(X.shape, y.shape)

  return X, y, scaler_X, encoder_X, encoder_y

def GetEmbeddingsWithEncoder(X, y, scaler_X, encoder_X, encoder_y):

  print("original shape: ")
  print(X.shape, y.shape)
  
  # get numerical and categorical columns
  X_numerical_columns = X.select_dtypes(include=[np.number]).columns
  X_categorical_columns = X.columns.difference(X_numerical_columns)

  # scaler for numerical data in X
  X[X_numerical_columns] = scaler_X.transform(X[X_numerical_columns])

  # onehot encoder for categorical data in X
  X_onehot = encoder_X.transform(X[X_categorical_columns])

  feature_names = [f"{col}__{val}" for col, vals in zip(X_categorical_columns, encoder_X.categories_) for val in vals]

  X_onehot_df = pd.DataFrame(X_onehot.toarray(), 
                          columns=feature_names, index=X.index)

  X = X.drop(columns=X_categorical_columns)
  X = pd.concat([X, X_onehot_df], axis=1)

  # one hot encoding
  y = encoder_y.transform(y)

  # print X shape
  print("transformed shape: ")
  print(X.shape, y.shape)

  return X, y


class Cleaning:
  def __init__(self, root_dir, dataset, label_column, verbose = False, labeling_budget = 20):
    # dataset path
    self.ROOT_DIR = root_dir
    self.DATASET = dataset
    self.LABEL_COLUMN = label_column

    # use for raha and baran
    self.VERBOSE = verbose
    self.LABELING_BUDGET = labeling_budget
    self.SAVE_DIR = os.path.join(root_dir, dataset, 'train')

    # rahabaran
    self.DATASET_DICT = {
      "name": dataset,
      "path": os.path.join(root_dir, dataset, 'train', dataset + '_dirty.csv'), # dirty_path
      "clean_path": os.path.join(root_dir, dataset, 'train', dataset + '_clean.csv'), # clean_path
    }
  
  def mixRahaBaran(self):  
    # init raha and baran
    myraha = raha.Detection()
    mybaran = raha.Correction()

    # set the parameters for raha and baran
    myraha.LABELING_BUDGET = self.LABELING_BUDGET
    mybaran.LABELING_BUDGET = self.LABELING_BUDGET
    myraha.VERBOSE = self.VERBOSE
    mybaran.VERBOSE = self.VERBOSE

    # initialize the dataset for raha
    d = myraha.initialize_dataset(self.DATASET_DICT)
    print(d.dataframe.shape)
    # the beginning of raha
    # step 1: run each strategy
    myraha.run_strategies(d)

    # step 2: generate features according to the results from different strategies
    myraha.generate_features(d)

    # step 3: clust the feature vectors
    myraha.build_clusters(d)
    
    while len(d.labeled_tuples) < myraha.LABELING_BUDGET:
      # step 4: sample from each cluster 
      myraha.sample_tuple(d)

      # step 5a: label each representative (use existing ground truth)
      if d.has_ground_truth:
        myraha.label_with_ground_truth(d)
      # step 5b: label each representative (otherwise, display a gui to label manually)
      else:
        print("Label the dirty cells in the following sampled tuple.")
        sampled_tuple = pd.DataFrame(data=[d.dataframe.iloc[d.sampled_tuple, :]], columns=d.dataframe.columns)
        IPython.display.display(sampled_tuple)
        for j in range(d.dataframe.shape[1]):
          cell = (d.sampled_tuple, j)
          value = d.dataframe.iloc[cell]
          correction = input("What is the correction for value '{}'? Type in the same value if it is not erronous.\n".format(value))
          user_label = 1 if value != correction else 0
          d.labeled_cells[cell] = [user_label, correction]
        d.labeled_tuples[d.sampled_tuple] = 1

    # step 6: propagate manual labels in the clusters
    myraha.propagate_labels(d)

    # step 7: train and predict the rest of data cells
    myraha.predict_labels(d)
    # the end of raha

    # initialize the dataset for baran
    d = mybaran.initialize_dataset(d)

    # initialize the models for baran
    mybaran.initialize_models(d)

    # step 1: set labeled tuples of baran to be the same as raha
    for si in d.labeled_tuples:
      # step 2: iteratively update the models
      d.sampled_tuple = si
      mybaran.update_models(d)
      mybaran.predict_corrections(d)

    repair_df = d.dataframe.copy()
    for cell in d.corrected_cells:
      repair_df.iloc[cell] = d.corrected_cells[cell]

    p, r, f = d.get_data_cleaning_evaluation(d.corrected_cells)[-3:]
    print("Baran's performance on {}:\nPrecision = {:.2f}\nRecall = {:.2f}\nF1 = {:.2f}".format(d.name, p, r, f))  
    
    print("Repair data shape: ", repair_df.shape)
    return repair_df
  
  def rock(self):
    dirty_path = os.path.join('/tmp/de4ml-rock', self.DATASET, 'train', self.DATASET + '_dirty.csv')

    save_rule_path = os.path.join('/tmp/de4ml-rock', self.DATASET, 'train', self.DATASET + '_rule.csv')
    rule_path = None

    clean_path = os.path.join('/tmp/de4ml-rock', self.DATASET, 'train', self.DATASET + '_clean.csv')
    output_path = os.path.join('/tmp/de4ml-rock', self.DATASET, 'train', self.DATASET + '_rock.csv')

    if self.DATASET == "adult_balanced":
      data = {
        "taskId": 110042,
        "csvInfo": {
            "tableName": "adult_balanced",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "age": "float64",
                "workclass": "string",
                "fnlwgt": "string",
                "education": "string",
                "educational-num": "float64",
                "marital-status": "string",
                "occupation": "string",
                "relationship": "string",
                "race": "string",
                "gender": "string",
                "capital-gain": "float64",
                "capital-loss": "float64",
                "hours-per-week": "float64",
                "native-country": "string",
                "income": "string"
            }
        },
        "rds": {
            "taskID": 110042,
            "tablesID": [],
            "conf": {
                "EnumSize":45,
                "TopKLayer":10,
                "TreeLevel":4,
                "DecisionTreeMaxDepth":10,
                "Confidence":0.8,
                "Support":0.001
            },
            "SkipYColumns": ["income", "fnlwgt"]
        },
        "correctInfo":{
            "maxCorrectInChase":1200,
            "maxChase": 150,
            "minCorrectRatio":0.5,
            "topRuleNumber":150
        },
        "enableEnum": True,
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "german":
      data = {
        "taskId": 140015,
        "csvInfo": {
            "tableName": "german",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "Status of existing checking account": "string",
                "Duration": "float64",
                "Credit history": "string",
                "Purpose": "string",
                "Credit amount": "float64",
                "Savings account/bonds": "string",
                "Present employment since": "string",
                "Installment rate": "float64",
                "personal status": "string",
                "Other debtors": "string",
                "Present residence since": "float64",
                "Property": "string",
                "Age": "float64",
                "Other installment plans": "string",
                "Housing": "string",
                "Number of existing credits": "float64",
                "Job": "string",
                "Number of people": "float64",
                "Telephone": "string",
                "foreign worker": "string",
                "class": "string"
            }
        },
        "rds": {
            "taskID": 140015,
            "tablesID": [],
            "conf": {
                "EnumSize":45,
                "TopKLayer":10,
                "TreeLevel":3,
                "DecisionTreeMaxDepth":10,
                "Confidence":0.85,
                "Support":0.001
            },
            "SkipYColumns": ["class"]
        },
        "correctInfo":{
            "maxCorrectInChase":55,
            "maxChase": 120,
            "minCorrectRatio":0.5,
            "topRuleNumber":200
        },
        "enableEnum": True,
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "marketing":
      data = {
        "taskId": 130247,
        "csvInfo": {
            "tableName": "marketing",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
              "Sex": "string",
              "Marital": "string",            
              "Age": "float64",
              "Education": "string", 
              "Occupation": "string",
              "Live": "string",
              "Dual": "string",
              "Person": "string",
              "Person_under_18": "string",
              "Householder": "string",
              "Hometype": "string",
              "Ethnic": "string",
              "Language": "string",
              "Income": "string"
            }
        },
        "rds": {
            "taskID": 130247,
            "tablesID": [],
            "conf": {
                "EnumSize":45,
                "TopKLayer":10,
                "TreeLevel":3,
                "DecisionTreeMaxDepth":10,
                "Confidence":0.7,
                "Support":0.01,
                "DecisionTreeMaxRowSize": 1000,
            },
            "SkipYColumns": ["Income"]
        },
        "correctInfo":{
            "maxCorrectInChase":500,
            "maxChase": 120,
            "minCorrectRatio":0.5,
            "topRuleNumber":300
        },
        "enableEnum": True,
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "mushroom":
      data = {
        "taskId": 180047,
        "csvInfo": {
            "tableName": "mushroom",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
              "cap-shape": "string",
              "cap-surface": "string",
              "cap-color": "string",
              "bruises": "string",
              "odor": "string",
              "gill-attachment": "string",
              "gill-spacing": "string",
              "gill-size": "string",
              "gill-color": "string",
              "stalk-shape": "string",
              "stalk-surface-above-ring": "string",
              "stalk-surface-below-ring": "string",
              "stalk-color-above-ring": "string",
              "stalk-color-below-ring": "string",
              "veil-type": "string",
              "veil-color": "string",
              "ring-number": "string",
              "ring-type": "string",
              "spore-print-color": "string",
              "population": "string",
              "habitat": "string",
              "label": "string"
            }
        },
        "rds": {
            "taskID": 180047,
            "tablesID": [],
            "conf": {
                "DecisionTreeMaxRowSize": 1000,
                "EnumSize": 45,
                "TopKLayer": 10,
                "TreeLevel": 3,
                "Confidence": 0.9,
                "Support": 0.001,
                "DecisionTreeMaxDepth": 10
            },
            "SkipYColumns": ["label"]
        },
        "correctInfo":{
            "maxChase": 100,
            "minCorrectRatio": 0.5,
            "maxCorrectInChase": 1000,
            "topRuleNumber": 50
        },
        "enableEnum": True,
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    elif self.DATASET == "Bank_balanced":
      data = {
        "taskId": 138247,
        "csvInfo": {
            "tableName": "Bank",
            "path": dirty_path,
            "cleanPath": clean_path,
            "columnType": {
                "age": "float64",
                "job": "string",
                "marital": "string",
                "education": "string",
                "default": "string",
                "balance": "float64",
                "housing": "string",
                "loan": "string",
                "contact": "string",
                "day": "float64",
                "month": "string",
                "duration": "float64",
                "campaign": "float64",
                "pdays": "float64",
                "previous": "float64",
                "poutcome": "float64",
                "y":"string"
            }
        },
        "rds": {
            "taskID": 138247,
            "tablesID": [],
            "conf": {
                "EnumSize": 45,
                "TopKLayer": 10,
                "TreeLevel": 3,
                "Confidence": 0.85,
                "Support": 0.001,
                "DecisionTreeMaxDepth": 10
            },
            "SkipYColumns": ["y", "poutcome"]
        },
        "correctInfo":{
            "maxChase": 120,
            "minCorrectRatio": 0.5,
            "maxCorrectInChase": 1000,
            "topRuleNumber": 200
        },
        "enableEnum": True,
        "resultSavePath": output_path,
        "ruleSavePath": save_rule_path, 
        "rulePath": rule_path
      }
    
    else:
      raise ValueError("Dataset not supported")
    
    data = json.dumps(data)
    
    ip_addr = "http://192.168.51.10:"
    response = requests.post(ip_addr + str(19123) + "/rock-4-ml", data=data)

    print(response.text)