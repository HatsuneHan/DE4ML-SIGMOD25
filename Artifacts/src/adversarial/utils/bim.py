import numpy as np
import pandas as pd

from time import time
from adversarial.utils.preprocessing import DfInfo
from adversarial.utils.preprocessing import inverse_dummy
from adversarial.utils.exceptions import UnsupportedNorm, UnspportedNum
import concurrent.futures

from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import SklearnClassifier, KerasClassifier
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnSVC


def art_wrap_models(models, feature_range):
    '''
    Wrap the model to meet the requirements to art.attacks.evasion.CarliniL0Method
    '''

    return {
        'lr': ScikitlearnLogisticRegression(models['lr'], clip_values=feature_range),
        # 'svc': ScikitlearnSVC(models['svc'], clip_values=feature_range),
        # 'nn_2': KerasClassifier(models['nn_2'], clip_values=feature_range),
    }

def get_bim_instance(wrapped_models, norm, bim_params):
    '''
    '''
    adv_instance = {}

    for k in wrapped_models.keys():

        adv_instance[k] = BasicIterativeMethod(estimator=wrapped_models[k], **bim_params)

    return adv_instance


def generate_bim_result(
        df_info: DfInfo,
        models,
        num_instances,
        batch_size,
        X_test, y_test,
        norm=None,
        models_to_run=['svc', 'lr', 'nn_2'],
):
    
    feature_range=(0,1)
        
    # Set the desired parameters for the attack
    bim_params = {
        'targeted': False,
        'batch_size': batch_size,
    }

    print("Feature range:" )
    print(feature_range)

    wrapped_models = art_wrap_models(models, feature_range)

    # Get adversarial examples generator instance.
    adv_instance = get_bim_instance(wrapped_models, norm=norm, bim_params=bim_params)

    # Initialise the result dictionary.(It will be the return value.)
    results = {}

    if isinstance(num_instances, int) and num_instances % bim_params['batch_size'] == 0:

        X_test_re=X_test[0:num_instances]
        y_test_re=y_test[0:num_instances]
    
    elif isinstance(num_instances, str) and num_instances == 'all':
        
        X_test_num = len(X_test) - (len(X_test)%bim_params['batch_size'])
        X_test_re=X_test[0:X_test_num]
        y_test_num = len(y_test) - (len(y_test)%bim_params['batch_size'])
        y_test_re=y_test[0:y_test_num]

    else:
        raise UnspportedNum()

    # Loop through every models (svc, lr, nn_2)
    for k in models_to_run:
        # Intialise the result for the classifier (predicting model).
        results[k] = []

        print(f"Finding adversarial examples for {k}")

        start_t = time()
        adv = adv_instance[k].generate(x=X_test_re)
        end_t = time()

        # Calculate the running time.
        running_time = end_t - start_t

        # Get the prediction from original predictive model in a human-understandable format.
        if k == 'nn_2':
            # nn return float [0, 1], so we need to define a threshold for it. (It's usually 0.5 for most of the classifier).
            prediction = np.argmax(models[k].predict(X_test_re), axis=1).astype(int)
            adv_prediction = np.argmax(models[k].predict(adv), axis=1).astype(int)
            
        else:
            # dt and rfc return int {1, 0}, so we don't need to define a threshold to get the final prediction.
            prediction = models[k].predict(X_test_re)
            adv_prediction = models[k].predict(adv)

        # Looping throguh first `num_instances` in the test set.
        # for idx, instance in enumerate(X_test_re):
        #     example = instance.reshape(1, -1)
        #     adv_example = adv[idx].reshape(1,-1)

        #     adv_example_df = inverse_dummy(pd.DataFrame(adv_example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)

        #     # Change the found input from ohe format to original format.
        #     input_df = inverse_dummy(pd.DataFrame(example, columns=df_info.ohe_feature_names), df_info.cat_to_ohe_cat)
        #     input_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([prediction[idx]])[0]

        #     results[k].append({
        #         "input": example,
        #         "input_df": input_df,
        #         "adv_example": adv_example,
        #         "adv_example_df": adv_example_df,
        #         "running_time": running_time,
        #         "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
        #         "prediction": df_info.target_label_encoder.inverse_transform([prediction[idx]])[0],
        #         "adv_prediction": df_info.target_label_encoder.inverse_transform([adv_prediction[idx]])[0],
        #     })
        results[k] = [None] * len(X_test_re)
        
        def process_batch(batch, adv, df_info, prediction, y_test, adv_prediction, running_time):
            batch_results = []
            for idx, instance in batch:
                try:
                    example = instance.reshape(1, -1)
                    adv_example = adv[idx].reshape(1, -1)

                    adv_example_df = inverse_dummy(
                        pd.DataFrame(adv_example, columns=df_info.ohe_feature_names),
                        df_info.cat_to_ohe_cat
                    )

                    input_df = inverse_dummy(
                        pd.DataFrame(example, columns=df_info.ohe_feature_names),
                        df_info.cat_to_ohe_cat
                    )
                    input_df.loc[0, df_info.target_name] = df_info.target_label_encoder.inverse_transform([prediction[idx]])[0]

                    batch_results.append({
                        "input": example,
                        "input_df": input_df,
                        "adv_example": adv_example,
                        "adv_example_df": adv_example_df,
                        "running_time": running_time,
                        "ground_truth": df_info.target_label_encoder.inverse_transform([y_test[idx]])[0],
                        "prediction": df_info.target_label_encoder.inverse_transform([prediction[idx]])[0],
                        "adv_prediction": df_info.target_label_encoder.inverse_transform([adv_prediction[idx]])[0],
                        "idx": idx
                    })

                except Exception as e:
                    batch_results.append({"idx": idx, "error": str(e)})

            return batch_results
        
        def create_batches(X, batch_size):
            batches = []
            total = len(X)
            for i in range(0, total, batch_size):
                batch = list(enumerate(X[i:i + batch_size], start=i))
                batches.append(batch)
            return batches
        
        batch_size = int(len(X_test_re) / 32)
        batches = create_batches(X_test_re, batch_size)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(process_batch, batch, adv, df_info, prediction, y_test, adv_prediction, running_time)
                for batch in batches
            ]

            for future in concurrent.futures.as_completed(futures):
                batch_result = future.result()
                for result in batch_result:
                    if 'error' in result:
                        print(f"Error processing instance {result['idx']}: {result['error']}")
                    else:
                        idx = result["idx"]
                        results[k][idx] = {
                            key: value for key, value in result.items() if key != "idx"
                        }
        
    return results
        
        # results[k] = [None] * len(X_test_re)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=1024) as executor:
        #     futures = {executor.submit(process_instance, idx, instance): idx for idx, instance in enumerate(X_test_re)}
        #     for future in concurrent.futures.as_completed(futures):
        #         try:
        #             result = future.result()
        #             results[k][result["idx"]] = result
        #         except Exception as e:
        #             idx = futures[future]
        #             print(f"error in idx {idx}: {e}")

    
