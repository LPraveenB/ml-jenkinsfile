import argparse
import time
import os
import dask
import numpy as np
import xgboost as xgb

from dask.distributed import Client
from dask import dataframe as dd
from distributed import LocalCluster
from google.cloud import storage
from urllib.parse import urlparse
from constant import *

"""
This component predicts for a particular SKU, LOCATION, DATE combination whether the SKU is Out of Stock or not
The individual model predictions are stored in 'temp' folder under the output path for each LOCATION_GROUP.

It makes use of 4 models for the same in the following manner:
Model 1 -> Binary classifier -> Predicts OOS
Model 2 -> Binary classifier -> Predicts whether inventory lower than the book
Model 3 -> Regression -> Predicts the inventory range for items whose inventory is lesser than book
Model 4 -> Regression -> Predicts inventory range for items whose inventory is greater than book

It generates a single prediction output CSV for each LOCATION_GROUP daily.
It can accept a list of location groups as input and processes each location group sequentially.
It can run on a local dask cluster as well as a remote dask cluster.
"""


def get_local_dask_cluster(num_workers_local_cluster, num_threads_per_worker, memory_limit_local_worker):
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    dask_cluster = LocalCluster(n_workers=num_workers_local_cluster, threads_per_worker=num_threads_per_worker,
                                memory_limit=memory_limit_local_worker)
    dask_client = Client(dask_cluster)
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    return dask_client


def get_remote_dask_client(dask_address, dask_connection_timeout):
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    dask_client = Client(dask_address, timeout=dask_connection_timeout)
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    return dask_client


def download_model(client, prod_model_dir, local_dir, local_file_name):
    gcs_path = urlparse(prod_model_dir, allow_fragments=False)
    bucket_name = gcs_path.netloc
    path = gcs_path.path.lstrip('/')
    bucket = client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(path)
    local_full_path = os.path.join(local_dir, local_file_name)
    blob.download_to_filename(local_full_path)


def download_and_load_model(gcs_client, step_name, source_path_model):
    print(f'Loading model: {step_name}')
    model_local_path = "."
    model_local_filename = f'model_{step_name}.json'
    download_model(gcs_client, source_path_model, model_local_path, model_local_filename)
    loaded_model = None
    if step_name in [INFERENCE_STEP_1, INFERENCE_STEP_2]:
        loaded_model = xgb.dask.DaskXGBClassifier()
    elif step_name in [INFERENCE_STEP_3, INFERENCE_STEP_4]:
        loaded_model = xgb.dask.DaskXGBRegressor()
    loaded_model.load_model(model_local_filename)
    feature_names = loaded_model.get_booster().feature_names
    return loaded_model, feature_names


def prepare_inference_data(dd_df, source_path, dask_client):
    print('Reading from', source_path)
    input_data_dd = dd_df.read_parquet(source_path, engine='pyarrow', calculate_divisions=False)
    input_data_dd = dask_client.persist(input_data_dd)
    return input_data_dd


def reverse_calc_daily_reg(df):
    df[ORIG_PRED_TEMP] = round(round(df[ORIG_PRED], 2) * df[CURDAY_IP_QTY_EOP_SOH])
    df[ORIG_PRED_LOW] = round(df[ORIG_PRED_TEMP] - (df[ORIG_PRED_TEMP] * 0.15))
    df[ORIG_PRED_HIGH] = round(df[ORIG_PRED_TEMP] + (df[ORIG_PRED_TEMP] * 0.15))
    df[ORIG_PRED_LOW] = df[ORIG_PRED_LOW].apply(np.floor)
    df[ORIG_PRED_HIGH] = df[ORIG_PRED_HIGH].apply(np.ceil)

    df.loc[(df[ORIG_PRED_LOW] == df[ORIG_PRED_HIGH]), ORIG_PRED_LOW] \
        = df[ORIG_PRED_LOW] - 2
    df.loc[(df[ORIG_PRED_HIGH] - df[ORIG_PRED_LOW]) == 1, ORIG_PRED_LOW] \
        = df[ORIG_PRED_LOW] - 1
    df.loc[df[ORIG_PRED_LOW] < 0, ORIG_PRED_HIGH] = df[ORIG_PRED_HIGH] + 1
    df.loc[df[ORIG_PRED_LOW] < 0, ORIG_PRED_LOW] = 0
    return df


def prepare_for_inference(input_data_dd, feature_names):
    """
    This method checks the data type each feature column in the input dataframe
    and if it finds it to be categorical, that column is cast as an int column
    :param input_data_dd: input data frame
    :param feature_names: feature name list of the model
    :return: input data frame with no categorical feature columns
    """
    for feature in feature_names:
        if input_data_dd[feature].dtype == 'O':
            input_data_dd[feature] = input_data_dd[feature].astype(int)
    return input_data_dd


def run_predict(input_data_dd, dask_client, loaded_model_s1, feature_names_s1, loaded_model_s2, feature_names_s2,
                loaded_model_s3,
                feature_names_s3, loaded_model_s4, feature_names_s4, decision_threshold_step_1,
                decision_threshold_step_2):
    print('Running prediction: step 1')
    input_data_dd = prepare_for_inference(input_data_dd, feature_names_s1)
    input_data_dd[BINARY_PRED] = xgb.dask.predict(dask_client, loaded_model_s1.get_booster(),
                                                  input_data_dd[feature_names_s1])
    input_data_dd_all_oos = input_data_dd[(input_data_dd[BINARY_PRED] > decision_threshold_step_1)]
    input_data_dd_all_oos[ORIG_PRED_LOW] = 0
    input_data_dd_all_oos[ORIG_PRED_HIGH] = 0
    input_data_dd_all_oos[ORIG_PRED] = 0

    print('Running prediction: step 2')
    input_data_dd_all_rest = input_data_dd[(input_data_dd[BINARY_PRED] <= decision_threshold_step_1)]
    input_data_dd_all_rest = prepare_for_inference(input_data_dd_all_rest, feature_names_s2)
    input_data_dd_all_rest[HL_ORIG_PRED] = xgb.dask.predict(dask_client, loaded_model_s2.get_booster(),
                                                            input_data_dd_all_rest[feature_names_s2])

    print('Running prediction: step 3')
    input_data_dd_all_rest_ls = input_data_dd_all_rest[
        input_data_dd_all_rest[HL_ORIG_PRED] > decision_threshold_step_2]
    input_data_dd_all_rest_ls = prepare_for_inference(input_data_dd_all_rest_ls, feature_names_s3)
    input_data_dd_all_rest_ls[ORIG_PRED] = xgb.dask.predict(dask_client, loaded_model_s3.get_booster(),
                                                            input_data_dd_all_rest_ls[feature_names_s3])
    input_data_dd_all_rest_ls = input_data_dd_all_rest_ls.reset_index(drop=True)
    input_data_dd_all_rest_ls = input_data_dd_all_rest_ls.map_partitions(reverse_calc_daily_reg)

    print('Running prediction: step 4')
    input_data_dd_all_rest_hs = input_data_dd_all_rest[
        input_data_dd_all_rest[HL_ORIG_PRED] <= decision_threshold_step_2]
    input_data_dd_all_rest_hs = prepare_for_inference(input_data_dd_all_rest_hs, feature_names_s4)
    input_data_dd_all_rest_hs[ORIG_PRED] = xgb.dask.predict(
        dask_client, loaded_model_s4.get_booster(), input_data_dd_all_rest_hs[feature_names_s4])
    input_data_dd_all_rest_hs = input_data_dd_all_rest_hs.reset_index(drop=True)
    input_data_dd_all_rest_hs = input_data_dd_all_rest_hs.map_partitions(reverse_calc_daily_reg)

    print("Concatenate results")
    input_data_dd_all_day_rev = dd.concat([input_data_dd_all_oos, input_data_dd_all_rest_ls, input_data_dd_all_rest_hs])
    # Prediction post-processing
    input_data_dd_all_day_rev[OOS] = 1
    input_data_dd_all_day_rev[OOS] = input_data_dd_all_day_rev[OOS].where(
        (input_data_dd_all_day_rev[ORIG_PRED_LOW] == 0), 0)
    return input_data_dd_all_day_rev, input_data_dd_all_rest, input_data_dd_all_rest_ls, input_data_dd_all_rest_hs


def execute(source_path_model_s1: str, source_path_model_s2: str, source_path_model_s3: str, source_path_model_s4: str,
            decision_threshold_step_1: float, decision_threshold_step_2: float, business_feature_store_base_path: str,
            output_path: str, location_group_list: list, load_date: str, local_dask_flag: str, dask_address: str,
            dask_connection_timeout: int, num_workers_local_cluster: int, num_threads_per_worker: int,
            memory_limit_local_worker: str):
    if local_dask_flag == 'Y':
        dask_client = get_local_dask_cluster(num_workers_local_cluster, num_threads_per_worker,
                                             memory_limit_local_worker)
    else:
        dask_client = get_remote_dask_client(dask_address, dask_connection_timeout)

    gcs_client = storage.Client()
    loaded_model_s1, feature_names_s1 = download_and_load_model(gcs_client, INFERENCE_STEP_1, source_path_model_s1)
    loaded_model_s2, feature_names_s2 = download_and_load_model(gcs_client, INFERENCE_STEP_2, source_path_model_s2)
    loaded_model_s3, feature_names_s3 = download_and_load_model(gcs_client, INFERENCE_STEP_3, source_path_model_s3)
    loaded_model_s4, feature_names_s4 = download_and_load_model(gcs_client, INFERENCE_STEP_4, source_path_model_s4)
    for location_group in location_group_list:
        print("Processing LOCATION_GROUP=" + location_group)
        input_path = (business_feature_store_base_path + '/' + LOCATION_GROUP_FOLDER_PREFIX + location_group +
                      '/' + LOAD_DATE_FOLDER_PREFIX + load_date + '/*' + PARQUET_FILE_EXTENSION)
        print(" path we are using to load dataframe = "+input_path)
        inference_dd = prepare_inference_data(dd, input_path, dask_client)
        prediction_dd, step_2_dd, step_3_dd, step_4_dd = run_predict(inference_dd, dask_client,
                                                                     loaded_model_s1, feature_names_s1,
                                                                     loaded_model_s2, feature_names_s2,
                                                                     loaded_model_s3, feature_names_s3,
                                                                     loaded_model_s4, feature_names_s4,
                                                                     decision_threshold_step_1,
                                                                     decision_threshold_step_2)

        output_path_multi_file_prediction = (output_path + '/' + TEMP_FOLDER_NAME + '/' + LOCATION_GROUP_FOLDER_PREFIX +
                                             location_group + '/' + OOS)
        output_path_concatenated = (output_path + '/' + TEMP_FOLDER_NAME + '/' + LOCATION_GROUP_FOLDER_PREFIX +
                                    location_group + '/' + CONCATENATED_FOLDER_NAME)
        output_path_step_2 = (output_path + '/' + TEMP_FOLDER_NAME + '/' + LOCATION_GROUP_FOLDER_PREFIX +
                              location_group + '/' + INFERENCE_STEP_2)
        output_path_step_3 = (output_path + '/' + TEMP_FOLDER_NAME + '/' + LOCATION_GROUP_FOLDER_PREFIX +
                              location_group + '/' + INFERENCE_STEP_3)
        output_path_step_4 = (output_path + '/' + TEMP_FOLDER_NAME + '/' + LOCATION_GROUP_FOLDER_PREFIX +
                              location_group + '/' + INFERENCE_STEP_4)

        prediction_dd[[SKU, LOCATION, DATE, OOS]].to_csv(output_path_multi_file_prediction, index=False)
        prediction_dd[[SKU, LOCATION, DATE, BINARY_PRED, ORIG_PRED_LOW, ORIG_PRED_HIGH, ORIG_PRED, HL_ORIG_PRED,
                       ORIG_PRED_TEMP, OOS]].to_csv(output_path_concatenated, index=False)
        step_2_dd[[SKU, LOCATION, DATE, BINARY_PRED, HL_ORIG_PRED]].to_csv(output_path_step_2, index=False)
        step_3_dd[[SKU, LOCATION, DATE, BINARY_PRED, ORIG_PRED_LOW, ORIG_PRED_HIGH, ORIG_PRED, HL_ORIG_PRED,
                   ORIG_PRED_TEMP]].to_csv(output_path_step_3, index=False)
        step_4_dd[[SKU, LOCATION, DATE, BINARY_PRED, ORIG_PRED_LOW, ORIG_PRED_HIGH, ORIG_PRED, HL_ORIG_PRED,
                   ORIG_PRED_TEMP]].to_csv(output_path_step_4, index=False)

        output_path_prediction = output_path + '/' + LOCATION_GROUP_FOLDER_PREFIX + location_group + '/prediction-*.csv'
        output_dd = dd.read_csv(output_path_multi_file_prediction + '/*.part')
        output_dd.repartition(1).to_csv(output_path_prediction, index=False)

        print("Finished processing LOCATION_GROUP=" + location_group)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Inference")
    parser.add_argument(
        '--source_path_model_s1',
        dest='source_path_model_s1',
        type=str,
        required=True,
        help='Step1 model directory')
    parser.add_argument(
        '--source_path_model_s2',
        dest='source_path_model_s2',
        type=str,
        required=True,
        help='Step2 model directory')
    parser.add_argument(
        '--source_path_model_s3',
        dest='source_path_model_s3',
        type=str,
        required=True,
        help='Step3 model directory')
    parser.add_argument(
        '--source_path_model_s4',
        dest='source_path_model_s4',
        type=str,
        required=True,
        help='Step4 model directory')
    parser.add_argument(
        '--decision_threshold_step_1',
        dest='decision_threshold_step_1',
        type=float,
        required=False,
        default=0.5,
        help='Decision threshold for Binary OOS prediction')
    parser.add_argument(
        '--decision_threshold_step_2',
        dest='decision_threshold_step_2',
        type=float,
        required=False,
        default=0.5,
        help='Decision threshold for Binary H/L prediction')
    parser.add_argument(
        '--business_feature_store_base_path',
        dest='business_feature_store_base_path',
        type=str,
        required=True,
        help='Base path of the business feature store')
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        required=True,
        help='Directory to write the predictions')
    parser.add_argument(
        '--location_group_list',
        dest='location_group_list',
        type=str,
        nargs='+',
        required=True,
        help='Space separated location groups to split')
    parser.add_argument(
        '--load_date',
        dest='load_date',
        type=str,
        required=True,
        help='UTC load date for inference operation in ISO format')
    parser.add_argument(
        '--local_dask_flag',
        dest='local_dask_flag',
        type=str,
        choices={'Y', 'N'},
        required=True,
        help='Flag to determine whether dask is local or not')
    parser.add_argument(
        '--dask_address',
        dest='dask_address',
        type=str,
        default=None,
        required=False,
        help='Address of the remote dask cluster')
    parser.add_argument(
        '--dask_connection_timeout',
        dest='dask_connection_timeout',
        type=int,
        default=-1,
        required=False,
        help='Remote dask connection timeout in seconds')
    parser.add_argument(
        '--num_workers_local_cluster',
        dest='num_workers_local_cluster',
        type=int,
        default=0,
        required=False,
        help='Number of workers for the local dask cluster')
    parser.add_argument(
        '--num_threads_per_worker',
        dest='num_threads_per_worker',
        type=int,
        default=0,
        required=False,
        help='Number of threads per local dask cluster worker')
    parser.add_argument(
        '--memory_limit_local_worker',
        dest='memory_limit_local_worker',
        type=str,
        default=None,
        required=False,
        help='Memory limit per worker in the local dask cluster')

    args = parser.parse_args(args)
    print("args:")
    print(args)

    if args.local_dask_flag == 'Y':
        if (args.num_workers_local_cluster == 0) or (args.num_threads_per_worker == 0) or \
                (args.memory_limit_local_worker is None):
            raise ValueError("num_workers_local_cluster, num_threads_per_worker & memory_limit_local_worker need to "
                             "have valid values for a local dask cluster")
    else:
        if (args.dask_address is None) or (args.dask_connection_timeout == -1):
            raise ValueError("dask_address & dask_connection_timeout need to have valid values for remote dask cluster")

    execute(args.source_path_model_s1, args.source_path_model_s2, args.source_path_model_s3, args.source_path_model_s4,
            args.decision_threshold_step_1, args.decision_threshold_step_2, args.business_feature_store_base_path,
            args.output_path, args.location_group_list, args.load_date, args.local_dask_flag, args.dask_address,
            args.dask_connection_timeout, args.num_workers_local_cluster, args.num_threads_per_worker,
            args.memory_limit_local_worker)
    print("<-----------Inference Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()