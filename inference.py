import tensorflow as tf
import argparse, os, datetime, csv, pdb, time
import pandas as pd
from common_utilities import get_parser, setup_system, get_dataset_object, extract_features
from train_and_eval import evaluate


hparams = ['frame_length', 'learning_rate', 'frame_step', 'optimizer', 'window_fn', 'pad_end', 'input_features', 'mel_bins', 'mfccs']
METRIC_LIST = ['mcc', 'f1_weighted', 'cohen_kappa', 'precision', 'recall']


def inference():

  parser = get_parser()
  args = parser.parse_args()
  experiment_dir = args.working_directory #+ '/experiments/'
  # args.working_directory = args.working_directory + '/experiments/'

  if args.exp_id is None:
    return print('ABORTING... Specify the experiment id on which to run inference.')

  DOMAINS = ['all'] + args.domains.split(',')
  metrics = []
  for m in METRIC_LIST: 
    for d in DOMAINS: 
      metrics.append('_'.join([d, m]))

  start =  time.time()

  exp_metadata = pd.read_csv(experiment_dir+'/experiment_metadata.csv')
  exp = exp_metadata[exp_metadata['exp_id']==args.exp_id]
  _exp_name = exp['exp_name'].values[0]
  _resample_rate = exp['resample_rate'].values[0]
  if args.resample_rate is None:
    args.resample_rate = _resample_rate
  args.model_arch = exp['model_arch'].values[0]
  args.equal_weighted = exp['equal_weighted'].values[0]
  args.training_epochs = None
  args.working_directory = setup_system(args.gpu, args.seed, args.working_directory, _exp_name)

  #Log inference metadata
  meta_fieldnames = ['exp_id', 'exp_name', 'model_arch', 'dataset_name', 'resample_rate', 'domains', 'equal_weighted','training_epochs']
  if os.path.isfile(experiment_dir + '/inference_metadata.csv') is False:
    with open(experiment_dir + '/inference_metadata.csv', 'w', newline='') as csv_file:
      csv_writer = csv.DictWriter(csv_file, fieldnames=meta_fieldnames)
      csv_writer.writeheader()

  with open(experiment_dir+ '/inference_metadata.csv', 'a', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=meta_fieldnames)
    csv_writer.writerow({'exp_id':args.exp_id, 'exp_name':args.exp_name, 'model_arch':args.model_arch, 'dataset_name':args.dataset_name, 'resample_rate':args.resample_rate, 'domains':args.domains, 'equal_weighted':args.equal_weighted, 'training_epochs':args.training_epochs})

  #Get hyperparameter data for experiment runs
  result_file = '/'.join([experiment_dir, _exp_name, 'results', 'results_summary_'+str(args.exp_id)+'.csv'])
  results_df = pd.read_csv(result_file)
  results_df['exp_id'] = args.exp_id
  try:
    results_df.drop(columns='resample_rate', inplace=True)
  except KeyError:
    pass

  if args.trained_model_path is not None:
    results_df = results_df[results_df['run_name'] == args.trained_model_path.split('/')[-1].split('.')[0]]

  for i, x in results_df.iterrows():
    #Construct input arguments for inference
    args.run_name = x['run_name']
    args.frame_length = x['frame_length']
    args.learning_rate =  None
    args.frame_step =  x['frame_step']
    args.optimizer = None
    args.window_fn = x['window_fn']
    args.pad_end = x['pad_end']
    args.input_features = x['input_features']
    args.mel_bins = x['mel_bins']
    args.mfccs = x['mfccs']
    args.trained_model_path = f"{args.working_directory}/models/{args.run_name}.hdf5"

    #Do inference
    session_num = 0      
    inference_start = time.time() 
    ds = get_dataset_object(args.dataset_name, args.resample_rate, args.domains)
    feature_extractor = extract_features(vars(args))
    results = evaluate(ds, args.trained_model_path, feature_extractor, args)
    inference_time = round(time.time() - inference_start, 1)

    #Log evaluation metrics
    result_dict = {}
    result_dict['run_name'] = args.run_name
    result_dict['inference_time'] = inference_time
    for k in vars(args):
      if k in hparams:
        result_dict[k] = vars(args)[k]

    for metric in metrics:
      d, m = metric.split('_', 1)
      result_dict[metric] = results[d][m]
      tf.summary.scalar(metric, results[d][m], step=1)

    fieldnames = ['run_name', 'inference_time'] + hparams + metrics
    if os.path.isfile(args.working_directory + '/results/inference_' + args.exp_name + '_' + str(args.exp_id) + '.csv') is False:
      with open(args.working_directory + '/results/inference_' + args.exp_name + '_' + str(args.exp_id) + '.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()  

    with open(args.working_directory + '/results/inference_' + args.exp_name + '_' + str(args.exp_id) + '.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writerow(result_dict)

    session_num += 1

    end =  time.time()
    print(inference_time)
  return print('Inference completed in ', str((end - start)/60), ' minutes')


if __name__=='__main__':
  inference()