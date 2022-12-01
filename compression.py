from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import argparse, os, datetime, csv, pdb, time
from common_utilities import get_parser, setup_system, get_dataset_object, extract_features, get_model_hparams
from train_and_eval import train, evaluate, prune

parser = get_parser()
args = parser.parse_args()
experiment_dir = args.working_directory
args = get_model_hparams(args)

METRIC_LIST = ['mcc', 'f1_weighted', 'cohen_kappa', 'precision', 'recall']
HP_PRUNING_LR = hp.HParam('pruning_learning_rate', hp.Discrete([1e-3, 1e-4, 1e-5]))
HP_PRUNING_SCHEDULE = hp.HParam('pruning_schedule', hp.Discrete(['constant_sparsity', 'polynomial_decay']))
HP_PRUNING_FREQUENCY = hp.HParam('pruning_frequency', hp.Discrete([10, 100]))
HP_PRUNING_FINAL_SPARSITY = hp.HParam('pruning_final_sparsity', hp.Discrete([0.2, 0.5, 0.75, 0.8, 0.85, 0.9]))
HP_QUANTIZATION_OPTIMIZATION = hp.HParam('quantization_optimization', hp.Discrete(['float16', 'w8']))

hparams = [HP_PRUNING_LR, HP_PRUNING_SCHEDULE, HP_PRUNING_FREQUENCY, HP_PRUNING_FINAL_SPARSITY, HP_QUANTIZATION_OPTIMIZATION]

DOMAINS = ['all'] + args.domains.split(',')
metrics = []
for m in METRIC_LIST: 
  for d in DOMAINS: 
    metrics.append('_'.join([d, m]))

start =  time.time()

# Log compression metadata
meta_fieldnames = ['exp_id', 'exp_name', 'model_arch', 'dataset_name', 'resample_rate', 'domains', 'trained_model_path','quantize','prune','equal_weighted']
if os.path.isfile(experiment_dir + '/compression_metadata.csv') is False:
  with open(experiment_dir + '/compression_metadata.csv', 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=meta_fieldnames)
    csv_writer.writeheader()

with open(experiment_dir+ '/compression_metadata.csv', 'a', newline='') as csv_file:
  csv_writer = csv.DictWriter(csv_file, fieldnames=meta_fieldnames)
  csv_writer.writerow({'exp_id':args.exp_id, 'exp_name':args.exp_name, 'model_arch':args.model_arch, 'dataset_name':args.dataset_name, 
  'resample_rate':args.resample_rate, 'domains':args.domains, 'trained_model_path':args.trained_model_path, 'quantize':args.quantize, 'prune':args.prune,
  'equal_weighted':args.equal_weighted})

# Run compression
session_num = 0      
inference_start = time.time() 
ds = get_dataset_object(args.dataset_name, args.resample_rate, args.domains)
feature_extractor = extract_features(vars(args))
model_file_path = f"{args.working_directory}/models/{args.trained_model_path}.hdf5"

session_num = 0
if args.prune is True:
  for lr in HP_PRUNING_LR.domain.values:
    for schedule in HP_PRUNING_SCHEDULE.domain.values:
      for frequency in HP_PRUNING_FREQUENCY.domain.values:
        for sparsity in HP_PRUNING_FINAL_SPARSITY.domain.values:
          run_name = "run-%d" % int(datetime.datetime.now().timestamp())

          hparam_dict = {
              HP_PRUNING_LR:lr,
              HP_PRUNING_SCHEDULE:schedule,
              HP_PRUNING_FREQUENCY:frequency,
              HP_PRUNING_FINAL_SPARSITY:sparsity,
              HP_QUANTIZATION_OPTIMIZATION:None
          }

          args.pruning_learning_rate = lr
          args.pruning_schedule = schedule
          args.pruning_frequency = frequency
          args.pruning_final_sparsity = sparsity
          args.quantization_optimization = None
          args.run_name = run_name

          trained_model_file_path = prune(ds, model_file_path, feature_extractor, args)
          results = evaluate(ds, trained_model_file_path, feature_extractor, args)
          compression_time = round(time.time() - inference_start, 1)

          #Log evaluation metrics
          result_dict = {}
          result_dict['run_name'] = args.run_name
          result_dict['trained_model_path'] = args.trained_model_path
          result_dict['compression_time'] = compression_time
          for k, v in hparam_dict.items():
            result_dict[k.name] = v

          for metric in metrics:
            d, m = metric.split('_', 1)
            result_dict[metric] = results[d][m]
            tf.summary.scalar(metric, results[d][m], step=1)

          fieldnames = ['run_name', 'trained_model_path', 'compression_time'] + [p.name for p in hparams] + metrics
          if os.path.isfile(args.working_directory + '/results/compression_'+ args.exp_name +'_'+ str(args.exp_id) + '.csv') is False:
            with open(args.working_directory + '/results/compression_'+ args.exp_name +'_'+ str(args.exp_id) + '.csv', 'w', newline='') as csv_file:
              csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
              csv_writer.writeheader()  

          with open(args.working_directory + '/results/compression_'+ args.exp_name +'_'+ str(args.exp_id) + '.csv', 'a', newline='') as csv_file:
              csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
              csv_writer.writerow(result_dict)

          session_num += 1


if args.quantize is True:
  for optimization in HP_QUANTIZATION_OPTIMIZATION.domain.values:
    run_name = "run-%d" % int(datetime.datetime.now().timestamp())
    hparam_dict = {
    HP_PRUNING_LR:None,
    HP_PRUNING_SCHEDULE:None,
    HP_PRUNING_FREQUENCY:None,
    HP_PRUNING_FINAL_SPARSITY:None,
    HP_QUANTIZATION_OPTIMIZATION:optimization
    }

    args.pruning_learning_rate = None
    args.pruning_schedule = None
    args.pruning_frequency = None
    args.pruning_final_sparsity = None
    args.quantization_optimization = optimization
    args.run_name = run_name

    results = evaluate(ds, model_file_path, feature_extractor, args)
    compression_time = round(time.time() - inference_start, 1)

    #Log evaluation metrics
    result_dict = {}
    result_dict['run_name'] = args.run_name
    result_dict['trained_model_path'] = args.trained_model_path
    result_dict['compression_time'] = compression_time
    for k, v in hparam_dict.items():
      result_dict[k.name] = v

    for metric in metrics:
      d, m = metric.split('_', 1)
      result_dict[metric] = results[d][m]
      tf.summary.scalar(metric, results[d][m], step=1)

    fieldnames = ['run_name', 'trained_model_path', 'compression_time'] + [p.name for p in hparams] + metrics
    if os.path.isfile(args.working_directory + '/results/compression_'+ str(args.exp_id) + '.csv') is False:
      with open(args.working_directory + '/results/compression_'+ str(args.exp_id) + '.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()  

    with open(args.working_directory + '/results/compression_'+ str(args.exp_id) + '.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writerow(result_dict)

    session_num += 1

  end =  time.time()
  print(compression_time)