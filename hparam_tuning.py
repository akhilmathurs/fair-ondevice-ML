'''
1) To view the results in tensorboard, run the following inside docker with the appropriate values. Port number is the port you opened while running the docker container. 

tensorboard --logdir={working_directory}/logs/hparam_tuning_{exp_name}/ --port 6055 --bind_all

2) Once tensorboard is running, you can ssh into the machine with port forwarding. Change the port as per your docker scripts

ssh -L 6055:localhost:6055 <server_machine_name>

Then go to your browser and open localhost:6055
'''

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import argparse, os, datetime, csv, pdb, time
from common_utilities import get_parser, setup_system, get_dataset_object, extract_features
from train_and_eval import train, evaluate

parser = get_parser()
args = parser.parse_args()
experiment_dir = args.working_directory
args.working_directory = setup_system(args.gpu, args.seed, args.working_directory, args.exp_name)

HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3]))
HP_OPTIM = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_FRAME_LENGTH = hp.HParam('frame_length', hp.Discrete([0.02, 0.025, 0.03, 0.04]))#
HP_FRAME_STEP = hp.HParam('frame_step', hp.Discrete([0.4, 0.5, 0.6]))
HP_WINDOW_FN = hp.HParam('window_fn', hp.Discrete(['hann', 'hamming'])) 
HP_PAD_END = hp.HParam('pad_end', hp.Discrete([True]))
HP_MEL_BINS = hp.HParam('mel_bins', hp.Discrete([20, 26, 32, 40, 60, 80]))
HP_MFCCS= hp.HParam('mfccs', hp.Discrete([-1, 10, 11, 12, 13, 14])) #value -1 is used when input_features != mfcc
HP_INPUT_FEATURES = hp.HParam('input_features', hp.Discrete(['log_mel_spectrogram','mfcc']))

hparams = [HP_FRAME_LENGTH, HP_LR, HP_FRAME_STEP, HP_OPTIM, HP_WINDOW_FN, HP_PAD_END, HP_INPUT_FEATURES, HP_MEL_BINS, HP_MFCCS]

METRIC_LIST = ['mcc', 'f1_weighted', 'cohen_kappa', 'precision', 'recall']#, 'roc_auc_weighted']
DOMAINS = ['all'] + args.domains.split(',')
metrics = []
for m in METRIC_LIST: 
  for d in DOMAINS: 
    metrics.append('_'.join([d, m]))

start =  time.time()
start_time = str(int(datetime.datetime.now().timestamp()))

#Log experiment metadata
meta_fieldnames = ['exp_id', 'exp_name', 'model_arch', 'dataset_name', 'resample_rate', 'domains', 'equal_weighted','training_epochs']
if os.path.isfile(experiment_dir + '/experiment_metadata.csv') is False:
  with open(experiment_dir + '/experiment_metadata.csv', 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=meta_fieldnames)
    csv_writer.writeheader()
with open(experiment_dir+ '/experiment_metadata.csv', 'a', newline='') as csv_file:
  csv_writer = csv.DictWriter(csv_file, fieldnames=meta_fieldnames)
  csv_writer.writerow({'exp_id':start_time, 'exp_name':args.exp_name, 'model_arch':args.model_arch, 'dataset_name':args.dataset_name, 'resample_rate':args.resample_rate, 'domains':args.domains, 'equal_weighted':args.equal_weighted, 'training_epochs':args.training_epochs})

#Create run writer
fieldnames = ['run_name', 'training_time'] + [p.name for p in hparams] + metrics
with open(args.working_directory + '/results/results_summary_' + start_time + '.csv', 'w', newline='') as csv_file:
  csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  csv_writer.writeheader()

session_num = 0
HP_MFCCS_TEMP = HP_MFCCS

for optim in HP_OPTIM.domain.values:
  for lr in HP_LR.domain.values:
    for frame_length in HP_FRAME_LENGTH.domain.values:
      for frame_step in HP_FRAME_STEP.domain.values:
        for window_fn in HP_WINDOW_FN.domain.values:
          for pad_end in HP_PAD_END.domain.values:
            for input_features in HP_INPUT_FEATURES.domain.values:
              if input_features != 'mfcc':
                HP_MFCCS = hp.HParam('mfccs', hp.Discrete([HP_MFCCS_TEMP.domain.values[0]]))
              else:
                HP_MFCCS = hp.HParam('mfccs', hp.Discrete(HP_MFCCS_TEMP.domain.values[1::]))
              for mfccs in HP_MFCCS.domain.values:
                for mel_bins in HP_MEL_BINS.domain.values:
                  run_name = "run-%d" % int(datetime.datetime.now().timestamp())

                  hparam_dict = {
                      HP_FRAME_STEP:frame_step,
                      HP_FRAME_LENGTH: frame_length,
                      HP_LR: lr,
                      HP_INPUT_FEATURES: input_features,
                      HP_MEL_BINS: mel_bins,
                      HP_MFCCS: mfccs,
                      HP_PAD_END: pad_end,
                      HP_WINDOW_FN: window_fn,
                      HP_OPTIM: optim,
                  }

                  args.learning_rate = lr
                  args.optimizer = optim
                  args.run_name = run_name
                  args.start_time = start_time
                  args.frame_length = frame_length
                  args.frame_step =  frame_step
                  args.window_fn = window_fn
                  args.pad_end = pad_end
                  args.input_features = input_features
                  args.mel_bins = mel_bins
                  args.mfccs = mfccs
                  
                  test_summary_writer = tf.summary.create_file_writer(args.working_directory + '/logs/hparam_tuning_' + start_time + "/" + run_name)
                  with test_summary_writer.as_default():
                    hp.hparams(hparam_dict)  # record the values used in this trial
                    
                    training_start = time.time() 
                    ds = get_dataset_object(args.dataset_name, args.resample_rate, args.domains)
                    feature_extractor = extract_features(vars(args)) #convert all arguments into a dictionary and pass to the function
                    trained_model_save_path = train(ds, feature_extractor, args)                      
                    results = evaluate(ds, trained_model_save_path, feature_extractor, args)
                    training_end = time.time()
                    training_time = round(training_end - training_start, 1)

                    result_dict = {}
                    result_dict['run_name'] = args.run_name
                    result_dict['training_time'] = training_time
                    for k, v in hparam_dict.items():
                        result_dict[k.name] = v

                    for metric in metrics:
                      d, m = metric.split('_', 1)
                      result_dict[metric] = results[d][m]
                      tf.summary.scalar(metric, results[d][m], step=1)

                  with open(args.working_directory + '/results/results_summary_' + start_time + '.csv', 'a', newline='') as csv_file:
                      csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                      csv_writer.writerow(result_dict)

                  session_num += 1


end =  time.time()
print(training_end - start)
print(end - start)