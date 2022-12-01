#!/usr/bin/env python3lsc

import argparse, os
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_model_optimization as tfmot
import librosa
from io_ops import load
import models
import pandas as pd
import pathlib
import pdb

def get_parser():
    parser = argparse.ArgumentParser(
        description='Inputs to data loading script')
    parser.add_argument('--dataset_name', choices=[
                        'speech_commands_gender', 'audio_mnist_gender', 'speech_commands_laughter', 
                        'speech_commands_wind', 'speech_commands_rain', 'mswc_rw','mswc_fr','mswc_de','mswc_en'], help='name of dataset file')
    parser.add_argument('--domains', default='male,female', help='domains in the dataset')
    parser.add_argument('--trained_model_path', default=None, type=str,
                        help='the path to the model to be loaded for training and/or evaluation. If None, a new model will be created') #should change this to trained_model_name
    parser.add_argument('--working_directory', default='/data/experiments/',
                        help='the directory where the trained models and results are saved')
    parser.add_argument('--exp_name', required=True,
                        help='the sub-directory where the trained models and results are saved')
    parser.add_argument('--exp_id', default=None, type=int,
                        help='specify the experiment_id for inference')
    parser.add_argument('--training_epochs', default=10, type=int,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size for learning')
    parser.add_argument('--seed', default=222, type=int,
                        help='random seed')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='the initial learning rate during contrastive learning')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'rmsprop'],
                        help='optimizer to use for training')
    parser.add_argument('--model_arch', default='cnn',
                        help='model architecture to use')
    parser.add_argument('--num_classes', type=int,
                        help='number of class labels')
    parser.add_argument('--gpu', default='2',
                        help='gpu to use for training')
    parser.add_argument('--resample_rate', type=int,
                        help='resample rate for dataset')
    parser.add_argument('--frame_length', default=0.025, type=float,
                        help='frame length in seconds to use for training')
    parser.add_argument('--frame_step', default=0.4, type=float,
                        help='frame step as a fraction of frame length to use for training (percentage overlap between frames is (1 - frame_step)*100')
    parser.add_argument('--fft_length', default=None,
                        help='fft_length to use for training')
    parser.add_argument('--window_fn', default='hann',
                        help='window function to use for training')
    parser.add_argument('--pad_end', default=True, action='store_false',
                        help='whether to pad at the end')
    parser.add_argument('--equal_weighted', default=False, action='store_true',
                        help='equal_weighted or not')
    parser.add_argument('--input_features', default='log_mel_spectrogram',
                        help='features to use for training')
    parser.add_argument('--mel_bins', default=40, type=int,
                        help='mel bins to use for training')
    parser.add_argument('--mfccs', default=13, type=int,
                        help='mfccs to use as feature input')
    parser.add_argument('--quantize', default=False, action='store_true',
                        help='quantize the model')
    parser.add_argument('--quantization_optimization', default='float16', type=str,
                        help='optimization to use for model quantization')
    parser.add_argument('--prune', default=False, action='store_true',
                        help='prune the model')
    parser.add_argument('--pruning_schedule', default='constant_sparsity', type=str,
                        help='quantize the model')
    parser.add_argument('--pruning_learning_rate', default=1e-5, type=float,
                        help='learning rate to apply when pruning the model')
    parser.add_argument('--pruning_frequency', default=100, type=int,
                        help='frequency with which to apply pruning to training steps')
    parser.add_argument('--pruning_final_sparsity', default=0.5, type=float,
                        help='final sparsity for pruning')

    return parser


def setup_system(gpu, seed, working_directory, exp_name):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for sel_gpu in gpus:
          tf.config.experimental.set_memory_growth(sel_gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        print(e)

    tf.compat.v1.set_random_seed(seed)

    working_directory = os.path.join(working_directory, exp_name)
    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'logs/'),exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'results/'),exist_ok=True)

    return working_directory


def get_dataset_object(dataset_name, resample_rate, domains):
    domains = domains.split(",")
    ds = load(dataset_name, resample_rate=resample_rate, domains=domains) 
    return ds
    
    
def extract_features(params):
    def feature_extractor(sample, label):
        if params['window_fn'] == "hann":
            window_function = tf.signal.hann_window
        elif params['window_fn'] == "hamming":
            window_function = tf.signal.hamming_window
        else:
            window_function = None

        frame_size = round(params['frame_length'] * params['resample_rate'])
        frame_stride = round(frame_size * params['frame_step'])

        spectrograms = tf.signal.stft(sample, frame_length=frame_size, frame_step=frame_stride, fft_length=params['fft_length'], window_fn=window_function, pad_end=params['pad_end'])
        spectrograms = tf.math.abs(spectrograms)
        # convert spectrograms to mel-scale
        num_spectrogram_bins = spectrograms.shape[-1]
        lower_hz, upper_hz, num_mel_bins = 40.0, params['resample_rate']/2 - 200, params['mel_bins']
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, params['resample_rate'], lower_hz, upper_hz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        if params['input_features'] == 'mel_spectrogram':
            features = tf.expand_dims(tf.transpose(mel_spectrograms), -1)
        
        elif params['input_features'] == 'log_mel_spectrogram':
            features = tf.math.log(mel_spectrograms + 1e-6)
            features = tf.expand_dims(tf.transpose(mel_spectrograms), -1)
        
        elif params['input_features'] == 'mfcc':
            features = tf.math.log(mel_spectrograms + 1e-6)
            features = tf.signal.mfccs_from_log_mel_spectrograms(features)[..., :params['mfccs']]
            features = tf.expand_dims(tf.transpose(features), -1)

        else:
            print("Undefined input features of type {}".format(params['input_features']))

        return features, label

    return feature_extractor


def get_model_hparams(args):
    experiment_dir = args.working_directory
    if args.trained_model_path is None or args.exp_id is None:
        return print('ABORTING... Specify the exp_id and trained_model_path for which you want to get hparams.')

    exp_metadata = pd.read_csv(experiment_dir+'/experiment_metadata.csv')
    exp = exp_metadata[exp_metadata['exp_id']==args.exp_id]
    _exp_name = exp['exp_name'].values[0]
    args.resample_rate = exp['resample_rate'].values[0]
    args.model_arch = exp['model_arch'].values[0]
    args.equal_weighted = exp['equal_weighted'].values[0]
    args.training_epochs = exp['training_epochs'].values[0]
    args.working_directory = setup_system(args.gpu, args.seed, experiment_dir, _exp_name)

    #Get hyperparameter data for experiment runs
    result_file = '/'.join([experiment_dir, _exp_name, 'results', 'results_summary_'+str(args.exp_id)+'.csv'])
    results_df = pd.read_csv(result_file)
    results_df = results_df[results_df['run_name'] == args.trained_model_path]
    
    #Construct input arguments for inference
    for i, x in results_df.iterrows():
        #Construct input arguments for inference
        args.run_name = x['run_name']
        args.frame_length = x['frame_length']
        args.learning_rate =  x['learning_rate']
        args.frame_step =  x['frame_step']
        args.optimizer = x['optimizer']
        args.window_fn = x['window_fn']
        args.pad_end = x['pad_end']
        args.input_features = x['input_features']
        args.mel_bins = x['mel_bins']
        args.mfccs = x['mfccs']

    return args


def representative_data_gen():
  for input_value in dataset.take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]


def quantize_model(model, args, dataset=None, save=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Add optimization
    if args.quantization_optimization is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT] # dynamic range quantization - weights quantized to 8-bits
        if args.quantization_optimization == 'w8':
            print(args.quantization_optimization)
            pass
        elif args.quantization_optimization == 'float16':
            print(args.quantization_optimization)
            converter.target_spec.supported_types = [tf.float16] # weights quantized to 16-bits
        elif 'int8' in args.quantization_optimization:
            print(args.quantization_optimization)
            converter.representative_dataset = dataset # integer quantization with float fallback
            if 'float' in args.quantization_optimization:
                pass
            elif 'full' in args.quantization_optimization:
                print(args.quantization_optimization)
                converter.inference_input_type = tf.int8  # full integer quantization
                converter.inference_output_type = tf.int8
            elif '-16' in args.quantization_optimization:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8] # weights quantized to 8-bits, activations to 16-bits

    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()  

    if save is True:
        os.makedirs(os.path.join(args.working_directory, 'tflite_models/'), exist_ok=True)
        tflite_model_file = pathlib.Path(f"{args.working_directory}/tflite_models/{args.exp_name}-{args.exp_id}.tflite")
        tflite_model_file.write_bytes(tflite_model)
        print('Saved model to: {}'.format(tflite_model_file))

    return interpreter


def prune_model(model, args, prune_end_step):

    if args.pruning_schedule == 'constant_sparsity': # https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/ConstantSparsity
        pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(target_sparsity=args.pruning_final_sparsity, begin_step=0, end_step=prune_end_step, frequency=args.pruning_frequency)
    elif args.pruning_schedule == 'polynomial_decay': # https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/PolynomialDecay 
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                            initial_sparsity=0.0, final_sparsity=args.pruning_final_sparsity, frequency=args.pruning_frequency,
                            begin_step=0, end_step=prune_end_step) #check if begin_step and end_step make sense
    pruning_params = {'pruning_schedule': pruning_schedule}

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    return pruned_model


def convert_to_tflite():
    """
    Convert and save tensorflow saved_model at args.trained_model_path (=run_name) to tflite model.
    """
    parser = get_parser()
    args = parser.parse_args()
    experiment_dir = args.working_directory

    if args.trained_model_path is None:
        return print('ABORTING... Specify the model trained_model_path that you want to compress.')

    exp_metadata = pd.read_csv(experiment_dir+'/experiment_metadata.csv')
    exp = exp_metadata[exp_metadata['exp_id']==args.exp_id]
    _exp_name = exp['exp_name'].values[0]
    args.resample_rate = exp['resample_rate'].values[0]
    args.model_arch = exp['model_arch'].values[0]
    args.equal_weighted = exp['equal_weighted'].values[0]
    args.training_epochs = None
    args.working_directory = setup_system(args.gpu, args.seed, experiment_dir, _exp_name)

    #Get hyperparameter data for experiment run
    result_file = '/'.join([experiment_dir, _exp_name, 'results', 'results_summary_'+str(args.exp_id)+'.csv'])
    results_df = pd.read_csv(result_file)

    x = results_df[results_df['run_name'] == args.trained_model_path]
    #Construct input arguments for inference
    args.run_name = x['run_name'].values[0]
    args.frame_length = x['frame_length'].values[0]
    args.learning_rate =  None
    args.frame_step =  x['frame_step'].values[0]
    args.optimizer = None
    args.window_fn = x['window_fn'].values[0]
    args.pad_end = x['pad_end'].values[0]
    args.input_features = x['input_features'].values[0]
    args.mel_bins = x['mel_bins'].values[0]
    args.mfccs = x['mfccs'].values[0]
    args.trained_model_path = f"{args.working_directory}/models/{args.run_name}.hdf5"

    print(args)

    if args.model_arch == 'cnn':
        model = models.sc_model_cnn(vars(args), saved_model=args.trained_model_path)
    elif args.model_arch == 'low_latency_cnn':
        model = models.sc_model_low_latency_cnn(vars(args), saved_model=args.trained_model_path)
    else:
        print("Model architecture not implemented")

    feature_extractor = extract_features(vars(args))
    ds = get_dataset_object(args.dataset_name, args.resample_rate, 'male,female')
    ds_take100 = ds.domains['female'].test.take(100)
    ds_take100 = ds_take100.map(feature_extractor)
    dataset = [t[0] for t in ds_take100]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    representative_dataset = representative_data_gen()

    if args.quantize is True:
        interpreter = quantize_model(model, args, representative_dataset, True)

if __name__=='__main__':
  convert_to_tflite()