from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import MutableMapping
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow_model_optimization as tfmot
tf.get_logger().setLevel(logging.ERROR)
import os, datetime, click, pdb, tempfile
import math, signal
import numpy as np
from common_utilities import get_parser, setup_system, get_dataset_object, extract_features, get_model_hparams, quantize_model, prune_model
from metrics import model_metrics
from io_ops import load
import models
from timeit import default_timer as timer
  

def evaluate(ds, model_file_path, feature_extractor, args):

    @tf.function
    def test_step(tensor, labels):
      predictions = model(tensor, training=False)
      predictions = tf.nn.softmax(predictions)
      return predictions, labels

    def predict_step(tensor, labels):
      input_index = interpreter.get_input_details()[0]["index"]
      output_index = interpreter.get_output_details()[0]["index"]
      predictions = []
      for t in tensor:
        start_time = timer()
        t = np.expand_dims(t, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, t)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        prediction = tf.nn.softmax(output()[0])
        predictions.append(prediction)
      return predictions, labels    

    # Select model architecture
    if args.model_arch == 'cnn':
      model = models.sc_model_cnn(vars(args), saved_model=model_file_path)
    elif args.model_arch == 'low_latency_cnn':
      model = models.sc_model_low_latency_cnn(vars(args), saved_model=model_file_path)
    else:
      print("Model architecture not implemented")

    # Apply post-training quantization
    if args.quantize is True:
      print('Quantizing the model')
      interpreter = quantize_model(model, args)

    # Prepare test dataset and evaluate the model
    test_metrics = {}
    test_list = [k for k in ds.domains.keys()]
    all_predictions = []
    all_labels = []

    for test_domain in test_list:
      test_ds = ds.domains[test_domain].test
      test_ds = test_ds.prefetch(tf.data.AUTOTUNE).map(feature_extractor, num_parallel_calls=tf.data.AUTOTUNE).cache()
      test_ds = test_ds.batch(200)
      np_predictions = []
      np_labels = []
      for test_tensor, test_labels in test_ds:
        if args.quantize is True:
          predictions, labels = predict_step(test_tensor, test_labels)
          for p in predictions:
            np_predictions.append(p) 
        else:
          predictions, labels = test_step(test_tensor, test_labels)
          for p in predictions.numpy():
            np_predictions.append(p)
        for l in labels:
          np_labels.append(l)  

      all_predictions.append(np_predictions)
      all_labels.append(np_labels) 

      # Get evaluation metrics for domain dataset
      np_predictions = np.array(np_predictions)
      np_labels = np.array(np_labels) 
      test_metrics[test_domain] = model_metrics(np_labels, np_predictions, return_dict=True) 
      print(args.exp_name, args.dataset_name, args.frame_length, args.frame_step, args.window_fn, args.input_features, args.mel_bins, args.mfccs, test_domain, test_metrics[test_domain]['mcc'])

    # Get evaluation metrics for entire dataset
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    test_metrics['all'] = model_metrics(all_labels, all_predictions, return_dict=True) 
    print(args.exp_name, args.dataset_name, args.frame_length, args.frame_step, args.window_fn, args.input_features, args.mel_bins, args.mfccs, 'all', test_metrics['all']['mcc'])

    return test_metrics


def train(ds, feature_extractor, args):
    @tf.function
    def train_step(tensor, labels):
      with tf.GradientTape(persistent=True) as tape:
        predictions = model(tensor)
        predictions = tf.keras.backend.softmax(predictions)
        loss = loss_object(labels, predictions)

      gradients_e = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients_e, model.trainable_variables))
      train_loss(loss)
      train_accuracy(labels, predictions)

    @tf.function
    def test_step(tensor, labels):
      predictions = model(tensor)
      predictions = tf.keras.backend.softmax(predictions)
      v_loss = loss_object(labels, predictions)
      val_loss(v_loss)
      val_accuracy(labels, predictions)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)

    # Select model architecture
    if args.model_arch == 'cnn':
      model = models.sc_model_cnn(vars(args))
    elif args.model_arch == 'low_latency_cnn':
      model = models.sc_model_low_latency_cnn(vars(args))
    else:
      ValueError("Model arch not implemented")
 
    # Prepare training and validation datasets
    train_ds = None
    train_ds_list = []
    val_ds = None

    for _, domain_obj in ds.domains.items(): # Join dataset domains into a single dataset for training
      train_ds_list.append(domain_obj.train)
      if val_ds is None:
        val_ds = domain_obj.val
      else:
        val_ds = val_ds.concatenate(domain_obj.val)   

    if len(train_ds_list)==1:
        train_ds = train_ds_list[0]
    else:
        if args.equal_weighted is True:
          domain_weights = len(train_ds_list)*[1/len(train_ds_list)]
        else:
          pass #TODO Don't pass silently
        train_ds = tf.data.experimental.sample_from_datasets(train_ds_list, domain_weights, args.seed) #deprecated, need change to tf.data.Dataset.sample... when updating tensorflow

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE).map(feature_extractor, num_parallel_calls=tf.data.AUTOTUNE).cache()
    train_ds = train_ds.shuffle(1000).batch(args.batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE).map(feature_extractor, num_parallel_calls=tf.data.AUTOTUNE).cache()
    val_ds = val_ds.batch(200)

    model_file_path = f"{args.working_directory}/models/{args.run_name}.hdf5"
    min_val_loss = 1000.0

    # Train the model
    for epoch in range(args.training_epochs):

      for tensor, labels in train_ds:
        train_step(tensor, labels)

      for val_tensor, val_labels in val_ds:
        test_step(val_tensor, val_labels)

      template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
      print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, val_loss.result(), val_accuracy.result()*100))

      if train_accuracy.result().numpy() >= 1.0 or val_accuracy.result().numpy() >= 1.0:
          return max_val_accuracy
      
      if val_loss.result().numpy() < min_val_loss:
          model.save(model_file_path)
          min_val_loss = val_loss.result().numpy()

      train_loss.reset_states()
      train_accuracy.reset_states()
      val_loss.reset_states()
      val_accuracy.reset_states()

    return model_file_path


def prune(ds, model_file_path, feature_extractor, args):
    @tf.function
    def train_step(tensor, labels):
      with tf.GradientTape(persistent=True) as tape:
        predictions = model(tensor, training=True)
        predictions = tf.keras.backend.softmax(predictions)
        loss = loss_object(labels, predictions)

      gradients_e = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients_e, model.trainable_variables))
      train_loss(loss)
      train_accuracy(labels, predictions)

    @tf.function
    def test_step(tensor, labels):
      predictions = model(tensor)
      predictions = tf.keras.backend.softmax(predictions)
      v_loss = loss_object(labels, predictions)
      val_loss(v_loss)
      val_accuracy(labels, predictions)

    # Select model architecture
    if args.model_arch == 'cnn':
      model = models.sc_model_cnn(vars(args), saved_model=model_file_path, prune=False)
    elif args.model_arch == 'low_latency_cnn':
      model = models.sc_model_low_latency_cnn(vars(args), saved_model=model_file_path, prune=False)
    else:
      print("Model architecture not implemented")

    # Prepare training and validation datasets
    train_ds = None
    train_ds_list = []
    val_ds = None

    for _, domain_obj in ds.domains.items(): # Join dataset domains into a single dataset for training
      train_ds_list.append(domain_obj.train)
      if val_ds is None:
        val_ds = domain_obj.val
      else:
        val_ds = val_ds.concatenate(domain_obj.val)   

    if len(train_ds_list)==1:
        train_ds = train_ds_list[0]
    else:
        if args.equal_weighted is True:
          domain_weights = len(train_ds_list)*[1/len(train_ds_list)]
        else:
          domain_weights = len(train_ds_list)*[1/len(train_ds_list)]
          #pass #TODO Don't pass silently
        train_ds = tf.data.experimental.sample_from_datasets(train_ds_list, domain_weights, args.seed)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE).map(feature_extractor, num_parallel_calls=tf.data.AUTOTUNE).cache()
    train_ds = train_ds.shuffle(1000).batch(args.batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE).map(feature_extractor, num_parallel_calls=tf.data.AUTOTUNE).cache()
    val_ds = val_ds.batch(200)
    os.makedirs(os.path.join(args.working_directory, 'pruned_models/'), exist_ok=True)
    trained_model_file_path = f"{args.working_directory}/pruned_models/{args.trained_model_path+'-'+args.exp_name+'-'+args.run_name}.hdf5"

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    num_batches = len([batch[1] for batch in train_ds])
    prune_epochs = 10
    prune_end_step = prune_epochs * num_batches

    # NB: from tensorflow github: guideline for custom pruning https://github.com/tensorflow/model-optimization/issues/271
    # The set_model and pruned_model.optimizer setting is unusual and could be missed.
    # Pruning does not work unless we explicitly add training=True argument in the model call
    model = prune_model(model, args, prune_end_step)
    optimizer = tf.keras.optimizers.Adam(learning_rate = args.pruning_learning_rate)
    model.optimizer = optimizer # Assign optimizer with new learning rate to pruned model
    step_callback = tfmot.sparsity.keras.UpdatePruningStep() # Add a pruning step callback to peg the pruning step to the optimizer's step. 
    step_callback.set_model(model)

    # https://colab.research.google.com/github/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/g3doc/guide/pruning/comprehensive_guide.ipynb
    step_callback.on_train_begin()
    for epoch in range(prune_epochs):
      for tensor, labels in train_ds:
        train_step(tensor, labels)
      for val_tensor, val_labels in val_ds:
        test_step(val_tensor, val_labels)

      template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
      print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, val_loss.result(), val_accuracy.result()*100))

      step_callback.on_epoch_end(batch=-1)
    
    model = tfmot.sparsity.keras.strip_pruning(model)
    model.save(trained_model_file_path, include_optimizer=True)

    return trained_model_file_path


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.trained_model_path is None:
      args.working_directory = setup_system(args.gpu, args.seed, args.working_directory, args.exp_name)
      ds = get_dataset_object(args.dataset_name, args.resample_rate, args.domains)
      feature_extractor = extract_features(vars(args))
      if not hasattr(args, 'start_time'):
        args.start_time = str(int(datetime.datetime.now().timestamp()))
      if not hasattr(args, 'run_name'):
        args.run_name = f"run-{args.start_time}"
      trained_model_file_path = train(ds, feature_extractor, args)    
      print(trained_model_file_path)  
    else: 
      args = get_model_hparams(args)
      ds = get_dataset_object(args.dataset_name, args.resample_rate, args.domains)
      feature_extractor = extract_features(vars(args))
      trained_model_file_path = f"{args.working_directory}/models/{args.trained_model_path}.hdf5"
      print(trained_model_file_path)  

    if args.prune is True:
      trained_model_file_path = prune(ds, trained_model_file_path, feature_extractor, args)

    result_dict = evaluate(ds, trained_model_file_path, feature_extractor, args)

    pdb.set_trace()

if __name__=='__main__':
  main()