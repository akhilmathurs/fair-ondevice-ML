# THIS SCRIPT IS FOR TESTING. The script for running experiments is run_hparam.sh !!!! 

#python3 /workspace/fair_embedded_ml/train_and_eval.py --exp_name cnn --equal_weighted --training_epochs 10 --model_arch cnn --input_features mfcc
# python3 /workspace/fair_embedded_ml/train_and_eval.py --exp_name sc8_cnn-sc8 --equal_weighted --model_arch cnn --exp_id 1628708415 --dataset_name speech_commands_gender --resample_rate 8000 --gpu 0 --trained_model_path experiments/sc8_cnn/models/run-1628708415.hdf5 --frame_length 0.02 --frame_step 0.4 --window_fn hamming --input_features log_mel_spectrogram --mel_bins 20 --compress

# 12 September 2022 MSWC Experiments Test
python3 /workspace/fair_embedded_ml/train_and_eval.py --exp_name mswc_test --equal_weighted --training_epochs 10 --model_arch cnn --input_features mfcc --dataset_name mswc_rw --resample_rate 16000 --num_classes 50