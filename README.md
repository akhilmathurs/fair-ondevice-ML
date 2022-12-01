# Pre- and Post-processing Bias in On-device ML

## About this project

## Experiments

### Datasets
#### Google Speech Commands 
https://www.tensorflow.org/datasets/catalog/speech_commands

- **Languages**: English (en)
- **Accents**: unspecified
- **Gender**: `Female`, `Male` (identified via crowdsourcing)
- **Sample Rate & Format**: 16kHz .wav
- **Keywords**: same as original dataset ["bed":0, "bird":1, "cat":2, "dog":3, "down":4, "eight":5, "five":6, "four":7, "go":8, "happy":9, "house":10, "left":11, "marvin":12, "nine":13, "no":14, "off":15, "on":16, "one":17, "right":18, "seven":19, "sheila":20, "six":21, "learn":22, "stop":23, "three":24, "tree":25, "two":26, "up":27, "wow":28, "yes":29, "zero":30, "backward":31, "follow":32, "forward":33, "visual":34]
- **Training/Validation/Testing splits**: same as original dataset, training set equally weighted for females and males with `tf.data.Dataset.sample_from_datasets(.... weights=[0.5, 0.5])` 

#### Multilingual Spoken Words Corpus
https://mlcommons.org/en/multilingual-spoken-words/
- **Languages**: Kinyarwanda (rw), French (fr), German (de), English (en)
- **Accents**: unspecified
- **Gender**: self-identified gender `FEMALE`, `MALE` (`OTHER`, `NONE` excluded) 
- **Sample Rate & Format**: converted from 48kHz .opus files to 16kHz .wav files </br>
used `/scripts/convert_opus2wav.sh` with data directory `/data/mswc`
- **Keywords**: selected 50 keywords with the most utterances for `FEMALE` and `MALE` gender labels </br>
randomly sampled equal number of utterance for females and males per keyword to have gender balance </br>
utterance count per keyword determined by the lesser of total female and male utterances for that keyword
- **Training/Validation/Testing splits** (selected separately for females and males): 0.8 / 0.5 of remainder / remainder 

NB: As there are more male than female speakers in the dataset, the diversity of male speakers is greater than the diversity of female speakers. We did not account for this in these experiments.


### Models

## Findings

## Run the code
We've set up the project to run in a docker container with the latest `tensorflow-gpu` image. The container can be launched (with an example script) by running `. run_container.sh`. 

#### Setup
Once the project workspace has been created, set up the environment:
1. Install libraries and dependencies: `pip install -r requirements.txt`
2. Install librosa dependencies: 
```
apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev \
```
3. If you are adding a new dataset, add the speech commands and path to the dataset to the `_COMMANDS` and `_DATA` variables in `fair_embedded_ml/io_ops.py`

#### Testing
* Add your bash command to `scripts/run_train_eval.sh` (follow example provided in the file). 
* This script runs `fair_embedded_ml.train_and_eval.main()`. 
* It only tests the training and evaluation loops, not logging and outputs.

#### Running experiments 
* Add your bash commands to `scripts/run_hparam.sh` (follow example provided in the file). 
* This script runs `fair_embedded_ml.hparam_tuning.py`. 
* All results are saved and logged to `working_directory/experiment_metadata.csv` and the directories created for the experiment (`working_directory/experiment/..`).
 
