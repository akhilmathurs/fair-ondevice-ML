import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import pdb

'''
Pipeline inspired by https://www.tensorflow.org/tutorials/audio/simple_audio
'''

_COMMANDS={'speech_commands_gender':{"bed":0, "bird":1, "cat":2, "dog":3, "down":4, "eight":5, "five":6, "four":7, "go":8, "happy":9, "house":10, "left":11, "marvin":12, "nine":13, "no":14, "off":15, "on":16, "one":17, "right":18, "seven":19, "sheila":20, "six":21, "learn":22, "stop":23, "three":24, "tree":25, "two":26, "up":27, "wow":28, "yes":29, "zero":30, "backward":31, "follow":32, "forward":33, "visual":34},
            'speech_commands_rain':{"bed":0, "bird":1, "cat":2, "dog":3, "down":4, "eight":5, "five":6, "four":7, "go":8, "happy":9, "house":10, "left":11, "marvin":12, "nine":13, "no":14, "off":15, "on":16, "one":17, "right":18, "seven":19, "sheila":20, "six":21, "learn":22, "stop":23, "three":24, "tree":25, "two":26, "up":27, "wow":28, "yes":29, "zero":30, "backward":31, "follow":32, "forward":33, "visual":34},
            'speech_commands_wind':{"bed":0, "bird":1, "cat":2, "dog":3, "down":4, "eight":5, "five":6, "four":7, "go":8, "happy":9, "house":10, "left":11, "marvin":12, "nine":13, "no":14, "off":15, "on":16, "one":17, "right":18, "seven":19, "sheila":20, "six":21, "learn":22, "stop":23, "three":24, "tree":25, "two":26, "up":27, "wow":28, "yes":29, "zero":30, "backward":31, "follow":32, "forward":33, "visual":34},
            'speech_commands_laughter':{"bed":0, "bird":1, "cat":2, "dog":3, "down":4, "eight":5, "five":6, "four":7, "go":8, "happy":9, "house":10, "left":11, "marvin":12, "nine":13, "no":14, "off":15, "on":16, "one":17, "right":18, "seven":19, "sheila":20, "six":21, "learn":22, "stop":23, "three":24, "tree":25, "two":26, "up":27, "wow":28, "yes":29, "zero":30, "backward":31, "follow":32, "forward":33, "visual":34},
            # 'mswc_rw':{'aba': 0, 'abantu': 1, 'aho': 2, 'ari': 3, 'ariko': 4, 'ati': 5, 'avuga': 6, 'bari': 7, 'benshi': 8, 'buryo': 9, 'byo': 10, 'cya': 11, 'cyane': 12, 'cyangwa': 13, 'gihe': 14, 'gukora': 15, 'gusa': 16, 'hari': 17, 'ibi': 18, 'ibyo': 19, 'icyo': 20, 'igihe': 21, 'iki': 22, 'imana': 23, 'imbere': 24, 'iyi': 25, 'iyo': 26, 'kandi': 27, 'kuba': 28, 'kugira': 29, 'kuko': 30, 'kuri': 31, 'mbere': 32, 'muri': 33, 'ndetse': 34, 'neza': 35, 'ngo': 36, 'nta': 37, 'ntabwo': 38, 'nyuma': 39, 'perezida': 40, 'rwanda': 41, 'ubu': 42, 'ubwo': 43, 'uko': 44, 'umuntu': 45, 'uyu': 46, 'yagize': 47, 'yari': 48, 'yavuze': 49},
            # 'mswc_fr':{'alors': 0, 'aussi': 1, 'aux': 2, 'avec': 3, 'bien': 4, 'cent': 5, 'ces': 6, 'cette': 7, 'comme': 8, 'c’est': 9, 'dans': 10, 'des': 11, 'deux': 12, 'dix': 13, 'elle': 14, 'est': 15, 'fait': 16, 'huit': 17, 'ils': 18, 'les': 19, 'lui': 20, 'mais': 21, 'mille': 22, 'monsieur': 23, 'même': 24, 'nous': 25, 'numéro': 26, 'par': 27, 'pas': 28, 'plus': 29, 'pour': 30, 'quatre': 31, 'que': 32, 'qui': 33, 'rue': 34, 'saint': 35, 'sept': 36, 'ses': 37, 'soixante': 38, 'son': 39, 'sont': 40, 'sur': 41, 'tout': 42, 'trois': 43, 'une': 44, 'vingt': 45, 'vous': 46, 'également': 47, 'était': 48, 'été': 49},
            # 'mswc_de':{'aber': 0, 'als': 1, 'auch': 2, 'auf': 3, 'aus': 4, 'bei': 5, 'das': 6, 'dass': 7, 'dem': 8, 'den': 9, 'der': 10, 'des': 11, 'die': 12, 'diese': 13, 'durch': 14, 'ein': 15, 'eine': 16, 'einem': 17, 'einen': 18, 'einer': 19, 'für': 20, 'haben': 21, 'hat': 22, 'hauptstadt': 23, 'ich': 24, 'ist': 25, 'kann': 26, 'man': 27, 'mit': 28, 'muss': 29, 'nach': 30, 'nicht': 31, 'noch': 32, 'nur': 33, 'schon': 34, 'sein': 35, 'sich': 36, 'sie': 37, 'sind': 38, 'und': 39, 'von': 40, 'war': 41, 'was': 42, 'wenn': 43, 'werden': 44, 'wie': 45, 'wir': 46, 'wird': 47, 'wurde': 48, 'zum': 49},
            # 'mswc_en':{'about': 0, 'after': 1, 'all': 2, 'also': 3, 'and': 4, 'are': 5, 'been': 6, 'boy': 7, 'but': 8, 'can': 9, 'did': 10, 'first': 11, 'for': 12, 'from': 13, 'had': 14, 'have': 15, 'her': 16, 'him': 17, 'his': 18, 'how': 19, 'its': 20, 'know': 21, 'like': 22, 'man': 23, 'many': 24, 'more': 25, 'new': 26, 'not': 27, 'one': 28, 'other': 29, 'out': 30, 'said': 31, 'some': 32, 'that': 33, 'their': 34, 'there': 35, 'these': 36, 'they': 37, 'this': 38, 'time': 39, 'two': 40, 'was': 41, 'were': 42, 'what': 43, 'when': 44, 'will': 45, 'with': 46, 'would': 47, 'you': 48, 'your': 49},
            'mswc_rw':{'abantu': 0, 'ariko': 1, 'avuga': 2, 'bari': 3, 'benshi': 4, 'buryo': 5, 'cyane': 6, 'gihe': 7, 'gukora': 8, 'gusa': 9, 'hari': 10, 'ibyo': 11, 'icyo': 12, 'igihe': 13, 'imana': 14, 'imbere': 15, 'kandi': 16, 'kuba': 17, 'kugira': 18, 'kuko': 19, 'kuri': 20, 'mbere': 21, 'muri': 22, 'ndetse': 23, 'neza': 24, 'ntabwo': 25, 'nyuma': 26, 'perezida': 27, 'rwanda': 28, 'ubwo': 29, 'umuntu': 30, 'umwe': 31, 'yagize': 32, 'yari': 33, 'yavuze': 34},
            'mswc_fr':{'alors': 0, 'aussi': 1, 'avec': 2, 'bien': 3, 'cent': 4, 'cette': 5, 'comme': 6, 'c’est': 7, 'dans': 8, 'deux': 9, 'donc': 10, 'elle': 11, 'fait': 12, 'huit': 13, 'mais': 14, 'mille': 15, 'monsieur': 16, 'même': 17, 'nous': 18, 'numéro': 19, 'plus': 20, 'pour': 21, 'quatre': 22, 'saint': 23, 'sept': 24, 'soixante': 25, 'sont': 26, 'tout': 27, 'trois': 28, 'très': 29, 'vingt': 30, 'vous': 31, 'également': 32, 'était': 33, 'être': 34},
            'mswc_de':{'aber': 0, 'alle': 1, 'auch': 2, 'dann': 3, 'dass': 4, 'diese': 5, 'doch': 6, 'durch': 7, 'eine': 8, 'gibt': 9, 'haben': 10, 'hauptstadt': 11, 'heute': 12, 'hier': 13, 'immer': 14, 'jetzt': 15, 'kann': 16, 'können': 17, 'mehr': 18, 'muss': 19, 'nach': 20, 'nicht': 21, 'noch': 22, 'oder': 23, 'schon': 24, 'sein': 25, 'sich': 26, 'sind': 27, 'wenn': 28, 'werden': 29, 'wieder': 30, 'wird': 31, 'wurde': 32, 'zwei': 33, 'über': 34},
            'mswc_en':{'about': 0, 'after': 1, 'also': 2, 'been': 3, 'could': 4, 'first': 5, 'from': 6, 'have': 7, 'however': 8, 'just': 9, 'know': 10, 'like': 11, 'many': 12, 'more': 13, 'most': 14, 'only': 15, 'other': 16, 'over': 17, 'people': 18, 'said': 19, 'school': 20, 'some': 21, 'that': 22, 'they': 23, 'this': 24, 'three': 25, 'time': 26, 'used': 27, 'were': 28, 'what': 29, 'when': 30, 'will': 31, 'with': 32, 'would': 33, 'your': 34}
            }

_DATA={'speech_commands_gender': {'path':'/data/speech_commands_v2/', 'sample_rate':16000},
        'audio_mnist_gender':{'path':'/data/audio_mnist', 'sample_rate':48000}, 
        'speech_commands_rain': {'path':'/data/speech_commands_v2_rain/', 'sample_rate':16000},
        'speech_commands_wind': {'path':'/data/speech_commands_v2_wind/', 'sample_rate':16000},
        'speech_commands_laughter': {'path':'/data/speech_commands_v2_laughter/', 'sample_rate':16000},
        'mswc_rw':{'path':'/data/mswc_rw/', 'sample_rate':16000},
        'mswc_fr':{'path':'/data/mswc_fr/', 'sample_rate':16000},
        'mswc_de':{'path':'/data/mswc_de/', 'sample_rate':16000},
        'mswc_en':{'path':'/data/mswc_en/', 'sample_rate':16000}
        }

class Domain(object):
    def __init__(self, name, train, val, test):
        self.name = name
        self.train = train
        self.val = val
        self.test = test

class Dataset(object):
    def __init__(self, name, resample_rate, domains=None):
        self.domains = {}
        self.base_dir = _DATA[name]['path']
        self.sample_rate = _DATA[name]['sample_rate']
        self.resample_rate = resample_rate        
        if self.resample_rate is None:
            self.resample_rate = self.sample_rate
        AUTOTUNE = tf.data.AUTOTUNE
        key, value = list(zip(*_COMMANDS[name].items()))
        self.table = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(key),
            values=tf.constant(value),
        ),
        default_value=tf.constant(-1),
        )  

        if 'speech_commands' in name:
            if domains is None:
                domains = ['male', 'female']
                
            for domain_name in domains:
                train_files = self.get_file_names(self.base_dir + os.path.sep +  'training_list_' + domain_name + '.txt')
                test_files = self.get_file_names(self.base_dir + os.path.sep + 'testing_list_' + domain_name + '.txt')
                val_files = self.get_file_names(self.base_dir + os.path.sep + 'validation_list_' + domain_name + '.txt')

                train_ds = tf.data.Dataset.from_tensor_slices(train_files)
                test_ds = tf.data.Dataset.from_tensor_slices(test_files)
                val_ds = tf.data.Dataset.from_tensor_slices(val_files)

                train_ds = train_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
                test_ds = test_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
                val_ds = val_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)

                self.domains[domain_name] = Domain(domain_name, train_ds, val_ds, test_ds)

        elif 'mswc' in name:
            if domains is None:
                domains = ['male', 'female']

            for domain_name in domains:
                train_files = self.get_file_names(self.base_dir + os.path.sep +  'training_list_' + domain_name + '.txt')
                test_files = self.get_file_names(self.base_dir + os.path.sep + 'testing_list_' + domain_name + '.txt')
                val_files = self.get_file_names(self.base_dir + os.path.sep + 'validation_list_' + domain_name + '.txt')

                train_ds = tf.data.Dataset.from_tensor_slices(train_files)
                test_ds = tf.data.Dataset.from_tensor_slices(test_files)
                val_ds = tf.data.Dataset.from_tensor_slices(val_files)

                train_ds = train_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
                test_ds = test_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
                val_ds = val_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)

                self.domains[domain_name] = Domain(domain_name, train_ds, val_ds, test_ds)

        elif name == 'audio_mnist_gender':
            if domains is None:
                domains = ['male', 'female']

            for domain_name in domains:
                test_files = self.get_file_names(self.base_dir + os.path.sep + 'testing_list_' + domain_name + '.txt')

                train_ds = None
                val_ds = None
                test_ds = tf.data.Dataset.from_tensor_slices(test_files)
                test_ds = test_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)

                self.domains[domain_name] = Domain(domain_name, train_ds, val_ds, test_ds)

        else:
            raise ValueError("Cannot process this dataset.")


    def get_file_names(self, file_path):
        filenames = tf.io.gfile.GFile(file_path, mode='r').readlines()
        filenames = [f.replace("\n", "") for f in filenames]
        filenames = tf.random.shuffle(filenames)
        return filenames

    def librosa_resample(self, sample):
        resampled_waveform = librosa.resample(sample, orig_sr=self.sample_rate, target_sr=self.resample_rate, res_type='scipy')
        return resampled_waveform

    def decode_audio(self, audio_binary):
        waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=self.sample_rate)
        waveform = tf.squeeze(waveform, axis=-1)
        if self.resample_rate == self.sample_rate:
            pass
        else:
            #waveform = tfio.audio.resample(waveform, rate_in=self.sample_rate, rate_out=self.resample_rate)
            waveform = tf.numpy_function(func=self.librosa_resample, inp=[waveform], Tout=tf.float32)
            waveform.set_shape(self.resample_rate,)
        return waveform

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return self.table.lookup(parts[0])

    def get_waveform_and_label(self, file_path):
        label = self.get_label(file_path)
        audio_binary = tf.io.read_file(self.base_dir + os.path.sep + file_path)
        waveform = self.decode_audio(audio_binary)
        return waveform, label


def load(dataset_name, resample_rate, domains=None):
    return Dataset(dataset_name, resample_rate=resample_rate, domains=domains)

if __name__ == '__main__':
    # Set seed for experiment reproducibility
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for sel_gpu in gpus:
                tf.config.experimental.set_memory_growth(sel_gpu, True)
        except RuntimeError as e:
            print(e)

    dataset_name = 'speech_commands_gender'
    gender_dataset = Dataset(dataset_name, resample_rate=16000, domains=['male', 'female'])
    pdb.set_trace()