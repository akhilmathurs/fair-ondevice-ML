# author: wiebke toussaint
# created: 17 november 2021


import os, sys
import re

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import scipy.stats

from fair_embedded_ml.metrics import domain_bias, model_bias

exp_names = dict(
    sc_train=['sc8_cnn', 'sc16_cnn', 'sc8_llcnn', 'sc16_llcnn'], 
    sc_resample=['sc16_cnn-sc8', 'sc16_llcnn-sc8', 'sc8_cnn-sc16', 'sc8_llcnn-sc16'],
    sc_laughter=['sc8_cnn-scl8', 'sc8_llcnn-scl8', 'sc16_cnn-scl16', 'sc16_llcnn-scl16'], 
    sc_wind=['sc8_cnn-scw8', 'sc8_llcnn-scw8', 'sc16_cnn-scw16', 'sc16_llcnn-scw16'],
    sc_rain=['sc8_cnn-scr8', 'sc8_llcnn-scr8', 'sc16_cnn-scr16', 'sc16_llcnn-scr16'], 
    am_gender=['sc8_cnn-am8', 'sc8_llcnn-am8', 'sc16_cnn-am16', 'sc16_llcnn-am16'],
    compress=['sc8_cnn-compress', 'sc16_cnn-compress', 'sc8_llcnn-compress', 'sc16_llcnn-compress'],
    compress_sc=['sc8_cnn-compress_ew2', 'sc16_cnn-compress_ew2', 'sc8_llcnn-compress_ew2', 'sc16_llcnn-compress_ew2'],
    mswc_de=['mswc35_de8_cnn', 'mswc35_de16_cnn', 'mswc35_de8_llcnn', 'mswc35_de16_llcnn'],
    mswc_fr=['mswc35_fr8_cnn', 'mswc35_fr16_cnn', 'mswc35_fr8_llcnn', 'mswc35_fr16_llcnn'],
    mswc_rw=['mswc35_rw8_cnn', 'mswc35_rw16_cnn', 'mswc35_rw8_llcnn', 'mswc35_rw16_llcnn'],
    mswc_en=['mswc35_en8_cnn', 'mswc35_en16_cnn', 'mswc35_en8_llcnn', 'mswc35_en16_llcnn'],
    compress_mswc_de=['mswc_de8_cnn-compress', 'mswc_de16_cnn-compress', 'mswc_de8_llcnn-compress', 'mswc_de16_llcnn-compress'],
    compress_mswc_fr=['mswc_fr8_cnn-compress', 'mswc_fr16_cnn-compress', 'mswc_fr8_llcnn-compress', 'mswc_fr16_llcnn-compress'],
    compress_mswc_en=['mswc_en8_cnn-compress', 'mswc_en16_cnn-compress', 'mswc_en8_llcnn-compress', 'mswc_en16_llcnn-compress'],
    compress_mswc_rw=['mswc_rw8_cnn-compress', 'mswc_rw16_cnn-compress', 'mswc_rw8_llcnn-compress', 'mswc_rw16_llcnn-compress']
)
exp_names['all_results'] = exp_names['sc_train'] + exp_names['sc_resample'] + exp_names['sc_laughter'] + exp_names['sc_wind'] + exp_names['sc_rain'] + exp_names['am_gender'] + exp_names['compress'] + exp_names['compress_sc'] + exp_names['mswc_de'] + exp_names['mswc_fr'] + exp_names['mswc_rw'] + exp_names['mswc_en']


def _get_exp_name(df_row, metadata:str):
    
    if metadata == 'experiment':
        exp_name = df_row['exp_name']
    else:
        exp_name = df_row['exp_name'].split('-')[0]
        if 'mswc_' in exp_name:
            exp_name = exp_name.replace('mswc_','mswc35_')
    
    return exp_name


def _get_file_name(df_row, metadata:str):
    
    if metadata == 'experiment':
        file_name = 'results_summary_'+str(df_row['exp_id'])+'.csv'
    elif metadata == 'inference':
        file_name = '_'.join(['inference', df_row['exp_name'], str(df_row['exp_id'])])+'.csv'    
    elif metadata == 'compression':
        file_name = '_'.join(['compression', df_row['exp_name'], str(df_row['exp_id'])])+'.csv'
    
    return file_name


def _generate_results_df(result_files:dict):
    
    results_list = []
    for k, v in result_files:
        try:
            _id, _name = next(iter(k.items()))
            _df = pd.read_csv(v)
            _df['exp_id'] = _id
            _df['exp_name'] = _name
            results_list.append(_df)
        except FileNotFoundError as e:
            print(e)
    
    results_df = pd.concat(results_list)
    
    return results_df


def join_metadata_and_results(results_dir, metadata:str):
    
    metadf = pd.read_csv(results_dir + metadata + '_metadata.csv')
    metadf['equal_weighted'].fillna(False, inplace = True)
    metadf.drop_duplicates(inplace = True, keep = 'last')
    
    file_list = ['/'.join([results_dir, _get_exp_name(x, metadata), 'results', _get_file_name(x, metadata)]) 
                        for i, x in metadf[metadf[['exp_name', 'exp_id']].duplicated()==False].iterrows()]
    
    names = [{x['exp_id']:x['exp_name']} for i, x in metadf[metadf[['exp_name', 'exp_id']].duplicated()==False].iterrows()]
    result_files = zip(names, file_list)
    results_df = _generate_results_df(result_files)
    
    if metadata == 'compression':
        results_df['quantize'] = np.where(results_df['quantization_optimization'].isna(), False, True)
        results = pd.merge(metadf, results_df, on=['exp_id', 'exp_name', 'trained_model_path', 'quantize'])
    else:
        results = pd.merge(metadf, results_df, on=['exp_id', 'exp_name'])
    
    results['exp_id'] = results['exp_id'].astype(str)
    results.sort_values(by='exp_name', inplace=True)
    
    return results


def get_results(results_dir):
    
    experiment_results = join_metadata_and_results(results_dir, 'experiment')
    inference_results = join_metadata_and_results(results_dir, 'inference')

    results = pd.concat([experiment_results, inference_results])
    results['training_time'] = results['training_time'].where(results['training_time'].notna(), results['inference_time'])
    results.rename({'training_time':'time'}, axis=1, inplace=True)
    results.drop('inference_time', axis=1, inplace=True)
    
    return results


def get_compression_results(results_dir):
    
    compression_results = join_metadata_and_results(results_dir, 'compression')
    compression_results.drop_duplicates(subset=compression_results.iloc[:,-16:].columns, keep='last', inplace=True)
    compression_results.reset_index(inplace=True, drop=True)
    
    return compression_results


def get_results_for_domains(results):
    
    domains = ['all','female','male']
    metrics_start_index = [idx for idx, col_name in enumerate(results.columns) if 'all' in col_name][0] 

    results_mcc = results.iloc[:,:metrics_start_index+len(domains)]
    results_mcc = results_mcc.melt(id_vars=list(results_mcc.columns)[:-len(domains)+1])
    new_cols = list(results_mcc.columns)[:-len(domains)+1] + ['domain','domain_mcc']
    results_mcc.columns = new_cols

    zero_scores=len(results_mcc.loc[results_mcc['all_mcc']==0])
    if zero_scores>0:
        print(f"{zero_scores} experiments have a performance score value of 0. Deleting these experiments from the analysis")
        results_mcc.drop(results_mcc.loc[results_mcc['all_mcc']==0].index, inplace=True)
    
    results_mcc['domain_bias'] = results_mcc.apply(lambda x: domain_bias(x['domain_mcc'],x['all_mcc']), axis=1)
    results_mcc.sort_values(by=['exp_name'], inplace=True)
    results_mcc.reset_index(drop=True, inplace=True)
    
    return results_mcc


def get_bias_results(results_for_domains):
    
    results_bias = results_for_domains.pivot(index=list(results_for_domains.columns)[:-3], columns='domain')
    results_bias.reset_index(inplace=True, col_level=0)
    clevel0 = ([c.split('_') for c in results_bias.columns.get_level_values(0)])
    clevel1 = [c.split('_') for c in results_bias.columns.get_level_values(1)]
    new_cols = ['_'.join([c[1][0], c[0][1]]) if len(c[1][0])>0 else '_'.join(c[0]) for c in zip(clevel0, clevel1)]
    results_bias = results_bias.droplevel(axis=1, level='domain')
    results_bias.columns = new_cols
    results_bias['model_bias'] = results_bias.apply(lambda x: model_bias([x['female_mcc'], x['male_mcc']],x['all_mcc']), axis=1)
    results_bias['model_base'] = results_bias['exp_name'].apply(lambda x: x.split('-')[0])
    results_bias.sort_values('exp_name', inplace=True)

    return results_bias
    

def get_top_result(results, metric, best=1, exp_name='all'):
    
    cols = ['exp_name','model_arch','dataset_name', 'exp_id','run_name',
            'resample_rate','input_features','frame_length','frame_step','mel_bins','mfccs','window_fn',
            'all_mcc','model_bias']
    
    if exp_name == 'all':
        if metric == 'all_mcc':
            top_result = results.loc[results.groupby(['exp_name'])[metric].nlargest(best).index.levels[1], cols]
        elif metric == 'model_bias':
            top_result = results.loc[results.groupby(['exp_name'])[metric].nsmallest(best).index.levels[1], cols]
        top_result.sort_values(by=['exp_name','resample_rate'], inplace=True)
    else:
        if metric == 'all_mcc':
            idx = results.groupby(['exp_name'])[metric].transform(max) == results[metric]
        elif metric == 'model_bias':
            idx = results.groupby(['exp_name'])[metric].transform(min) == results[metric]
        top_result = dict(zip(cols, results.where(results['exp_name'] == exp_name)[idx].dropna(how='all')[cols].values[0]))

    return top_result


def select_results_by_name(results, exp_names:dict, selection:str):

    selected_results = results[results['exp_name'].isin(exp_names[selection])]
    
    return selected_results


def model_selection(df, exp_name, min_percentage_of_mcc):

    best_accuracy = pd.DataFrame.from_dict(get_top_result(df, 'all_mcc', exp_name=exp_name, best=1), orient='index').T
    fairest = pd.DataFrame.from_dict(get_top_result(df, 'model_bias', exp_name=exp_name, best=1), orient='index').T
    mcc_range = select_fairest_models_in_mcc_range(df, exp_name, best=1, min_percentage_of_mcc=min_percentage_of_mcc)
    
    models = pd.DataFrame([best_accuracy.iloc[0], fairest.iloc[0], mcc_range.iloc[0]], 
                          index=['accuracy','bias','>{}% accuracy'.format(min_percentage_of_mcc*100)])
    
    return models


def select_fairest_models_in_mcc_range(df, exp_name, best, min_percentage_of_mcc=0.99):
        
    min_mcc = df.where(df['exp_name'] == exp_name)['all_mcc'].dropna(how='all').nlargest(1).values[0]*min_percentage_of_mcc
    try:
        selected_models = get_top_result(df.where((df['exp_name'] == exp_name) & (df['all_mcc'] >= min_mcc)).dropna(how='all'), 
                          'model_bias', best=best)
    except AttributeError:
        selected_models = pd.DataFrame.from_dict(get_top_result(df, metric='all_mcc', exp_name=exp_name), orient='index').T
        
    return selected_models


def compare_param_performance(df, metric, best_model, compare_model, swap_param:dict=None):
    
    cols = ['exp_name','model_arch','dataset_name', 'exp_id','run_name','resample_rate','input_features','frame_length',
            'frame_step','mel_bins','mfccs','window_fn','all_mcc','male_mcc','female_mcc','model_bias']
    baseline = get_top_result(df, metric, exp_name=best_model, best=1)

    param_dict = {}
    for param in ['input_features','frame_length','frame_step','mel_bins','mfccs','window_fn']:
        param_dict[param] = baseline.get(param)
    if swap_param is not None:
        for k, v in swap_param.items():
            param_dict[k] = v
    
    mod1 = pd.DataFrame.from_records([baseline])
    mod2 = df.loc[(df['exp_name']==compare_model) & (df[list(param_dict)]==pd.Series(param_dict)).all(axis=1), cols]
    comparison = pd.concat([mod1, mod2]).reset_index(drop=True)
    
    return comparison


def fcrit(df, params, significance:float=0.01):
    
    param_vals = {}
    for p in params:
        vals = list(df[p].unique())
        try:
            vals.sort()
        except TypeError:
            pass
        param_vals[p] = vals

    # statistical analysis: F > Fcrit & p < significance value.
    dof = np.prod([len(p) for p in param_vals.values()])-2
    fcrit = scipy.stats.f.ppf(q=1-significance, dfn=1, dfd=dof)
    
    return dof, fcrit


def parameter_importance(df, dataset, metric, sample_rate, x_sampled, model_arch=None, verbose=False):
    """
    Calculate Pearson correlation (f-value) and mutual information for parameters given metric target variable and a sample rate.
    
    INPUTS
    metric: 'all_mcc', 'model_bias'
    sample_rate: 8000, 16000, 'all' ('all' takes all sample rates)
    x_sampled: True, False (consider testing at same sample rate as training if True)
    
    OUTPUTS
    
    f_test [dict]: F values for parameters
    mi [dict]: mutual information values for parameters
    """
    features = ['frame_length', 'frame_step','mel_bins','mfccs','model_arch','window_fn','input_features']
    features_pretty = ['frame length', 'frame step','# Mel bins','MFCCs','architecture','window type','feature type']
    if sample_rate != 'all':
        df = df[df['resample_rate']==sample_rate]
    elif sample_rate == 'all':
        df = df
    if x_sampled is False:
        df = df[(df['dataset_name']==dataset)&(~df['training_epochs'].isna())]
    elif x_sampled is True:
        df = df[(df['dataset_name']==dataset)&(df['training_epochs'].isna())]
    if model_arch is not None:
        features = ['frame_length', 'frame_step','mel_bins','mfccs','window_fn','input_features']
        features_pretty = ['frame length', 'frame step','# Mel bins','MFCCs','window type','feature type']
        df = df[df['model_arch']==model_arch]

    try:
        enc = OrdinalEncoder()
        enc.fit(df[features])
        f_test, p_vals = f_regression(enc.transform(df[features]), y=df[metric])
        mi = mutual_info_regression(enc.transform(df[features]), y=df[metric], random_state=13)
        ridge = make_pipeline(StandardScaler(), RidgeCV()).fit(enc.transform(df[features]), y=df[metric])[1].coef_

        if verbose is True:
            print('F-test scores: {}'.format(metric),'\n')
            for k, v in dict(sorted(dict(zip(features, p_vals)).items(), key=lambda item: item[1], reverse=True)).items(): print(k, v.round(5)) 
            print('\n','Mutual information scores: {}'.format(metric),'\n')
            for k, v in dict(sorted(dict(zip(features, mi)).items(), key=lambda item: item[1], reverse=True)).items(): print(k, v.round(3)) 
            print('\n','Ridge regression coefficients: {}'.format(metric),'\n')
            for k, v in dict(sorted(dict(zip(features, ridge)).items(), key=lambda item: item[1], reverse=True)).items(): print(k, v.round(4)) 

        return dict(zip(features_pretty, f_test.round(3))), dict(zip(features_pretty, p_vals)), dict(zip(features_pretty, mi.round(3))), dict(zip(features_pretty, ridge.round(4)))
    
    except ValueError:
        return dict(zip(features_pretty, np.full(len(features_pretty), np.nan))), dict(zip(features_pretty, np.full(len(features_pretty), np.nan))), dict(zip(features_pretty, np.full(len(features_pretty), np.nan))), dict(zip(features_pretty, np.full(len(features_pretty), np.nan)))



def compression_parameter_importance(df, dataset, metric, sample_rate, compression_type, trained_model_path=False, model_arch=None, verbose=False):
    """
    Calculate Pearson correlation (f-value) and mutual information for parameters given metric target variable, compression type and a sample rate.
    
    INPUTS
    metric: 'all_mcc', 'model_bias'
    compression_type: 'prune','quantize'
    sample_rate: 8000, 16000, None (None takes all sample rates)
    x_sampled: True, False (consider testing at same sample rate as training if True)
    
    OUTPUTS
    f_test [dict]: F values for parameters
    mi [dict]: mutual information values for parameters
    """
    df = df[df['dataset_name']==dataset]
    
    if compression_type == 'prune': 
        features = ['trained_model_path','pruning_learning_rate','pruning_schedule', 'pruning_frequency', 'pruning_final_sparsity','model_arch']
        quantize = False
    elif compression_type == 'quantize':
        features = ['trained_model_path','quantization_optimization','model_arch']
        quantize = True
    
    if sample_rate != 'all':
        df = df[(df['resample_rate']==sample_rate)&(df['quantize']==quantize)]
    elif sample_rate == 'all':
        df = df[df['quantize']==quantize]
    
    if model_arch is not None:
        features = features[:-1]
        df = df[df['model_arch']==model_arch]
        
    if trained_model_path is False:
        features = features[1::]
        # think if this is the correct way of doing this...should maybe iterate through models and then get the mean F score for each hyperparam, 
        # otherwise I'm ignoring the effect of the actual model
    
    features_pretty = features
    
    try:
        enc = OrdinalEncoder()
        enc.fit(df[features])
        enc.categories_
        f_test, p_vals = f_regression(enc.transform(df[features]), y=df[metric])
        mi = mutual_info_regression(enc.transform(df[features]), y=df[metric], random_state=13)
        ridge = make_pipeline(StandardScaler(), RidgeCV()).fit(enc.transform(df[features]), y=df[metric])[1].coef_

        if verbose is True:
            print('F-test scores: {}'.format(metric),'\n')
            for k, v in dict(sorted(dict(zip(features, p_vals)).items(), key=lambda item: item[1], reverse=True)).items(): print(k, v.round(5)) 
            print('\n','Mutual information scores: {}'.format(metric),'\n')
            for k, v in dict(sorted(dict(zip(features, mi)).items(), key=lambda item: item[1], reverse=True)).items(): print(k, v.round(3)) 
            print('\n','Ridge regression coefficients: {}'.format(metric),'\n')
            for k, v in dict(sorted(dict(zip(features, ridge)).items(), key=lambda item: item[1], reverse=True)).items(): print(k, v.round(4)) 

        return dict(zip(features_pretty, f_test.round(3))), dict(zip(features_pretty, p_vals)), dict(zip(features_pretty, mi.round(3))), dict(zip(features_pretty, ridge.round(4)))
    
    except ValueError:
        return dict(zip(features_pretty, np.full(len(features_pretty), np.nan))), dict(zip(features_pretty, np.full(len(features_pretty), np.nan))), dict(zip(features_pretty, np.full(len(features_pretty), np.nan))), dict(zip(features_pretty, np.full(len(features_pretty), np.nan)))
    
    
def generate_importance_tables(df, dataset, metric, parameters="preprocessing", model_arch=None):
    
    if parameters == 'compression':
        param_options = ['prune', 'quantize']
        key_options = dict(zip(param_options, param_options))
        func = compression_parameter_importance
    elif parameters == 'preprocessing':
        param_options = [True, False]
        key_options = dict(zip(param_options, ['x','']))
        func = parameter_importance
    
    importance_tables = {}
    for sr in ['all', 16000, 8000]:
        for p in param_options:
            f_test, p_vals, mi, ridge = df.pipe(func, dataset, metric, sr, p, model_arch=model_arch)            
            key = '_'.join([str(sr), key_options[p]]) 
            importance_tables[key] = pd.DataFrame.from_dict({'F Score':f_test, 'p-val':p_vals,'MI':mi, 'Ridge coef':ridge})#.sort_values(by='F Score', ascending=False)

    return importance_tables



def pruning_results(results, compression_results, pruning_experiment, train_experiment, best_runs, fairest_runs, accurate_fair_runs):
    
    results_compress = pd.merge(compression_results[compression_results['exp_name'].isin(exp_names[pruning_experiment])],
                                results[results['exp_name'].isin(exp_names[train_experiment])],#.iloc[:,6:22], 
                                how='left',left_on=['exp_id','trained_model_path'], right_on=['exp_id','run_name'], suffixes=[None,'_trained'])
    results_compress['equal_weighted'].fillna(False, inplace=True)
    results_compress['model_bias'] = results_compress.apply(lambda x: model_bias([x['female_mcc'], x['male_mcc']],x['all_mcc']), axis=1)
    results_compress['model_bias_trained'] = results_compress.apply(lambda x: model_bias([x['female_mcc_trained'], x['male_mcc_trained']],x['all_mcc_trained']), axis=1)
    results_compress['delta_all_mcc'] = results_compress['all_mcc'] - results_compress['all_mcc_trained']
    results_compress['delta_male_mcc'] = results_compress['male_mcc'] - results_compress['male_mcc_trained']
    results_compress['delta_female_mcc'] = results_compress['female_mcc'] - results_compress['female_mcc_trained']
    results_compress['delta_model_bias'] = results_compress['model_bias'] - results_compress['model_bias_trained']
    results_compress['model_selected_because'] = np.where(results_compress['trained_model_path'].isin(best_runs), 'best', 
                                                        np.where(results_compress['trained_model_path'].isin(fairest_runs), 'fairest',
                                                                          np.where(results_compress['trained_model_path'].isin(accurate_fair_runs), 
                                                                                   'accurate_fair_runs', np.nan)))#)
    results_compress.rename(columns={'model_bias':'model_bias_pruned','model_bias_trained':'model_bias_trained',
                    'delta_model_bias':'delta_model_bias'}, inplace=True)
    
    return results_compress