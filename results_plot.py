# author: wiebke toussaint
# created: 17 november 2021

import os, sys
import re

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
    
from fair_embedded_ml.metrics import domain_bias, model_bias
from fair_embedded_ml.results_analysis import select_results_by_name, generate_importance_tables 


def plot_param_performance(results, hparam, metric, exp_names:dict, selection='all_results', domain='all', x_axis='exp', save=False):
    
    results = select_results_by_name(results, exp_names, selection)
    results = results.sort_values(by=['exp_name', hparam])
    
    if x_axis == 'exp':
        title = hparam; x = 'exp_name'; hue = hparam
    elif x_axis == 'hparam':
        title = 'experiment'; x = hparam; hue = 'exp_name'
        
    metric = '_'.join([domain, metric])
    hp_vals = results[hparam].unique()
        
    # Initialize the figure
    f, ax = plt.subplots(figsize=(7*np.log(len(hp_vals))+3,5))
    sns.despine(bottom=True, left=True)

    # Show the conditional means
    if hue=='exp_name': 
        palette="tab20"
    else: 
        palette="tab10"
    sns.violinplot(x=metric, y=x, hue=hue, data=results, dodge=True, palette=palette, inner="quartile", scale='area',
                   scale_hue=False, bw=.2, linewidth=1)

    # Improve the legend 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(handles)], labels[:len(labels)], title=title, handletextpad=1,
              columnspacing=0, loc="lower left", ncol=1, frameon=True, bbox_to_anchor=(1, 0.7))
    ax.set_title('{}: {}'.format(metric, hparam))

    if save is True:
        os.makedirs('/figures', exist_ok=True)
        plt.savefig('/figures/{}_{}.png'.format(metric, hparam))
    
    return f, ax


def plot_domain_performance(results, metric:str, domain:str, hue:str, exp_names:dict, results_selection='no_filter', filter_key=None, filter_value=None, style=None, save=False):
    
    results = select_results_by_name(results, exp_names, results_selection)
    results = results.sort_values(by=['exp_name', hue])
    
    metric_list = ['_'.join([x, metric]) for x in domain.split(', ')]
    
    if filter_key is None:
        data = results
    elif filter_value == 'all':
        data = results
    elif filter_key == 'filter input':
        data = results[results['input_features'] == filter_value]
    elif filter_key == 'filter architecture':
        data = results[results['model_arch'] == filter_value] 
    elif filter_key == 'domains':
        data = results[results['domain'].apply(lambda x: x.split('_')[0]) == filter_value]

    g = sns.relplot(data=data, x=metric_list[0], y=metric_list[1], hue=hue, col="exp_name", style=style, 
                    col_wrap=round(len(data['exp_name'].unique())/2), palette="tab10", height=4.5, aspect=1)
    for ax in g.axes.flat:
        ax.axline((0.89,0.89), (0.9,0.9), c="black", ls='--', linewidth=0.8)
        
        
def plot_param_importance(df, dataset, metrics:dict, parameters:str, select_tables:list, fcrit, save_fig=False, **kwargs):
    
    if 'palette' in kwargs.keys():
        palette = kwargs['palette']
    else:
        palette = ['orange', 'blue']
    
    data = pd.DataFrame()
    for m in metrics.keys():
        for arch in ['cnn','low_latency_cnn']:    
            imp_tab = generate_importance_tables(df, dataset, m, parameters=parameters, model_arch=arch)
            for sr in select_tables:
                    df_sr = imp_tab[sr].reset_index()
                    df_sr = df_sr.rename({'index':'parameters'}, axis='columns')
                    df_sr['exp_name'] = sr.split('000_')[0]+'k '+arch
                    df_sr['metric'] = metrics[m]
                    df_sr['arch'] = '{}{}'.format(arch[:-3], arch[-3:].upper()).replace('_', ' ')
                    df_sr['sample_rate'] = sr.split('_')[0]
                    data = pd.concat([data, df_sr], axis=0)
    cat_order = np.sort(data['parameters'].unique())
    g = sns.catplot(data=data.sort_values(by=['arch','sample_rate']), x='parameters', y='F Score', hue='metric', col='exp_name', kind='bar', 
                    order= cat_order, aspect=1.2,#['trained_model_path','pruning_learning_rate','pruning_schedule','pruning_frequency', 'pruning_final_sparsity'], 
                    palette = palette)
    
    for ax in g.axes.flat:
        ax.set_yscale('log')
        if 'pretty_params' in kwargs.keys():
            assert isinstance(kwargs['pretty_params'], dict), "pretty_params should be a dictionary"
            xlabels = [kwargs['pretty_params'][x.get_text()] for x in ax.get_xticklabels()]
        else:
            xlabels = ax.get_xticklabels()
        ax.set_xticklabels(xlabels, rotation=45, horizontalalignment='right')
        ax.hlines(y=fcrit, xmin=-0.5, xmax=0.9*len(cat_order), color="black", ls='--', linewidth=2)
        ax.set_xlabel('')
        ax.set_ylabel('$F\ Score$ (log scale)')
        ax.set_title('{}{}'.format(ax.get_title().split(' = ')[-1][:-3], ax.get_title().split(' = ')[-1][-3:].upper()).replace('low_latency_', 'll'))
        for bar in ax.patches:
            if bar.get_height() < fcrit:
                bar.set_color('lightgrey') 
    
    if 'plot_title' in kwargs.keys():
        plot_title = kwargs['plot_title']
    else:
        plot_title = "Parameter Importance"
    title = plt.suptitle(plot_title, fontsize='x-large', va='top', y=1.1)

    if save_fig == True:
        plt.savefig('figures/pruning_param_importance.png',bbox_inches='tight')


def unique_sorted_values(array):
    unique = array.unique().tolist()
    unique.sort()
    return unique

def relplot_pruning_results(results_compress, pruning_experiment):

    sns.set_context('paper', font_scale=1.3)
    output = widgets.Output()
    cols = list(results_compress.columns)

    # model_arch = widgets.SelectMultiple(description='models', options = list(models.keys()), value = list(models.keys()), rows=4)
    model_arch = widgets.SelectMultiple(description='models', options = pruning_experiment, value = pruning_experiment, rows=4)
    resample_rate = widgets.SelectMultiple(description='resample rate', options = unique_sorted_values(results_compress['resample_rate']), 
                                           value = unique_sorted_values(results_compress['resample_rate']), rows=2)
    model_selected_because = widgets.SelectMultiple(description='select because', options = unique_sorted_values(results_compress['model_selected_because']), 
                                                value = unique_sorted_values(results_compress['model_selected_because'])[:3], rows=4)
    hue_options = widgets.RadioButtons(description='colour', options = ['trained_model_path','pruning_learning_rate','pruning_schedule',
                                                                        'pruning_final_sparsity','pruning_frequency','quantization_optimization'], value = 'trained_model_path', rows=6)
    row_options = widgets.RadioButtons(description='rows', options = [None, 'model_arch','resample_rate'], value = 'resample_rate', rows=3)
    axes_options = widgets.RadioButtons(description='axes value', options = ['absolute', 'delta','subgroup accuracy'], value = 'delta', rows=2)
    equal_weighted = widgets.RadioButtons(description='equal weighted', options = [True, False], value = True, rows=2)

    input_widgets = widgets.HBox([widgets.VBox([axes_options, row_options]), widgets.VBox([model_selected_because, model_arch]), 
                                  widgets.VBox([equal_weighted, hue_options])])
    def common_filtering(model_arch, model_selected_because, equal_weighted, resample_rate, hue_options, axes_options, row_options):
        output.clear_output()
        kwargs = dict(zip(['model_arch','model_selected_because'],[ model_arch, model_selected_because]))
        common_filter = results_compress
        for k, v in kwargs.items():
            if k == 'model_arch':
                k = 'exp_name'
                # v = [exp for exp in unique_sorted_values(results_compress['exp_name']) if any(model in exp for model in [models[val] for val in v])]
                v = [exp for exp in unique_sorted_values(results_compress['exp_name']) if any(model in exp for model in v)]
            common_filter = common_filter.loc[common_filter[k].isin(v), cols]

        common_filter = common_filter[common_filter['equal_weighted']==equal_weighted]
        common_filter.reset_index(drop=True, inplace=True)

        if axes_options == 'absolute':
            x = 'all_mcc'
            y = 'model_bias_pruned'
        elif axes_options == 'delta':
            x = 'delta_all_mcc'
            y = 'delta_model_bias'
        elif axes_options == 'subgroup accuracy':
            x = 'male_mcc'
            y = 'female_mcc'

        g = sns.relplot(data=common_filter, x=x, y=y, hue=hue_options, row=row_options, col='model_selected_because', 
                        col_order=model_selected_because, palette="tab10", height=4.5, aspect=1.3)
        for ax in g.axes.flat:
            if axes_options == 'delta':
                ax.axhline(y=0, color="black", ls='-', linewidth=0.5)
                ax.axvline(x=0, color="black", ls='-', linewidth=0.5)
            elif axes_options == 'subgroup accuracy':
                ax.axline((0.89,0.89), (0.9,0.9), c="black", ls='--', linewidth=0.5)
                ax.set_xlim([max(0.7, common_filter[x].min()*0.99),common_filter[x].max()*1.01])
                ax.set_ylim([max(0.7, common_filter[y].min()*0.99),common_filter[y].max()*1.01])
        display(g)

    output = widgets.interactive_output(common_filtering, dict(zip(['model_arch','model_selected_because','equal_weighted',
                                                                    'resample_rate','hue_options', 'axes_options','row_options'],
                                                                   [model_arch, model_selected_because, equal_weighted, 
                                                                    resample_rate, hue_options, axes_options, row_options])));

    display(input_widgets, output)
    
    
def boxplot_pruning_results(results_compress, pruning_experiments:list):
    
    sns.set_context('paper', font_scale=2)
    output = widgets.Output()
    
    param_options = widgets.RadioButtons(description='hyperparam', 
                                       options = ['trained_model_path','pruning_learning_rate',
                                                  'pruning_schedule','pruning_final_sparsity',
                                                  'pruning_frequency','dataset_name'], 
                                       value = 'pruning_final_sparsity', rows=5)
    row_options = widgets.RadioButtons(description='row options', 
                                       options = [None,'dataset_name', 'model_selected_because','resample_rate'], 
                                       value = 'model_selected_because', rows=4)
    score_options = widgets.RadioButtons(description='score value', 
                                       options = ['metric (pruned)','delta (trained - pruned)'], 
                                       value = 'delta (trained - pruned)', rows=2)
    
    input_widgets = widgets.HBox([score_options, row_options, param_options])
    
    def common_filtering(param_options, row_options, score_options):
        
        output.clear_output()
    
        if score_options=='metric (pruned)':
            value_vars=['all_mcc', 'model_bias_pruned']
        elif score_options=='delta (trained - pruned)':
            value_vars=['delta_all_mcc', 'delta_model_bias']        
        
        df = results_compress[results_compress.exp_name.isin(pruning_experiments)
                             ].melt(id_vars=results_compress.columns.to_list()[:17]+['model_selected_because'], 
                                    value_vars=value_vars, var_name='score')
        
        g = sns.FacetGrid(df, height=8, aspect=1.5, col='score', row=row_options, sharey=False, sharex=False)
        g.map(sns.boxplot, "model_arch", "value", param_options, palette="tab10", showfliers=False,
              hue_order=df[param_options].unique().sort(), 
              order=results_compress.model_arch.unique(), 
              showmeans=False, meanline=True, meanprops=dict(color="black"),
              saturation=0.7,boxprops=dict(alpha=.3))
        
        n_rows = 1 if row_options is None else len(df[row_options].unique())
        ylims=[g.axes[i,1].get_ylim() for i in range(0, n_rows)]
        
        g.map(sns.stripplot, "model_arch", "value", param_options, palette="tab10", dodge=True,
              hue_order=df[param_options].unique().sort(), 
              order=results_compress.model_arch.unique(), size=1.5)
        ylim_max = max(list(zip(*ylims))[1])
        ylim_min = min(list(zip(*ylims))[0])
        for i in range(0, n_rows):
            g.axes[i,1].set(ylim=(ylim_min,ylim_max))
        
        g.add_legend(loc='upper right')
        display(g)
    
    output = widgets.interactive_output(common_filtering, {'param_options':param_options, 
                                                           'row_options':row_options,
                                                           'score_options':score_options})
    
    display(input_widgets, output)