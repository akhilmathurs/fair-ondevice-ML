from sklearn.metrics import roc_auc_score, matthews_corrcoef, cohen_kappa_score, f1_score, precision_score, recall_score
import numpy as np


def model_metrics(labels, predictions, sample_weight=None, return_dict=False):
    """
    Evaluate the prediction results of a model with metrics from sklearn:
    matthews_corrcoef, roc_auc_score, f1_score (weighted), cohen_kappa_score, precision_score, recall_score
    
    Parameters:
        labels (array): the ground-truth labels
        predictions (array): predictions made by the model
        return_dict (bool): whether to return the results in dictionary form (return a tuple if False), default==True
    
    Return:
        dictionary with one entry for each metric if return_dict=True
        tuple with one entry for each metric if return_dict=False
    """

    try:
        predictions_argmax = np.argmax(predictions, axis=1)
    except:
        predictions_argmax = predictions

    mcc = matthews_corrcoef(labels, predictions_argmax, sample_weight=sample_weight)
#    roc_auc = roc_auc_score(labels, predictions, average='weighted', sample_weight=sample_weight, max_fpr=None, multi_class='ovr')
    f1_weighted = f1_score(labels, predictions_argmax, average='weighted')
    cohen_kappa = cohen_kappa_score(labels, predictions_argmax, sample_weight=sample_weight)
    precision = precision_score(labels, predictions_argmax, average='weighted', sample_weight=sample_weight, zero_division=0)
    recall = recall_score(labels, predictions_argmax, average='weighted', sample_weight=sample_weight)
    
    if return_dict:
        return {
          'mcc': mcc, 
        #          'roc_auc_weighted': roc_auc, 
          'f1_weighted': f1_weighted, 
          'cohen_kappa': cohen_kappa,
          'precision': precision, 
          'recall': recall,
        }
    else:
        return (mcc, f1_weighted, cohen_kappa, precision, recall)


def domain_bias(domain_score, overall_score):
    """
    Subgroup bias is calculated as the natural logarithm of the ratio
    of the subgroup's predictive performance score to the overall predictive
    performance score.

    Caveats: 
    The range of interest of metrics should be between [0, 1].
    """
    domain_bias = np.log(domain_score/overall_score)
    return domain_bias


def model_bias(domain_scores:list, overall_score:float):
    """
    Sum the absolute bias of individual domains to calculate model bias.
    """
    model_bias = sum([abs(domain_bias(ds, overall_score)) for ds in domain_scores])
    return model_bias