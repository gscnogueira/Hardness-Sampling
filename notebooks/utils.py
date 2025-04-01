import os

DATA_DIR = '../datasets/'
RESULTS_V2_PATH = '../results/v2'

DATASET_LIST = [ f.strip() for f in open('../datasets/datasets.txt') ]

ABREV_DICT = {
    'random_sampling': 'Rnd',
    'margin_sampling': 'MS',
    'training_utility_sampling': 'TU',
    'density_weighted_sampling': 'ID',
    'expected_error_reduction': 'EER$_{ent}$',
    'borderline_points_sampling': 'N1I',
    'k_disagreeing_neighbors_sampling': 'kDN',
    'local_set_cardinality_sampling': 'LSCI',
    'ls_radius_sampling': 'LSR',
    'intra_extra_ratio_sampling': r'N2I',
    'harmfulness_sampling': 'H',
    'usefulness_sampling': 'U',
    'class_likelihood_sampling': 'CL',
    'class_likeliood_diff_sampling': 'CLD',
    'f1_sampling': 'F1I',
    'f2_sampling': 'F2I',
    'f3_sampling': 'F3I',
    'f4_sampling': 'F4I',
    'disjunct_class_percentage_sampling': 'DCP',
    'disjunct_size_sampling': 'DS',
    'tree_depth_pruned_sampling': 'TD$_P$',
    'tree_depth_unpruned_sampling': 'TD$_U$',
    'class_balance_sampling': 'CB',
    'minority_value_sampling': 'MV'
}

CLASSIC = { 'margin_sampling', 'training_utility_sampling', 'density_weighted_sampling', 'expected_error_reduction', 'random_sampling'}
NEIGHBORHOOD = {'borderline_points_sampling', 'k_disagreeing_neighbors_sampling', 'local_set_cardinality_sampling',
                'ls_radius_sampling', 'intra_extra_ratio_sampling', 'harmfulness_sampling', 'usefulness_sampling'}
LIKELIHOOD = {'class_likelihood_sampling', 'class_likeliood_diff_sampling'}
FEATURE_BASED = {'f1_sampling', 'f2_sampling', 'f3_sampling', 'f4_sampling'}
TREE_BASED = {'disjunct_class_percentage_sampling', 'disjunct_size_sampling', 'tree_depth_pruned_sampling',
              'tree_depth_unpruned_sampling'}
CLASS_BALANCE = {'class_balance_sampling', 'minority_value_sampling'}
ABREV_MODEL = {
   'GaussianNB': 'NB',
   'SVC': 'SVM',
   'DecisionTree': 'CART',
   '5NN': '5NN',
}

def get_hm_type(x: str):
    if x in CLASSIC: return 'Classic'
    if x in NEIGHBORHOOD: return 'Neighbor-based'
    if x in LIKELIHOOD: return 'Likelihood-based'
    if x in FEATURE_BASED: return 'Feature-based'
    if x in TREE_BASED: return 'Tree-based'
    if x in CLASS_BALANCE: return 'Class-Balance'
        
def get_style(method):
    if method == 'random_sampling':
        return '--'
    elif method in CLASSIC:
        return '-'
    elif method in NEIGHBORHOOD:
        return '^-'
    elif method in LIKELIHOOD:
        return 's-'
    elif method in FEATURE_BASED:
        return '*-'
    elif method in TREE_BASED:
        return 'h-'
    else:
        return 'd-'

def get_hm_hatch(x: str):
    if x == 'Neighbor-based': return '///'
    if x == 'Likelihood-based':  return '||||||'
    if x == 'Feature-based': return '//////'
    if x == 'Tree-based' : return '\\\\\\\\\\\\\\'
    if x == 'Class-Balance': return '\\\\'