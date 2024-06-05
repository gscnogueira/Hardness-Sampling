import numpy as np
import pandas as pd
from modAL.utils.selection import multi_argmax
from modAL import ActiveLearner
from pyhard.measures import ClassificationMeasures


def __get_hardness_obj(learner: ActiveLearner, X: np.ndarray) -> pd.DataFrame:
    columns = [f'f_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    y_pred = learner.predict(X)
    print('Classes preditas:', np.unique(y_pred))

    return ClassificationMeasures(X_df.assign(y=y_pred))


def __generic_hardness_sampling(learner: ActiveLearner, X: np.ndarray,
                                measure: str, n_instances=1):
    measures_obj = __get_hardness_obj(learner, X)
    results = getattr(measures_obj, measure)()

    return multi_argmax(results, n_instances)


def k_disagreeing_neighbors_sampling(learner: ActiveLearner, X: np.ndarray,
                                     n_instances=1):
    """Implementation for the kDN hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'k_disagreeing_neighbors', n_instances)


def disjunct_size_sampling(learner, X, n_instances):
    """Implementation for the DS hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'disjunct_size', n_instances)


def disjunct_class_percentage_sampling(learner, X, n_instances):
    """Implementation for the DCP hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'disjunct_class_percentage', n_instances)


def tree_depth_pruned_sampling(learner, X, n_instances):
    """Implementation for the TD_P hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'tree_depth_pruned', n_instances)


def tree_depth_unpruned_sampling(learner, X, n_instances):
    """Implementation for the TD_U hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'tree_depth_unpruned', n_instances)


def class_likelihood_sampling(learner, X, n_instances):
    """Implementation for the CL hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'class_likeliood', n_instances)


def class_likeliood_diff_sampling(learner, X, n_instances):
    """Implementation for the CLD hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'class_likeliood_diff', n_instances)


def minority_value_sampling(learner, X, n_instances):
    """Implementation for the MV hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'minority_value', n_instances)


def class_balance_sampling(learner, X, n_instances):
    """Implementation for the CB hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'class_balance', n_instances)


def borderline_points_sampling(learner, X, n_instances):
    """Implementation for the N1 hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'borderline_points', n_instances)


def intra_extra_ratio_sampling(learner, X, n_instances):
    """Implementation for the N2 hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'intra_extra_ratio', n_instances)


def local_set_cardinality_sampling(learner, X, n_instances):
    """Implementation for the LSC hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'local_set_cardinality', n_instances)


def ls_radius_sampling(learner, X, n_instances):
    """Implementation for the LSR hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'ls_radius', n_instances)


def harmfulness_sampling(learner, X, n_instances):
    """Implementation for the Harmfulness hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'harmfulness', n_instances)


def usefulness_sampling(learner, X, n_instances):
    """Implementation for the Usefulness hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'usefulness', n_instances)


def f1_sampling(learner, X, n_instances):
    """Implementation for the F1 hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'f1', n_instances)


def f2_sampling(learner, X, n_instances):
    """Implementation for the F2 hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'f2', n_instances)


def f3_sampling(learner, X, n_instances):
    """Implementation for the F3 hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'f3', n_instances)


def f4_sampling(learner, X, n_instances):
    """Implementation for the F4 hardness measure."""
    return __generic_hardness_sampling(
        learner, X, 'f4', n_instances)
