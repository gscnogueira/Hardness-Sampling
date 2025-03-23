import warnings
from functools import wraps
import itertools

import numpy as np
import pandas as pd
from modAL.utils.selection import multi_argmax
from modAL import ActiveLearner
from scipy.sparse.csgraph import minimum_spanning_tree
from pyhard import get_seed
from pyhard.measures import ClassificationMeasures, minmax, maxmin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import DistanceMetric

from .random import random_sampling


class HardnessError(Exception):
    """Exceção para ser lançada quando não for possível realizar o calculo de uma HM."""
    def __init__(self, mensagem):
        super().__init__(mensagem)

def hardness_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            scores =  func(*args, **kwargs)
            # print("Deu bom")
            return multi_argmax(scores, kwargs['n_instances'])
        except HardnessError as e:

            warnings.warn(f'An error occurred while calculating {func.__name__}: '
                          f'({type(e).__name__}) "{e}" .'
                          ' Falling back to random sampling.', UserWarning)

            return random_sampling(*args, **kwargs)

    return wrapper


# TREE BASED
def __fit_tree(X, y, prunning=False):

    if prunning:

        # Número de splits do GridSearchCV é 5
        if len(X) <  5:
            raise HardnessError("Número de instâncias insuficiente para calculo do ccp_alpha.")

        parameters = {'ccp_alpha': np.linspace(0.001, 0.1, num=100)}
        dtc = DecisionTreeClassifier(criterion='gini', random_state=get_seed())
        clf = GridSearchCV(dtc, parameters, n_jobs=-1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            clf.fit(X, y)

        ccp_alpha = clf.best_params_['ccp_alpha']
        dtc = DecisionTreeClassifier(criterion='gini', ccp_alpha=ccp_alpha, random_state=get_seed())


    else:
        dtc = DecisionTreeClassifier(min_samples_split=2, criterion='gini', random_state=get_seed())

    dtc.fit(X, y)

    return dtc


@hardness_method
def disjunct_size_sampling(learner, X,n_instances=1):
    """Implementation for the DS hardness measure."""

    y = learner.predict(X)
    dtc = __fit_tree(X, y)
    data = pd.DataFrame(X).assign(y=y)

    data['leaf_id'] = dtc.apply(X)
    df_count = data.groupby('leaf_id').count().iloc[:, 0].to_frame('count').subtract(1)
    data = data.join(df_count, on='leaf_id')
    DS = data['count'].divide(data['count'].max())

    scores =  1 - DS.values

    return scores


@hardness_method
def disjunct_class_percentage_sampling(learner, X, n_instances):
    """Implementation for the DCP hardness measure."""

    y = learner.predict(X)
    dtc_pruned = __fit_tree(X, y, prunning=True)
    data = pd.DataFrame(X).assign(y=y)
    target_col = data.columns[-1]

    data['leaf_id'] = dtc_pruned.apply(X)
    dcp = []
    for index, row in data.iterrows():
        df_leaf = data[data['leaf_id'] == row['leaf_id']]
        dcp.append(len(df_leaf[df_leaf[target_col] == row[target_col]]) / len(df_leaf))

    return 1 - np.array(dcp)


@hardness_method
def tree_depth_pruned_sampling(learner, X, n_instances):
    """Implementation for the TD_P hardness measure."""

    y = learner.predict(X)

    dtc_pruned = __fit_tree(X, y, prunning=True)
    dtc_depth = dtc_pruned.get_depth()

    if dtc_depth == 0:
        raise HardnessError("Tree depth must greater than 0.")
    

    scores = np.apply_along_axis(lambda x: dtc_pruned.decision_path([x]).sum() -1,
                                 1, X) / dtc_depth

    return scores


@hardness_method
def tree_depth_unpruned_sampling(learner, X, n_instances):
    """Implementation for the TD_U hardness measure."""

    y = learner.predict(X)
    dtc = __fit_tree(X, y)

    dtc_depth = dtc.get_depth()

    if dtc_depth == 0:
        raise HardnessError("Tree depth must greater than 0.")
    
    scores = np.apply_along_axis(lambda x: dtc.decision_path([x]).sum() -1,
                                 1, X) / dtc_depth

    return scores

# LIKELIHOOD-BASED
def __get_calibrated_nb(X, y):
    n_c = len(np.unique(y))
    priors = np.ones((n_c,)) / n_c
    
    nb = GaussianNB(priors=priors)
    calibrated_nb = CalibratedClassifierCV(
        estimator=nb,
        method='sigmoid',
        cv=3,
        ensemble=False,
        n_jobs=-1)

    try:
        calibrated_nb.fit(X, y)
    except ValueError:
        # TODO: Precisa calibrar?
        raise HardnessError("Naive Bayes classifier could not be calibrated.")

    return calibrated_nb


@hardness_method
def class_likelihood_sampling(learner, X, n_instances):
    """Implementation for the CL hardness measure."""
    
    y = learner.predict(X)
    calibrated_nb = __get_calibrated_nb(X, y)

    proba = calibrated_nb.predict_proba(X)

    y = y.reshape(-1, 1)
    classes = calibrated_nb.classes_.reshape((1, -1))

    CL = proba[np.equal(y, classes)]

    return 1 - CL


@hardness_method
def class_likeliood_diff_sampling(learner, X, n_instances):
    """Implementation for the CLD hardness measure."""

    y = learner.predict(X)
    if len(np.unique(y)) < 2:
        raise HardnessError("Não há um número de classes suficiente para calcular CLD.")

    calibrated_nb = __get_calibrated_nb(X, y)
    N = len(y)

    proba = calibrated_nb.predict_proba(X)

    y = y.reshape(-1, 1)
    classes = calibrated_nb.classes_.reshape((1, -1))
    CL = proba[np.equal(y, classes)]
    CLD = CL - np.max(proba[~np.equal(y, classes)].reshape((N, -1)), axis=1)

    return (1 - CLD) / 2

# CLASS BALANCE
# -------------------------------------------------
@hardness_method
def minority_value_sampling(learner, X, n_instances):
    """Implementation for the MV hardness measure."""


    y = learner.predict(X)
    data = pd.DataFrame(X).assign(y=y)
    target_col = data.columns[-1]

    mv_class = data.groupby(target_col).count().iloc[:, 0]
    mv_class = mv_class.divide(mv_class.max())

    labels = pd.Series(y)
    return labels.apply(lambda c: 1 - mv_class[c]).values

@hardness_method
def class_balance_sampling(learner, X, n_instances):
    """Implementation for the CB hardness measure."""

    y = learner.predict(X)
    data = pd.DataFrame(X).assign(y=y)
    target_col = data.columns[-1]
    N = len(y)

    cb_class = data.groupby(target_col).count().iloc[:, 0]
    cb_class = cb_class.divide(N) - 1 / len(np.unique(y))

    labels = pd.Series(y)
    n_classes =len(np.unique(y))
    return (1 - labels.apply(lambda c: cb_class[c]).values) * n_classes / (n_classes + 1)


# NEIGHBORHOOD BASED
# -----------------------------------------------------
def gower_distance(X: pd.DataFrame) -> np.ndarray:
    """
    Calculates the Gower's distance (similarity). The samples may contain both categorical and numerical features. It
    returns a value between 0 (identical) and 1 (maximally dissimilar).

    Args:
        X (pd.DataFrame): the feature matrix

    Returns:
        float: coefficient measuring the similarity between two samples.
    """
    N = len(X)
    n_feat = X.shape[1]
    cumsum_dist = np.zeros((N, N))

    for i in range(n_feat):
        feature = X.iloc[:, [i]]
        if feature.dtypes.iloc[0] == object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature)
            feature_dist /= max(np.ptp(feature.values), 1e-8)

        cumsum_dist += feature_dist * 1 / n_feat

    return cumsum_dist

@hardness_method
def k_disagreeing_neighbors_sampling(learner: ActiveLearner, X: np.ndarray,
                                     n_instances=1, k=10):
    """Implementation for the kDN hardness measure."""

    y = learner.predict(X)

    dist_matrix_gower = gower_distance(pd.DataFrame(X))
    delta = np.diag(-np.ones(dist_matrix_gower.shape[0]))
    indices_gower = np.argsort(dist_matrix_gower + delta, axis=1)
    distances_gower = np.sort(dist_matrix_gower, axis=1)

    indices = indices_gower[:, :k + 1]

    kDN = []
    for i in range(0, len(X)):
        v = y[indices[i]]
        kDN.append(np.sum(v[1:] != v[0]) / k)

    scores =  np.array(kDN)
    return scores

@hardness_method
def borderline_points_sampling(learner, X, n_instances):
    """Implementation for the N1 hardness measure."""

    y = learner.predict(X)

    dist_matrix = gower_distance(pd.DataFrame(X))
    Tcsr = minimum_spanning_tree(dist_matrix)
    mst = Tcsr.toarray()
    mst = np.where(mst > 0, mst, np.inf)

    N1 = np.zeros(y.shape)
    for i in range(len(y)):
        idx = np.argwhere(np.minimum(mst[i, :], mst[:, i]) < np.inf)
        N1[i] = np.sum(y[idx[:, 0]] != y[i]) / len(y[idx[:, 0]])

    return N1


@hardness_method
def intra_extra_ratio_sampling(learner, X, n_instances):
    """Implementation for the N2 hardness measure."""

    y = pd.Series(learner.predict(X))

    if len(y.unique()) < 2:
        raise HardnessError("Número de classes preditas é insuficiente para calcular N2I")
    if y.value_counts().min() < 2:
        raise HardnessError("Número de instâncias preditas por classe é insuficiente para calcular N2I")

    dist_matrix_gower = gower_distance(pd.DataFrame(X))
    delta = np.diag(-np.ones(dist_matrix_gower.shape[0]))
    indices_gower = np.argsort(dist_matrix_gower + delta, axis=1)
    distances_gower = np.sort(dist_matrix_gower, axis=1)

    indices = indices_gower
    distances = distances_gower

    N2 = np.zeros(y.values.shape)
    for i, label in y.items():
        nn = y.loc[indices[i, :]]
        intra = nn.eq(label)
        extra = nn.ne(label)
        assert np.all(np.diff(distances[i, intra]) >= 0)
        assert np.all(np.diff(distances[i, extra]) >= 0)
    N2[i] = distances[i, intra][1] / max(distances[i, extra][0], 1e-15)

    return 1 - 1 / (N2 + 1)


@hardness_method
def local_set_cardinality_sampling(learner, X, n_instances):
    """Implementation for the LSC hardness measure."""
    y = pd.Series(learner.predict(X))


    dist_matrix_gower = gower_distance(pd.DataFrame(X))
    delta = np.diag(-np.ones(dist_matrix_gower.shape[0]))
    indices_gower = np.argsort(dist_matrix_gower + delta, axis=1)
    distances_gower = np.sort(dist_matrix_gower, axis=1)

    indices = indices_gower

    LSC = np.zeros(y.values.shape)
    n_class = y.value_counts()
    for i, label in y.items():
        nn = y.loc[indices[i, :]].values
        LSC[i] = np.argmax(nn != label) / n_class[label]

    return 1 - LSC


@hardness_method
def ls_radius_sampling(learner, X, n_instances):
    """Implementation for the LSR hardness measure."""
    y = pd.Series(learner.predict(X))


    dist_matrix_gower = gower_distance(pd.DataFrame(X))
    delta = np.diag(-np.ones(dist_matrix_gower.shape[0]))
    indices_gower = np.argsort(dist_matrix_gower + delta, axis=1)
    distances_gower = np.sort(dist_matrix_gower, axis=1)

    indices = indices_gower
    distances = distances_gower

    LSR = np.zeros(y.values.shape)
    for i, label in y.items():
        nn = y.loc[indices[i, :]].values
        aux = (nn == label)[::-1]
        ind_nn_max = len(aux) - np.argmax(aux) - 1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            LSR[i] = min(1.0, distances[i, np.argmax(nn != label)] / distances[i, ind_nn_max])

    scores = 1 - LSR

    return scores


@hardness_method
def harmfulness_sampling(learner, X, n_instances):
    """Implementation for the Harmfulness hardness measure."""
    N = len(X)
    y = learner.predict(X)

    dist_matrix_gower = gower_distance(pd.DataFrame(X))
    delta = np.diag(-np.ones(dist_matrix_gower.shape[0]))
    indices_gower = np.argsort(dist_matrix_gower + delta, axis=1)

    indices = indices_gower
        
    ne_pos = np.argmax(y[indices] != y[:, None], axis=1)
    ne = indices[np.arange(len(indices)), ne_pos]
    n_class =pd.Series(y).value_counts()

    H = np.sum(np.reshape(np.arange(N), (N, -1)) == ne, axis=1)

    return H / (N - n_class[y].values)


@hardness_method
def usefulness_sampling(learner, X, n_instances):
    """Implementation for the Usefulness hardness measure."""
    N = len(X)
    y = learner.predict(X)

    dist_matrix_gower = gower_distance(pd.DataFrame(X))
    delta = np.diag(-np.ones(dist_matrix_gower.shape[0]))
    indices_gower = np.argsort(dist_matrix_gower + delta, axis=1)

    indices = indices_gower

    ne_pos = np.argmax(y[indices] != y[:, None], axis=1)
    n_class = pd.Series(y).value_counts()

    u = np.zeros(y.shape)
    for i in range(N):
        ls = indices[i, 1:ne_pos[i]]
        u[ls] += 1

    scores = 1 - (u / n_class[y].values)

    return scores


@hardness_method
def f1_sampling(learner, X, n_instances):
    """Implementation for the F1 hardness measure."""

    y = pd.Series(learner.predict(X))
    df = pd.DataFrame(X).assign(y=y)
    X = pd.DataFrame(X)
    target_col = df.columns[-1]

    features = X.columns.to_list()
    n_features = len(features)
    classes = y.unique().tolist()

    F1 = pd.Series(0, index=df.index)
    for p in itertools.combinations(classes, 2):
        sub_df = df[(y == p[0]) | (y == p[1])]
        indicator = pd.Series(0, index=sub_df.index)
        for f in features:
            m1 = maxmin(sub_df[f].values, sub_df[target_col].values)
            m2 = minmax(sub_df[f].values, sub_df[target_col].values)
            indicator += sub_df[f].between(m1, m2, inclusive='both') * 1
        F1 = F1.add(indicator.divide(n_features), fill_value=0)

    F1 = F1.divide(len(classes) - 1)
    return F1.values

def do_t(learner, X):

    y = pd.Series(learner.predict(X))
    df = pd.DataFrame(X).assign(y=y)
    X = pd.DataFrame(X)
    target_col = df.columns[-1]

    features = X.columns.to_list()
    classes = y.unique().tolist()
    
    dot = pd.DataFrame(0, columns=X.columns, index=X.index)
    for p in itertools.combinations(classes, 2):
        sub_df = df[(y == p[0]) | (y == p[1])]
        for f in features:
            m1 = maxmin(sub_df[f].values, sub_df[target_col].values)
            m2 = minmax(sub_df[f].values, sub_df[target_col].values)
            do = (m2 - sub_df[f]) / (m2 - m1)
            dot[f] = dot[f].add(1 / (1 + abs(0.5 - do)), fill_value=0)
            
    k = len(classes)
    dot = dot.div(k * (k - 1) / 2)
    return dot


@hardness_method
def f2_sampling(learner, X, n_instances):
    """Implementation for the F2 hardness measure."""

    dot = do_t(learner, X)
    return dot.min(axis=1).values

@hardness_method
def f3_sampling(learner, X, n_instances):
    """Implementation for the F3 hardness measure."""

    dot = do_t(learner, X)
    return dot.mean(axis=1).values

@hardness_method
def f4_sampling(learner, X, n_instances):
    """Implementation for the F4 hardness measure."""

    dot = do_t(learner, X)
    return dot.max(axis=1).values
