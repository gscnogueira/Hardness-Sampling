import numpy as np


def random_sampling(learner, X, n_instances=1):

    query_idx = np.random.choice(range(len(X)), size=n_instances,
                                 replace=False)

    return query_idx, X[query_idx]
