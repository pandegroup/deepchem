"""Evaluation metrics."""

import numpy as np
import scipy.stats
from sklearn.metrics import matthews_corrcoef  # noqa
from sklearn.metrics import recall_score  # noqa
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import r2_score  # noqa
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score  # noqa
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as ras  # noqa
from sklearn.metrics import accuracy_score  # noqa
from sklearn.metrics import balanced_accuracy_score  # noqa
from sklearn.metrics import top_k_accuracy_score  # noqa

# kappa_score is an alias for `sklearn.metrics.cohen_kappa_score`
kappa_score = cohen_kappa_score


def pearsonr(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Pearson correlation coefficient.

    Parameters
    ----------
    y: np.ndarray
        ground truth array
    y_pred: np.ndarray
        predicted array

    Returns
    -------
    float
        The Pearson correlation coefficient.
    """
    return scipy.stats.pearsonr(y, y_pred)[0]


def pearson_r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Pearson R^2 (square of Pearson correlation).

    Parameters
    ----------
    y: np.ndarray
        ground truth array
    y_pred: np.ndarray
        predicted array

    Returns
    -------
    float
        The Pearson-R^2 score.
    """
    return scipy.stats.pearsonr(y, y_pred)[0]**2


def jaccard_index(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Jaccard Index which is the Intersection Over Union metric
    which is commonly used in image segmentation tasks.

    DEPRECATED: WILL BE REMOVED IN A FUTURE VERSION OF DEEEPCHEM. USE `jaccard_score` instead.

    Parameters
    ----------
    y: np.ndarray
        ground truth array
    y_pred: np.ndarray
        predicted array

    Returns
    -------
    score: float
        The jaccard index. A number between 0 and 1.
    """
    return jaccard_score(y, y_pred)


def pixel_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """An error metric in case y, y_pred are images.

    Defined as 1 - the maximal F-score of pixel similarity, or squared
    Euclidean distance between the original and the result labels.

    Parameters
    ----------
    y: np.ndarray
        ground truth array
    y_pred: np.ndarray
        predicted array

    Returns
    -------
    score: float
        The pixel-error. A number between 0 and 1.
    """
    return 1 - f1_score(y, y_pred)


def prc_auc_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute area under precision-recall curve

    Parameters
    ----------
    y: np.ndarray
        A numpy array of shape `(N, n_classes)` or `(N,)` with true labels
    y_pred: np.ndarray
        Of shape `(N, n_classes)` with class probabilities.

    Returns
    -------
    float
        The area under the precision-recall curve. A number between 0 and 1.
    """
    precision, recall, _ = precision_recall_curve(y[:, 1], y_pred[:, 1])
    return auc(recall, precision)


def rms_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes RMS error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MAE."""
    return mean_absolute_error(y_true, y_pred)


def bedroc_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 20.0):
    """Compute BEDROC metric.

    BEDROC metric implemented according to Truchon and Bayley that modifies
    the ROC score by allowing for a factor of early recognition.
    Please confirm details from [1]_.

    Parameters
    ----------
    y_true: np.ndarray
        Binary class labels. 1 for positive class, 0 otherwise
    y_pred: np.ndarray
        Predicted labels
    alpha: float, default 20.0
        Early recognition parameter

    Returns
    -------
    float
        Value in [0, 1] that indicates the degree of early recognition

    Notes
    -----
    This function requires RDKit to be installed.

    References
    ----------
    .. [1] Truchon et al. "Evaluating virtual screening methods: good and bad metrics
      for the “early recognition” problem." Journal of chemical information and modeling
      47.2 (2007): 488-508.
    """
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    # validation
    assert len(y_true) == len(y_pred), 'Number of examples do not match'
    assert np.array_equal(np.unique(y_true).astype(int),
                          [0, 1]), ('Class labels must be binary: %s' %
                                    np.unique(y_true))

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    yt = yt.flatten()
    yp = yp[:, 1].flatten()  # Index 1 because one_hot predictions

    scores = list(zip(yt, yp))
    scores = sorted(scores, key=lambda pair: pair[1], reverse=True)

    return CalcBEDROC(scores, 0, alpha)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Concordance index.

    Statistical metric indicates the quality of the predicted ranking.
    Please confirm details from [1]_.

    Parameters
    ----------
    y_true: np.ndarray
        continous value
    y_pred: np.ndarray
        Predicted value

    Returns
    -------
    float
        score between [0,1]

    References
    ----------
    .. [1] Steck, Harald, et al. "On ranking in survival analysis:
      Bounds on the concordance index." Advances in neural information processing systems (2008): 1209-1216.
    """

    idx = np.argsort(y_true)
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    pairs = 0
    correct_pairs = 0.0

    for i in range(len(y_true)):
        true_a = y_true[i]
        pred_a = y_pred[i]

        for j in range(i + 1, len(y_true)):
            true_b = y_true[j]
            pred_b = y_pred[j]
            if true_a != true_b:
                pairs += 1
                if pred_a == pred_b:
                    correct_pairs += 0.5
                elif pred_a < pred_b:
                    correct_pairs += true_a < true_b
                else:
                    correct_pairs += true_a > true_b

    assert pairs > 0, 'No pairs for comparision'

    return correct_pairs / pairs


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes the Area under the Receiving Operating Characteristic Curve (ROC AUC) from the predicted scores.

    Parameters:
    ----------
        y_true (np.ndarray):
        Corresponds to the true labels or ground truth labels.
        The shape of the array can be [n_samples, n_classes] or [n_samples, ] depending on the type of the classifier.
        y_score (np.ndarray):
        Corresponds to an array of probability estimates which are the probability of the sample for a given class.
        The shape of the array can be [n_samples, n_classes] or [n_samples, ] depending on the type of the classifier.

    Returns:
    -------
        float:
        The ROC AUC score.
    """

    # validation
    assert len(y_true) == len(y_score), 'Number of examples do not match'
    if len(y_score.shape) == 2:
        for row in y_score:
            assert np.isclose(np.sum(row), 1.0), 'Probabilities do not sum to 1'
            assert np.all(row >= 0.0), 'Probabilities are negative'

    return ras(y_true, y_score, multi_class='ovr')
