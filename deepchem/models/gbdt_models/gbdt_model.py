"""
Gradient Boosting Decision Tree wrapper interface
"""

import os
import logging
import tempfile
import warnings
from typing import Callable, Optional, Union, List, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from deepchem.data import Dataset
from deepchem.models.sklearn_models import SklearnModel

logger = logging.getLogger(__name__)


class GBDTModel(SklearnModel):
    """Wrapper class that wraps GBDT models as DeepChem models.

    This class supports LightGBM/XGBoost models.
    """

    def __init__(self,
                 model: BaseEstimator,
                 model_dir: Optional[str] = None,
                 early_stopping_rounds: int = 50,
                 eval_metric: Optional[Union[str, Callable]] = None,
                 **kwargs):
        """
        Parameters
        ----------
        model: BaseEstimator
            The model instance of scikit-learn wrapper LightGBM/XGBoost models.
        model_dir: str, optional (default None)
            Path to directory where model will be stored.
        early_stopping_rounds: int, optional (default 50)
            Activates early stopping. Validation metric needs to improve at least once
            in every early_stopping_rounds round(s) to continue training.
        eval_metric: Union[str, Callable]
            If string, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see official note for more details.
        """

        try:
            import xgboost
            import lightgbm
        except ModuleNotFoundError:
            raise ImportError(
                'This function requires XGBoost and LightGBM modules '
                'to be installed.')

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            model_dir = tempfile.mkdtemp()
        self.model_dir = model_dir
        self.model = model
        self.model_class = model.__class__
        self.early_stopping_rounds = early_stopping_rounds
        self.model_type = self._check_model_type()
        self.callbacks: List[Union[Any, Any]]

        if self.early_stopping_rounds <= 0:
            raise ValueError("Early Stopping Rounds cannot be less than 1.")

        if self.model.__class__.__name__.startswith('XGB'):
            self.callbacks = [
                xgboost.callback.EarlyStopping(  # type: ignore
                    rounds=self.early_stopping_rounds)
            ]
            self.model.callbacks = self.callbacks
        elif self.model.__class__.__name__.startswith('LGBM'):
            self.callbacks = [
                lightgbm.early_stopping(  # type: ignore
                    stopping_rounds=self.early_stopping_rounds),
            ]

        if eval_metric is None:
            if self.model_type == "classification":
                self.eval_metric: Optional[Union[str, Callable]] = "auc"
            elif self.model_type == "regression":
                self.eval_metric = "mae"
            else:
                self.eval_metric = eval_metric
        else:
            self.eval_metric = eval_metric

        if self.model.__class__.__name__.startswith('XGB'):
            self.model.eval_metric = self.eval_metric

    def _check_model_type(self) -> str:
        class_name = self.model.__class__.__name__
        if class_name.endswith("Classifier"):
            return "classification"
        elif class_name.endswith("Regressor"):
            return "regression"
        elif class_name == "NoneType":
            return "none"
        else:
            raise ValueError(
                "{} is not a supported model instance.".format(class_name))

    def fit(self, dataset: Dataset):
        """Fits GDBT model with all data.

        First, this function splits all data into train and valid data (8:2),
        and finds the best n_estimators. And then, we retrain all data using
        best n_estimators * 1.25.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        X = dataset.X
        y = np.squeeze(dataset.y)

        # GDBT doesn't support multi-output(task)
        if len(y.shape) != 1:
            raise ValueError("GDBT model doesn't support multi-output(task)")

        seed = self.model.random_state
        stratify = None
        if self.model_type == "classification":
            stratify = y

        # Find optimal n_estimators based on original learning_rate and early_stopping_rounds
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=seed,
                                                            stratify=stratify)

        if self.model.__class__.__name__.startswith('XGB'):
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
            )

        elif self.model.__class__.__name__.startswith('LGBM'):
            self.model.fit(
                X_train,
                y_train,
                callbacks=self.callbacks,
                eval_metric=self.eval_metric,
                eval_set=[(X_test, y_test)],
            )

        # retrain model to whole data using best n_estimators * 1.25 [ XGBoost requires an evalset if early stopping setup is done.]
        if self.model.__class__.__name__.startswith('XGB'):
            estimated_best_round = np.round(
                (self.model.best_iteration + 1) * 1.25)
        else:
            estimated_best_round = np.round(self.model.best_iteration_ * 1.25)
        self.model.n_estimators = np.int64(estimated_best_round)
        if self.model.__class__.__name__.startswith('XGB'):
            if self.early_stopping_rounds == 0:
                self.model.fit(X, y)
        if self.model.__class__.__name__.startswith('LGBM'):
            self.model.fit(X, y, eval_metric=self.eval_metric)

    def fit_with_eval(self, train_dataset: Dataset, valid_dataset: Dataset):
        """Fits GDBT model with valid data.

        Parameters
        ----------
        train_dataset: Dataset
            The `Dataset` to train this model on.
        valid_dataset: Dataset
            The `Dataset` to validate this model on.
        """
        X_train, X_valid = train_dataset.X, valid_dataset.X
        y_train, y_valid = np.squeeze(train_dataset.y), np.squeeze(
            valid_dataset.y)

        # GDBT doesn't support multi-output(task)
        if len(y_train.shape) != 1 or len(y_valid.shape) != 1:
            raise ValueError("GDBT model doesn't support multi-output(task)")

        if self.model.__class__.__name__.startswith('XGB'):
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
            )

        elif self.model.__class__.__name__.startswith('LGBM'):
            self.model.fit(
                X_train,
                y_train,
                callbacks=self.callbacks,
                eval_metric=self.eval_metric,
                eval_set=[(X_valid, y_valid)],
            )


#########################################
# Deprecation warnings for XGBoostModel
#########################################


class XGBoostModel(GBDTModel):

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "XGBoostModel is deprecated and has been renamed to GBDTModel.",
            FutureWarning,
        )
        super(XGBoostModel, self).__init__(*args, **kwargs)
