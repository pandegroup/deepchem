"""
Contains class for gaussian process hyperparameter optimizations.
"""
import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from deepchem.data import Dataset
from deepchem.trans import Transformer
from deepchem.models import Model
from deepchem.metrics import Metric
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename

logger = logging.getLogger(__name__)


def compute_parameter_range(
    params_dict: Dict,
    search_range: Union[int, float,
                        Dict]) -> Dict[str, Tuple[str, List[float]]]:
    """Convenience Function to compute parameter search space.

    Parameters
    ----------
    params_dict: Dict
        Dictionary mapping strings to Ints/Floats. An explicit list of
        parameters is computed with `search_range`. The optimization range
        computed is specified in the documentation for `search_range`
        below.
    search_range: int/float/Dict (default 4)
        The `search_range` specifies the range of parameter values to
        search for. If `search_range` is an int/float, it is used as the
        global search range for parameters. This creates a search
        problem on the following space:

        optimization on [initial value / search_range,
                        initial value * search_range]

        If `search_range` is a dict, it must contain the same keys as
        for `params_dict`. In this case, `search_range` specifies a
        per-parameter search range. This is useful in case some
        parameters have a larger natural range than others. For a given
        hyperparameter `hp` this would create the following search
        range:

        optimization on hp on [initial value[hp] / search_range[hp],
                              initial value[hp] * search_range[hp]]

    Returns
    -------
    param_range: Dict
        Dictionary mapping hyperparameter names to tuples. Each tuple is
        of form `(value_type, value_range)` where `value_type` is a string
        that is either "int" or "cont" and `value_range` is a list of two
        elements of the form `[low, hi]`. This format is expected by
        pyGPGO which `GaussianProcessHyperparamOpt` uses to perform
        optimization.
    """
    # Range of optimization
    param_range = {}
    if isinstance(search_range, dict):
        if sorted(params_dict.keys()) != sorted(search_range.keys()):
            raise ValueError(
                "If search_range is provided as a dictionary, it must have the same keys as params_dict."
            )
    elif (not isinstance(search_range, int)) and (not isinstance(
            search_range, float)):
        raise ValueError("search_range must be a dict or int or float.")
    for hp, value in params_dict.items():
        if isinstance(search_range, dict):
            hp_search_range = search_range[hp]
        else:
            # We know from guard above that this is an int/float
            hp_search_range = search_range
        if isinstance(value, int):
            value_range = [value // hp_search_range, value * hp_search_range]
            param_range[hp] = ("int", value_range)
        elif isinstance(value, float):
            value_range = [value / hp_search_range, value * hp_search_range]
            param_range[hp] = ("cont", value_range)
    return param_range


class GaussianProcessHyperparamOpt(HyperparamOpt):
    """
    Gaussian Process Global Optimization(GPGO)

    This class uses Gaussian Process optimization to select
    hyperparameters. Underneath the hood it uses pyGPGO to optimize
    models. If you don't have pyGPGO installed, you won't be able to use
    this class.

    Note that `params_dict` has a different semantics than for
    `GridHyperparamOpt`. `param_dict[hp]` must be an int/float and is
    used as the center of a search range.

    Examples
    --------
    This example shows the type of constructor function expected.

    >>> import deepchem as dc
    >>> optimizer = dc.hyper.GaussianProcessHyperparamOpt(lambda **p: dc.models.GraphConvModel(n_tasks=1, **p))

    Here's a more sophisticated example that shows how to optimize only
    some parameters of a model. In this case, we have some parameters we
    want to optimize, and others which we don't. To handle this type of
    search, we create a `model_builder` which hard codes some arguments
    (in this case, `n_tasks` and `n_features` which are properties of a
    dataset and not hyperparameters to search over.)

    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor as RF
    >>> def model_builder(**model_params):
    ...   n_estimators = model_params['n_estimators']
    ...   min_samples_split = model_params['min_samples_split']
    ...   rf_model = RF(n_estimators=n_estimators, min_samples_split=min_samples_split)
    ...   rf_model = RF(n_estimators=n_estimators)
    ...   return dc.models.SklearnModel(rf_model)
    >>> optimizer = dc.hyper.GaussianProcessHyperparamOpt(model_builder)
    >>> params_dict = {"n_estimators":100, "min_samples_split":2}
    >>> train_dataset = dc.data.NumpyDataset(X=np.random.rand(50, 5),
    ...   y=np.random.rand(50, 1))
    >>> valid_dataset = dc.data.NumpyDataset(X=np.random.rand(20, 5),
    ...   y=np.random.rand(20, 1))
    >>> metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

    >> best_model, best_hyperparams, all_results =\
    optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset, metric, max_iter=2)
    >> type(best_hyperparams)
    <class 'dict'>

    Parameters
    ----------
    model_builder: constructor function.
        This parameter must be constructor function which returns an
        object which is an instance of `dc.models.Model`. This function
        must accept two arguments, `model_params` of type `dict` and
        `model_dir`, a string specifying a path to a model directory.
    max_iter: int, default 20
        number of optimization trials
    search_range: int/float/Dict (default 4)
        The `search_range` specifies the range of parameter values to
        search for. If `search_range` is an int/float, it is used as the
        global search range for parameters. This creates a search
        problem on the following space:

        optimization on [initial value / search_range,
                        initial value * search_range]

        If `search_range` is a dict, it must contain the same keys as
        for `params_dict`. In this case, `search_range` specifies a
        per-parameter search range. This is useful in case some
        parameters have a larger natural range than others. For a given
        hyperparameter `hp` this would create the following search
        range:

        optimization on hp on [initial value[hp] / search_range[hp],
                              initial value[hp] * search_range[hp]]

    Notes
    -----
    This class requires pyGPGO to be installed.
    """

    def __init__(self,
                 model_builder: Callable[..., Model],
                 max_iter: int = 20,
                 search_range: Union[int, float, Dict] = 4):
        super(GaussianProcessHyperparamOpt,
              self).__init__(model_builder=model_builder)
        self.max_iter = max_iter
        self.search_range = search_range

    def hyperparam_search(
            self,
            params_dict: Dict,
            train_dataset: Dataset,
            valid_dataset: Dataset,
            metric: Metric,
            output_transformers: List[Transformer] = [],
            nb_epoch: int = 10,
            use_max: bool = True,
            logfile: str = 'results.txt',
            logdir: Optional[str] = None,
            **kwargs) -> Tuple[Model, Dict[str, Any], Dict[str, Any]]:
        """Perform hyperparameter search using a gaussian process.

        Parameters
        ----------
        params_dict: Dict
            Maps hyperparameter names (strings) to possible parameter
            values. The semantics of this list are different than for
            `GridHyperparamOpt`. `params_dict[hp]` must map to an int/float,
            which is used as the center of a search with radius
            `search_range` since pyGPGO can only optimize numerical
            hyperparameters.
        train_dataset: Dataset
            dataset used for training
        valid_dataset: Dataset
            dataset used for validation(optimization on valid scores)
        metric: Metric
            metric used for evaluation
        output_transformers: list[Transformer]
            Transformers for evaluation. This argument is needed since
            `train_dataset` and `valid_dataset` may have been transformed
            for learning and need the transform to be inverted before
            the metric can be evaluated on a model.
        nb_epoch: int, (default 10)
            Specifies the number of training epochs during each iteration of optimization.
            Not used by all model types.
        use_max: bool, (default True)
            Specifies whether to maximize or minimize `metric`.
            maximization(True) or minimization(False)
        logdir: str, optional, (default None)
            The directory in which to store created models. If not set, will
            use a temporary directory.
        logfile: str, optional (default `results.txt`)
            Name of logfile to write results to. If specified, this is must
            be a valid file. If not specified, results of hyperparameter
            search will be written to `logdir/results.txt`.


        Returns
        -------
        Tuple[`best_model`, `best_hyperparams`, `all_scores`]
            `(best_model, best_hyperparams, all_scores)` where `best_model` is
            an instance of `dc.model.Model`, `best_hyperparams` is a
            dictionary of parameters, and `all_scores` is a dictionary mapping
            string representations of hyperparameter sets to validation
            scores.
        """
        # try:
        #     from pyGPGO.covfunc import matern32
        #     from pyGPGO.acquisition import Acquisition
        #     from pyGPGO.surrogates.GaussianProcess import GaussianProcess
        #     from pyGPGO.GPGO import GPGO
        # except ModuleNotFoundError:
        #     raise ImportError("This class requires pyGPGO to be installed.")

        from deepchem.hyper.pyGPGO_classes import matern32, Acquisition, GaussianProcess, GPGO

        # Specify logfile
        log_file = None
        if logfile:
            log_file = logfile
        elif logdir is not None:
            # Make logdir if it doesn't exist.
            if not os.path.exists(logdir):
                os.makedirs(logdir, exist_ok=True)
            log_file = os.path.join(logdir, "results.txt")

        # setup range
        param_range = compute_parameter_range(params_dict, self.search_range)
        param_keys = list(param_range.keys())

        # Stores all results
        all_results: Dict[Any, Any] = {}
        # Store all model references so we don't have to reload
        all_models = {}
        # Stores all model locations
        model_locations = {}

        def _optimize(nb_epoch, **placeholders):
            """Private Optimizing function

            Take in hyper parameter values and number of training epochs.
            Return valid set performances.

            Parameters
            ----------
            nb_epoch: int
                Number of epochs to train model being optimized during each iteration.
                Not used by all model types.
            placeholders: keyword arguments
                Should be various hyperparameters as specified in `param_keys` above.

            Returns:
            --------
            valid_scores: float
                valid set performances
            """
            hyper_parameters = {}
            for hp in param_keys:
                if param_range[hp][0] == "int":
                    # param values are always float in BO, so this line converts float to int
                    # see : https://github.com/josejimenezluna/pyGPGO/issues/10
                    hyper_parameters[hp] = int(placeholders[hp])
                else:
                    hyper_parameters[hp] = float(placeholders[hp])
            logger.info("Running hyperparameter set: %s" %
                        str(hyper_parameters))
            if log_file:
                with open(log_file, 'w+') as f:
                    # Record hyperparameters
                    f.write("Parameters: %s" % str(hyper_parameters))
                    f.write('\n')

            hp_str = _convert_hyperparam_dict_to_filename(hyper_parameters)
            if hp_str in all_results:
                # We have already evaluated the model for these hyperparameters.
                if use_max:
                    return all_results[hp_str]
                else:
                    return -all_results[hp_str]
            if logdir is not None:
                filename = "model%s" % hp_str
                model_dir = os.path.join(logdir, filename)
                logger.info("model_dir is %s" % model_dir)
                try:
                    os.makedirs(model_dir)
                except OSError:
                    if not os.path.isdir(model_dir):
                        logger.info(
                            "Error creating model_dir, using tempfile directory"
                        )
                        model_dir = tempfile.mkdtemp()
            else:
                model_dir = tempfile.mkdtemp()
            # Add it on to the information needed for the constructor
            hyper_parameters["model_dir"] = model_dir
            model = self.model_builder(**hyper_parameters)
            try:
                model.fit(train_dataset, nb_epoch=nb_epoch)
            # Not all models have nb_epoch
            except TypeError:
                model.fit(train_dataset)
            try:
                model.save()
            # Some models autosave
            except NotImplementedError:
                pass

            multitask_scores = model.evaluate(valid_dataset, [metric],
                                              output_transformers)
            score = multitask_scores[metric.name]

            if log_file:
                with open(log_file, 'a') as f:
                    # Record performances
                    f.write("Score: %s" % str(score))
                    f.write('\n')
            # Store all results
            all_results[hp_str] = score
            # Store reference to model
            all_models[hp_str] = model
            model_locations[hp_str] = model_dir
            # GPGO maximize performance by default
            # set performance to its negative value for minimization
            if use_max:
                return score
            else:
                return -score

        # Demarcating internal function for readability
        def optimizing_function(**placeholders):
            """Wrapper function

            Take in hyper parameter values.
            Calls a private optimize function (_optimize) with number of epochs.
            Returns valid set performances.

            Parameters
            ----------
            placeholders: keyword arguments
                Should be various hyperparameters as specified in `param_keys` above.

            Returns:
            --------
            valid_scores: float
                valid set performances
            """
            return _optimize(nb_epoch=nb_epoch, **placeholders)

        # execute GPGO
        cov = matern32()
        gp = GaussianProcess(cov)
        acq = Acquisition(mode='ExpectedImprovement')
        gpgo = GPGO(gp, acq, optimizing_function, param_range)
        logger.info("Max number of iteration: %i" % self.max_iter)
        gpgo.run(max_iter=self.max_iter)

        hp_opt, valid_performance_opt = gpgo.getResult()
        hyper_parameters = {}
        for hp in param_keys:
            if param_range[hp][0] == "int":
                hyper_parameters[hp] = int(hp_opt[hp])
            else:
                # FIXME: Incompatible types in assignment
                hyper_parameters[hp] = float(hp_opt[hp])  # type: ignore
        hp_str = _convert_hyperparam_dict_to_filename(hyper_parameters)

        # Let's fetch the model with the best parameters
        best_model = all_models[hp_str]

        # Compare best model to default hyperparameters
        if log_file:
            with open(log_file, 'a') as f:
                # Record hyperparameters
                f.write("params_dict:")
                f.write(str(params_dict))
                f.write('\n')

        # Return default hyperparameters
        return best_model, hyper_parameters, all_results
