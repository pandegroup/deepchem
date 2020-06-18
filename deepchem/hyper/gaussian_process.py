"""
Contains class for gaussian process hyperparameter optimizations.
"""
import logging
import numpy as np
import tempfile
import os
import deepchem
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.utils.evaluate import Evaluator
from deepchem.molnet.run_benchmark_models import benchmark_classification, benchmark_regression

logger = logging.getLogger(__name__)


class GaussianProcessHyperparamOpt(HyperparamOpt):
  """
  Gaussian Process Global Optimization(GPGO)

  This class uses Gaussian Process optimization to select
  hyperparameters. Note that this class can only optimize 20
  parameters at a time.

  TODO: This class is too tied up with the MoleculeNet benchmarking.
  This needs to be refactored out cleanly.
  """

  def hyperparam_search(
      self,
      params_dict,
      train_dataset,
      valid_dataset,
      transformers,
      metric,
      use_max=True,
      logdir=None,
      n_features=1024,
      n_tasks=1,
      max_iter=20,
      search_range=4,
      log_file='GPhypersearch.log'):
    """Perform hyperparameter search using a gaussian process.

    Parameters
    ----------
    params_dict: dict
      dict including parameters and their initial values
    train_dataset: `dc.data.Dataset`
      dataset used for training
    valid_dataset: `dc.data.Dataset`
      dataset used for validation(optimization on valid scores)
    transformers: list[dc.trans.Transformer]
      transformers for evaluation
    metric: `dc.metrics.Metric`
      metric used for evaluation
    use_max: bool, (default True)
      maximization(True) or minimization(False)
    logdir: str, optional
      The directory in which to store created models. If not set, will
      use a temporary directory.
    n_features: int, (default 1024)
      number of input features
    n_tasks: int, (default 1)
      number of tasks
    max_iter: int, (default 20)
      number of optimization trials
    search_range: int(float) (default 4)
      optimization on [initial values / search_range,
                       initial values * search_range]
      names of parameters that should not be optimized
    logfile: string
      name of log file, hyperparameters and results for each trial
      will be recorded

    Returns
    -------
    `(best_model, best_hyperparams, all_scores)` where `best_model` is
    an instance of `dc.model.Models`, `best_hyperparams` is a
    dictionary of parameters, and `all_scores` is a dictionary mapping
    string representations of hyperparameter sets to validation
    scores.
    """
    if len(params_dict) > 20:
      raise ValueError("This class can only search over 20 parameters in one invocation.")
    #hyper_parameters = params_dict
    #hp_list = list(hyper_parameters.keys())
    #hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]
    #assert set(hp_list_class) <= set([list, int, float])
    ## Float or int hyper parameters(ex. batch_size, learning_rate)
    #hp_list_single = [
    #    hp_list[i] for i in range(len(hp_list)) if not hp_list_class[i] is list
    #]
    ## List of float or int hyper parameters(ex. layer_sizes)
    #hp_list_multiple = [(hp_list[i], len(hyper_parameters[hp_list[i]]))
    #                    for i in range(len(hp_list))
    #                    if hp_list_class[i] is list]

    # Number of parameters
    n_param = len(hp_list_single)
    if len(hp_list_multiple) > 0:
      n_param = n_param + sum([hp[1] for hp in hp_list_multiple])
    ## Range of optimization
    #param_range = []
    #for hp in hp_list_single:
    #  if hyper_parameters[hp].__class__ is int:
    #    param_range.append((('int'), [
    #        hyper_parameters[hp] // search_range,
    #        hyper_parameters[hp] * search_range
    #    ]))
    #  else:
    #    param_range.append((('cont'), [
    #        hyper_parameters[hp] / search_range,
    #        hyper_parameters[hp] * search_range
    #    ]))
    #for hp in hp_list_multiple:
    #  if hyper_parameters[hp[0]][0].__class__ is int:
    #    param_range.extend([(('int'), [
    #        hyper_parameters[hp[0]][i] // search_range,
    #        hyper_parameters[hp[0]][i] * search_range
    #    ]) for i in range(hp[1])])
    #  else:
    #    param_range.extend([(('cont'), [
    #        hyper_parameters[hp[0]][i] / search_range,
    #        hyper_parameters[hp[0]][i] * search_range
    #    ]) for i in range(hp[1])])

    # Dummy names
    param_name = ['l' + format(i, '02d') for i in range(20)]
    param = dict(zip(param_name[:n_param], param_range))

    data_dir = deepchem.utils.get_data_dir()
    log_file = os.path.join(data_dir, log_file)

    def f(l00=0,
          l01=0,
          l02=0,
          l03=0,
          l04=0,
          l05=0,
          l06=0,
          l07=0,
          l08=0,
          l09=0,
          l10=0,
          l11=0,
          l12=0,
          l13=0,
          l14=0,
          l15=0,
          l16=0,
          l17=0,
          l18=0,
          l19=0):
      """ Optimizing function

      Take in hyper parameter values and return valid set performances

      Parameters
      ----------
      l00~l19: int or float
        placeholders for hyperparameters being optimized,
        hyper_parameters dict is rebuilt based on input values of placeholders

      Returns:
      --------
      valid_scores: float
        valid set performances
      """
      args = locals()
      # Input hyper parameters
      i = 0
      for hp in hp_list_single:
        hyper_parameters[hp] = float(args[param_name[i]])
        if param_range[i][0] == 'int':
          hyper_parameters[hp] = int(hyper_parameters[hp])
        i = i + 1
      for hp in hp_list_multiple:
        hyper_parameters[hp[0]] = [
            float(args[param_name[j]]) for j in range(i, i + hp[1])
        ]
        if param_range[i][0] == 'int':
          hyper_parameters[hp[0]] = list(map(int, hyper_parameters[hp[0]]))
        i = i + hp[1]

      logger.info(hyper_parameters)
      # Run benchmark
      with open(log_file, 'a') as f:
        # Record hyperparameters
        f.write(str(hyper_parameters))
        f.write('\n')
      if isinstance(self.model_class, str):
        try:
          train_scores, valid_scores, _ = benchmark_classification(
              train_dataset,
              valid_dataset,
              valid_dataset, ['task_placeholder'] * n_tasks,
              transformers,
              n_features,
              metric,
              self.model_class,
              hyper_parameters=hyper_parameters)
        except AssertionError:
          train_scores, valid_scores, _ = benchmark_regression(
              train_dataset,
              valid_dataset,
              valid_dataset, ['task_placeholder'] * n_tasks,
              transformers,
              n_features,
              metric,
              self.model_class,
              hyper_parameters=hyper_parameters)
        score = valid_scores[self.model_class][metric[0].name]
      else:
        model_dir = tempfile.mkdtemp()
        model = self.model_class(hyper_parameters, model_dir)
        model.fit(train_dataset, **hyper_parameters)
        model.save()
        evaluator = Evaluator(model, valid_dataset, transformers)
        multitask_scores = evaluator.compute_model_performance([metric])
        score = multitask_scores[metric.name]

      with open(log_file, 'a') as f:
        # Record performances
        f.write(str(score))
        f.write('\n')
      # GPGO maximize performance by default, set performance to its negative value for minimization
      if use_max:
        return score
      else:
        return -score

    import pyGPGO
    from pyGPGO.covfunc import matern32
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    from pyGPGO.GPGO import GPGO
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, f, param)
    logger.info("Max number of iteration: %i" % max_iter)
    gpgo.run(max_iter=max_iter)

    hp_opt, valid_performance_opt = gpgo.getResult()
    # Readout best hyper parameters
    i = 0
    for hp in hp_list_single:
      hyper_parameters[hp] = float(hp_opt[param_name[i]])
      if param_range[i][0] == 'int':
        hyper_parameters[hp] = int(hyper_parameters[hp])
      i = i + 1
    for hp in hp_list_multiple:
      hyper_parameters[hp[0]] = [
          float(hp_opt[param_name[j]]) for j in range(i, i + hp[1])
      ]
      if param_range[i][0] == 'int':
        hyper_parameters[hp[0]] = list(map(int, hyper_parameters[hp[0]]))
      i = i + hp[1]

    # Compare best model to default hyperparameters
    with open(log_file, 'a') as f:
      # Record hyperparameters
      f.write(str(params_dict))
      f.write('\n')
    if isinstance(self.model_class, str):
      try:
        train_scores, valid_scores, _ = benchmark_classification(
            train_dataset,
            valid_dataset,
            valid_dataset, ['task_placeholder'] * n_tasks,
            transformers,
            n_features,
            metric,
            self.model_class,
            hyper_parameters=params_dict)
      except AssertionError:
        train_scores, valid_scores, _ = benchmark_regression(
            train_dataset,
            valid_dataset,
            valid_dataset, ['task_placeholder'] * n_tasks,
            transformers,
            n_features,
            metric,
            self.model_class,
            hyper_parameters=params_dict)
      score = valid_scores[self.model_class][metric[0].name]
      with open(log_file, 'a') as f:
        # Record performances
        f.write(str(score))
        f.write('\n')
      if not use_max:
        score = -score
      if score > valid_performance_opt:
        # Optimized model is better, return hyperparameters
        return params_dict, score

    # Return default hyperparameters
    return hyper_parameters, valid_performance_opt
