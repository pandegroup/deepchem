import numpy as np
from scipy.linalg import cholesky, solve
from scipy.stats import norm, t
# from scipy.special import gamma, kv
from scipy.spatial.distance import cdist
from collections import OrderedDict
from scipy.optimize import minimize
from joblib import Parallel, delayed


class GPGO:

    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=1):
        """
        Bayesian Optimization class.
        Parameters
        ----------
        Surrogate: Surrogate model instance
            Gaussian Process surrogate model instance.
        Acquisition: Acquisition instance
            Acquisition instance.
        f: fun
            Function to maximize over parameters specified by `parameter_dict`.
        parameter_dict: dict
            Dictionary specifying parameter, their type and bounds.
        n_jobs: int. Default 1
            Parallel threads to use during acquisition optimization.
        Attributes
        ----------
        parameter_key: list
            Parameters to consider in optimization
        parameter_type: list
            Parameter types.
        parameter_range: list
            Parameter bounds during optimization
        history: list
            Target values evaluated along the procedure.
        """
        self.GP = surrogate
        self.A = acquisition
        self.f = f
        self.parameters = parameter_dict
        self.n_jobs = n_jobs

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []
        self.logger = EventLogger(self)

    def _sampleParam(self):
        """
        Randomly samples parameters over bounds.
        Returns
        -------
        dict:
            A random sample of specified parameters.
        """
        d = OrderedDict()
        for index, param in enumerate(self.parameter_key):
            if self.parameter_type[index] == 'int':
                d[param] = np.random.randint(self.parameter_range[index][0],
                                             self.parameter_range[index][1])
            elif self.parameter_type[index] == 'cont':
                d[param] = np.random.uniform(self.parameter_range[index][0],
                                             self.parameter_range[index][1])
            else:
                raise ValueError('Unsupported variable type.')
        return d

    def _firstRun(self, n_eval=3):
        """
        Performs initial evaluations before fitting GP.
        Parameters
        ----------
        n_eval: int
            Number of initial evaluations to perform. Default is 3.
        """
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)
        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)

    def _acqWrapper(self, xnew):
        """
        Evaluates the acquisition function on a point.
        Parameters
        ----------
        xnew: np.ndarray, shape=((len(self.parameter_key),))
            Point to evaluate the acquisition function on.
        Returns
        -------
        float
            Acquisition function value for `xnew`.
        """
        new_mean, new_var = self.GP.predict(xnew, return_std=True)
        new_std = np.sqrt(new_var + 1e-6)
        return -self.A.eval(self.tau, new_mean, new_std)

    def _optimizeAcq(self, method='L-BFGS-B', n_start=100):
        """
        Optimizes the acquisition function using a multistart approach.
        Parameters
        ----------
        method: str. Default 'L-BFGS-B'.
            Any `scipy.optimize` method that admits bounds and gradients. Default is 'L-BFGS-B'.
        n_start: int.
            Number of starting points for the optimization procedure. Default is 100.
        """
        start_points_dict = [self._sampleParam() for i in range(n_start)]
        start_points_arr = np.array(
            [list(s.values()) for s in start_points_dict])
        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))
        if self.n_jobs == 1:
            for index, start_point in enumerate(start_points_arr):
                res = minimize(self._acqWrapper,
                               x0=start_point,
                               method=method,
                               bounds=self.parameter_range)
                x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]
        else:
            opt = Parallel(n_jobs=self.n_jobs)(
                delayed(minimize)(self._acqWrapper,
                                  x0=start_point,
                                  method=method,
                                  bounds=self.parameter_range)
                for start_point in start_points_arr)
            x_best = np.array([res.x for res in opt])
            f_best = np.array([np.atleast_1d(res.fun)[0] for res in opt])

        self.best = x_best[np.argmin(f_best)]

    def updateGP(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        kw = {param: self.best[i] for i, param in enumerate(self.parameter_key)}
        f_new = self.f(**kw)
        self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
        self.tau = np.max(self.GP.y)
        self.history.append(self.tau)

    def getResult(self):
        """
        Prints best result in the Bayesian Optimization procedure.
        Returns
        -------
        OrderedDict
            Point yielding best evaluation in the procedure.
        float
            Best function evaluation.
        """
        argtau = np.argmax(self.GP.y)
        opt_x = self.GP.X[argtau]
        res_d = OrderedDict()
        for i, (key, param_type) in enumerate(
                zip(self.parameter_key, self.parameter_type)):
            if param_type == 'int':
                res_d[key] = int(round(opt_x[i]))
            else:
                res_d[key] = opt_x[i]
        return res_d, self.tau

    def run(self, max_iter=10, init_evals=3, resume=False):
        """
        Runs the Bayesian Optimization procedure.
        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        init_evals: int
            Initial function evaluations before fitting a GP. Default is 3.
        resume: bool
            Whether to resume the optimization procedure from the last evaluation. Default is `False`.
        """
        if not resume:
            self.init_evals = init_evals
            self._firstRun(self.init_evals)
            self.logger._printInit(self)
        for iteration in range(max_iter):
            self._optimizeAcq()
            self.updateGP()
            self.logger._printCurrent(self)


class GaussianProcess:

    def __init__(self, covfunc, optimize=False, usegrads=False, mprior=0):
        """
        Gaussian Process regressor class. Based on Rasmussen & Williams [1]_ algorithm 2.1.
        Parameters
        ----------
        covfunc: instance from a class of covfunc module
            Covariance function. An instance from a class in the `covfunc` module.
        optimize: bool:
            Whether to perform covariance function hyperparameter optimization.
        usegrads: bool
            Whether to use gradient information on hyperparameter optimization. Only used
            if `optimize=True`.
        Attributes
        ----------
        covfunc: object
            Internal covariance function.
        optimize: bool
            User chosen optimization configuration.
        usegrads: bool
            Gradient behavior
        mprior: float
            Explicit value for the mean function of the prior Gaussian Process.
        Notes
        -----
        [1] Rasmussen, C. E., & Williams, C. K. I. (2004). Gaussian processes for machine learning.
        International journal of neural systems (Vol. 14). http://doi.org/10.1142/S0129065704001899
        """
        self.covfunc = covfunc
        self.optimize = optimize
        self.usegrads = usegrads
        self.mprior = mprior

    def getcovparams(self):
        """
        Returns current covariance function hyperparameters
        Returns
        -------
        dict
            Dictionary containing covariance function hyperparameters
        """
        d = {}
        for param in self.covfunc.parameters:
            d[param] = self.covfunc.__dict__[param]
        return d

    def fit(self, X, y):
        """
        Fits a Gaussian Process regressor
        Parameters
        ----------
        X: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to X.
        """
        self.X = X
        self.y = y
        self.nsamples = self.X.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads:
                grads = self._grad
            self.optHyp(param_key=self.covfunc.parameters,
                        param_bounds=self.covfunc.bounds,
                        grads=grads)

        self.K = self.covfunc.K(self.X, self.X)
        self.L = cholesky(self.K).T
        self.alpha = solve(self.L.T, solve(self.L, y - self.mprior))
        self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(
            np.log(np.diag(self.L))) - self.nsamples / 2 * np.log(2 * np.pi)

    def param_grad(self, k_param):
        """
        Returns gradient over hyperparameters. It is recommended to use `self._grad` instead.
        Parameters
        ----------
        k_param: dict
            Dictionary with keys being hyperparameters and values their queried values.
        Returns
        -------
        np.ndarray
            Gradient corresponding to each hyperparameters. Order given by `k_param.keys()`
        """
        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param, bounds=self.covfunc.bounds)
        K = covfunc.K(self.X, self.X)
        L = cholesky(K).T
        alpha = solve(L.T, solve(L, self.y))
        inner = np.dot(np.atleast_2d(alpha).T,
                       np.atleast_2d(alpha)) - np.linalg.inv(K)
        grads = []
        for param in k_param_key:
            gradK = covfunc.gradK(self.X, self.X, param=param)
            gradK = .5 * np.trace(np.dot(inner, gradK))
            grads.append(gradK)
        return np.array(grads)

    def _lmlik(self, param_vector, param_key):
        """
        Returns marginal negative log-likelihood for given covariance hyperparameters.
        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.
        Returns
        -------
        float
            Negative log-marginal likelihood for chosen hyperparameters.
        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        self.covfunc = self.covfunc.__class__(**k_param,
                                              bounds=self.covfunc.bounds)

        # This fixes recursion
        original_opt = self.optimize
        original_grad = self.usegrads
        self.optimize = False
        self.usegrads = False

        self.fit(self.X, self.y)

        self.optimize = original_opt
        self.usegrads = original_grad
        return (-self.logp)

    def _grad(self, param_vector, param_key):
        """
        Returns gradient for each hyperparameter, evaluated at a given point.
        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.
        Returns
        -------
        np.ndarray
            Gradient for each evaluated hyperparameter.
        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return -self.param_grad(k_param)

    def optHyp(self, param_key, param_bounds, grads=None, n_trials=5):
        """
        Optimizes the negative marginal log-likelihood for given hyperparameters and bounds.
        This is an empirical Bayes approach (or Type II maximum-likelihood).
        Parameters
        ----------
        param_key: list
            List of hyperparameters to optimize.
        param_bounds: list
            List containing tuples defining bounds for each hyperparameter to optimize over.
        """
        xs = [[1, 1, 1]]
        fs = [self._lmlik(xs[0], param_key)]
        for trial in range(n_trials):
            x0 = []
            for param, bound in zip(param_key, param_bounds):
                x0.append(np.random.uniform(bound[0], bound[1], 1)[0])
            if grads is None:
                res = minimize(self._lmlik,
                               x0=x0,
                               args=(param_key),
                               method='L-BFGS-B',
                               bounds=param_bounds)
            else:
                res = minimize(self._lmlik,
                               x0=x0,
                               args=(param_key),
                               method='L-BFGS-B',
                               bounds=param_bounds,
                               jac=grads)
            xs.append(res.x)
            fs.append(res.fun)

        argmin = np.argmin(fs)
        opt_param = xs[argmin]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param,
                                              bounds=self.covfunc.bounds)

    def predict(self, Xstar, return_std=False):
        """
        Returns mean and covariances for the posterior Gaussian Process.
        Parameters
        ----------
        Xstar: np.ndarray, shape=((nsamples, nfeatures))
            Testing instances to predict.
        return_std: bool
            Whether to return the standard deviation of the posterior process. Otherwise,
            it returns the whole covariance matrix of the posterior process.
        Returns
        -------
        np.ndarray
            Mean of the posterior process for testing instances.
        np.ndarray
            Covariance of the posterior process for testing instances.
        """
        Xstar = np.atleast_2d(Xstar)
        kstar = self.covfunc.K(self.X, Xstar).T
        fmean = self.mprior + np.dot(kstar, self.alpha)
        v = solve(self.L, kstar.T)
        fcov = self.covfunc.K(Xstar, Xstar) - np.dot(v.T, v)
        if return_std:
            fcov = np.diag(fcov)
        return fmean, fcov

    def update(self, xnew, ynew):
        """
        Updates the internal model with `xnew` and `ynew` instances.
        Parameters
        ----------
        xnew: np.ndarray, shape=((m, nfeatures))
            New training instances to update the model with.
        ynew: np.ndarray, shape=((m,))
            New training targets to update the model with.
        """
        y = np.concatenate((self.y, ynew), axis=0)
        X = np.concatenate((self.X, xnew), axis=0)
        self.fit(X, y)


default_bounds = {
    'l': [1e-4, 1],
    'sigmaf': [1e-4, 2],
    'sigman': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4],
    'period': [1e-3, 10]
}


def l2norm_(X, Xstar):
    """
    Wrapper function to compute the L2 norm
    Parameters
    ----------
    X: np.ndarray, shape=((n, nfeatures))
        Instances.
    Xstar: np.ndarray, shape=((m, nfeatures))
        Instances
    Returns
    -------
    np.ndarray
        Pairwise euclidian distance between row pairs of `X` and `Xstar`.
    """
    return cdist(X, Xstar)


def kronDelta(X, Xstar):
    """
    Computes Kronecker delta for rows in X and Xstar.
    Parameters
    ----------
    X: np.ndarray, shape=((n, nfeatures))
        Instances.
    Xstar: np.ndarray, shape((m, nfeatures))
        Instances.
    Returns
    -------
    np.ndarray
        Kronecker delta between row pairs of `X` and `Xstar`.
    """
    return cdist(X, Xstar) < np.finfo(np.float32).eps


class matern32:

    def __init__(self,
                 l=1,
                 sigmaf=1,
                 sigman=1e-6,
                 bounds=None,
                 parameters=['l', 'sigmaf', 'sigman']):
        """
        Matern v=3/2 kernel class.
        Parameters
        ----------
        l: float
            Characteristic length-scale. Units in input space in which posterior GP values do not
            change significantly.
        sigmaf: float
            Signal variance. Controls the overall scale of the covariance function.
        sigman: float
            Noise variance. Additive noise in output space.
        bounds: list
            List of tuples specifying hyperparameter range in optimization procedure.
        parameters: list
            List of strings specifying which hyperparameters should be optimized.
        """

        self.l = l  # noqa: E741
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def K(self, X, Xstar):
        """
        Computes covariance function values over `X` and `Xstar`.
        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances
        Returns
        -------
        np.ndarray
            Computed covariance matrix.
        """
        r = l2norm_(X, Xstar)
        one = (1 + np.sqrt(3 * (r / self.l)**2))
        two = np.exp(-np.sqrt(3 * (r / self.l)**2))
        return self.sigmaf * one * two + self.sigman * kronDelta(X, Xstar)

    def gradK(self, X, Xstar, param):
        """
        Computes gradient matrix for instances `X`, `Xstar` and hyperparameter `param`.
        Parameters
        ----------
        X: np.ndarray, shape=((n, nfeatures))
            Instances
        Xstar: np.ndarray, shape=((n, nfeatures))
            Instances
        param: str
            Parameter to compute gradient matrix for.
        Returns
        -------
        np.ndarray
            Gradient matrix for parameter `param`.
        """
        if param == 'l':
            r = l2norm_(X, Xstar)
            num = 3 * (r**2) * self.sigmaf * np.exp(-np.sqrt(3) * r / self.l)
            return num / (self.l**3)
        elif param == 'sigmaf':
            r = l2norm_(X, Xstar)
            one = (1 + np.sqrt(3 * (r / self.l)**2))
            two = np.exp(-np.sqrt(3 * (r / self.l)**2))
            return one * two
        elif param == 'sigman':
            return kronDelta(X, Xstar)
        else:
            raise ValueError('Param not found')


class Acquisition:

    def __init__(self, mode, eps=1e-06, **params):
        """
        Acquisition function class.
        Parameters
        ----------
        mode: str
            Defines the behaviour of the acquisition strategy. Currently supported values are
            `ExpectedImprovement`, `IntegratedExpectedÌmprovement`, `ProbabilityImprovement`,
            `IntegratedProbabilityImprovement`, `UCB`, `IntegratedUCB`, `Entropy`, `tExpectedImprovement`,
            and `tIntegratedExpectedImprovement`. Integrated improvement functions are only to be used
            with MCMC surrogates.
        eps: float
            Small floating value to avoid `np.sqrt` or zero-division warnings.
        params: float
            Extra parameters needed for certain acquisition functions, e.g. UCB needs
            to be supplied with `beta`.
        """
        self.params = params
        self.eps = eps

        mode_dict = {
            'ExpectedImprovement':
                self.ExpectedImprovement,
            'IntegratedExpectedImprovement':
                self.IntegratedExpectedImprovement,
            'ProbabilityImprovement':
                self.ProbabilityImprovement,
            'IntegratedProbabilityImprovement':
                self.IntegratedProbabilityImprovement,
            'UCB':
                self.UCB,
            'IntegratedUCB':
                self.IntegratedUCB,
            'Entropy':
                self.Entropy,
            'tExpectedImprovement':
                self.tExpectedImprovement,
            'tIntegratedExpectedImprovement':
                self.tIntegratedExpectedImprovement
        }

        self.f = mode_dict[mode]

    def ProbabilityImprovement(self, tau, mean, std):
        """
        Probability of Improvement acquisition function.
        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        Returns
        -------
        float
            Probability of improvement.
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return norm.cdf(z)

    def ExpectedImprovement(self, tau, mean, std):
        """
        Expected Improvement acquisition function.
        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        Returns
        -------
        float
            Expected improvement.
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)[0]

    def UCB(self, tau, mean, std, beta=1.5):
        """
        Upper-confidence bound acquisition function.
        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        beta: float
            Hyperparameter controlling exploitation/exploration ratio.
        Returns
        -------
        float
            Upper confidence bound.
        """
        return mean + beta * std

    def Entropy(self, tau, mean, std, sigman=1.0):
        """
        Predictive entropy acquisition function
        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        sigman: float
            Noise variance
        Returns
        -------
        float:
            Predictive entropy.
        """
        sp2 = std**2 + sigman
        return 0.5 * np.log(2 * np.pi * np.e * sp2)

    def IntegratedExpectedImprovement(self, tau, meanmcmc, stdmcmc):
        """
        Integrated expected improvement. Can only be used with `GaussianProcessMCMC` instance.

        Parameters
        ----------
        tau: float
            Best observed function evaluation
        meanmcmc: array-like
            Means of posterior predictive distributions after sampling.
        stdmcmc
            Standard deviations of posterior predictive distributions after sampling.

        Returns
        -------
        float:
            Integrated Expected Improvement
        """
        acq = [
            self.ExpectedImprovement(tau, np.array([mean]), np.array([std]))
            for mean, std in zip(meanmcmc, stdmcmc)
        ]
        return np.average(acq)

    def IntegratedProbabilityImprovement(self, tau, meanmcmc, stdmcmc):
        """
        Integrated probability of improvement. Can only be used with `GaussianProcessMCMC` instance.
        Parameters
        ----------
        tau: float
            Best observed function evaluation
        meanmcmc: array-like
            Means of posterior predictive distributions after sampling.
        stdmcmc
            Standard deviations of posterior predictive distributions after sampling.
        Returns
        -------
        float:
            Integrated Probability of Improvement
        """
        acq = [
            self.ProbabilityImprovement(tau, np.array([mean]), np.array([std]))
            for mean, std in zip(meanmcmc, stdmcmc)
        ]
        return np.average(acq)

    def IntegratedUCB(self, tau, meanmcmc, stdmcmc, beta=1.5):
        """
        Integrated probability of improvement. Can only be used with `GaussianProcessMCMC` instance.
        Parameters
        ----------
        tau: float
            Best observed function evaluation
        meanmcmc: array-like
            Means of posterior predictive distributions after sampling.
        stdmcmc
            Standard deviations of posterior predictive distributions after sampling.

        beta: float
            Hyperparameter controlling exploitation/exploration ratio.

        Returns
        -------
        float:
            Integrated UCB.
        """
        acq = [
            self.UCB(tau, np.array([mean]), np.array([std]), beta)
            for mean, std in zip(meanmcmc, stdmcmc)
        ]
        return np.average(acq)

    def tExpectedImprovement(self, tau, mean, std, nu=3.0):
        """
        Expected Improvement acquisition function. Only to be used with `tStudentProcess` surrogate.
        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        Returns
        -------
        float
            Expected improvement.
        """
        gamma = (mean - tau - self.eps) / (std + self.eps)
        return gamma * std * t.cdf(gamma, df=nu) + std * (1 + (gamma**2 - 1) /
                                                          (nu - 1)) * t.pdf(
                                                              gamma, df=nu)

    def tIntegratedExpectedImprovement(self, tau, meanmcmc, stdmcmc, nu=3.0):
        """
        Integrated expected improvement. Can only be used with `tStudentProcessMCMC` instance.
        Parameters
        ----------
        tau: float
            Best observed function evaluation
        meanmcmc: array-like
            Means of posterior predictive distributions after sampling.
        stdmcmc
            Standard deviations of posterior predictive distributions after sampling.
        nu:
            Degrees of freedom.
        Returns
        -------
        float:
            Integrated Expected Improvement
        """

        acq = [
            self.tExpectedImprovement(tau,
                                      np.array([mean]),
                                      np.array([std]),
                                      nu=nu)
            for mean, std in zip(meanmcmc, stdmcmc)
        ]
        return np.average(acq)

    def eval(self, tau, mean, std):
        """
        Evaluates selected acquisition function.
        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        Returns
        -------
        float
            Acquisition function value.
        """
        return self.f(tau, mean, std, **self.params)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class EventLogger:

    def __init__(self, gpgo):
        self.gpgo = gpgo
        self.header = 'Evaluation \t Proposed point \t  Current eval. \t Best eval.'
        self.template = '{:6} \t {}. \t  {:6} \t {:6}'
        print(self.header)

    def _printCurrent(self, gpgo):
        eval = str(len(gpgo.GP.y) - gpgo.init_evals)
        proposed = str(gpgo.best)
        curr_eval = str(gpgo.GP.y[-1])
        curr_best = str(gpgo.tau)
        if float(curr_eval) >= float(curr_best):
            curr_eval = bcolors.OKGREEN + curr_eval + bcolors.ENDC
        print(self.template.format(eval, proposed, curr_eval, curr_best))

    def _printInit(self, gpgo):
        for init_eval in range(gpgo.init_evals):
            print(
                self.template.format('init', gpgo.GP.X[init_eval],
                                     gpgo.GP.y[init_eval], gpgo.tau))
