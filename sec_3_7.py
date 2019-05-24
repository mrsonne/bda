# Bayesian data analysis. Gelman et al. 2004. 
# Example 3.7 

import itertools
import pandas as pd
import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib.patches import Ellipse

sns.set(color_codes=True)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
#        vals, vecs = np.linalg.eig(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    ax = ax or plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)
    return ellip


def normal_approximation(ab_vals, map_estimate, params_mle, cov_mle):
    mean_data = np.mean(ab_vals, axis=0)
    cov_data = np.cov(ab_vals.T)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(ab_vals[:,0], ab_vals[:,1], color='k', marker='.', alpha=0.5, 
            linewidth=0, label='Samples (true posterior)')
    # python 2 compatibility
    ax.plot(mean_data[0], mean_data[1], 'o',
            label='Data mean: {}'.format(np.array2string(mean_data, precision=2)), 
            zorder=999, color='magenta')
    ax.plot(params_mle[1], params_mle[0], 'o',
            label='MLE: {}'.format(np.array2string(params_mle[::-1], precision=2)),
            zorder=999, color='green')
    ax.plot(map_estimate[0], map_estimate[1], 'o',
            label='MAP: {}'.format(np.array2string(np.array(map_estimate), precision=2)),
            zorder=999, color='blue')
    plot_cov_ellipse(cov_data, mean_data, ax=ax, nstd=2, label="Data COV  @ 2x std", fill=None, edgecolor="magenta", linewidth=1, zorder=999)
    plot_cov_ellipse(np.rot90(cov_mle, k=2), params_mle[::-1], ax=ax, nstd=2, label="MLE COV @ 2x std", fill=None, edgecolor="green", linewidth=1, zorder=999)
    ax.set_ylim((-10, 40))
    ax.set_xlim((-5, 10))
    ax.legend()
    plt.show()


def proba(x, a, b):
    """Probabiltiy in logistic regression"""
    return expit(a + b*x)


def likelihood(x, n, ny, a, b):
    """Likelihood"""
    _expit = expit(a + b*x)
    return _expit**ny * (1 - _expit)**(n-ny)


def log_likelihood(xs, ns, nys, a, b):
    """ Log-Likelihood for the data """
    return sum(np.log([likelihood(x, n, ny, a, b) for x, n, ny in zip(xs, ns, nys)]))


def pdf_joint_posterior(xs, ns, nys, a, b):
    """Joint posterior distriution"""
    product = 1.
    for x, n, ny in zip(xs, ns, nys):
        product *= likelihood(x, n, ny, a, b)
    return product

vfun = np.vectorize(pdf_joint_posterior, excluded=[0, 1, 2])

def posterior_density_grid(xs, ns, nys, na=100, nb=100, a_minmax=(-5,10), b_minmax=(-10,40)):
    """ Calculate posterior density on a grid """
    a_vals = np.linspace(a_minmax[0], a_minmax[1], na)
    b_vals = np.linspace(b_minmax[0], b_minmax[1], nb)
    delta_a = float(a_minmax[1]-a_minmax[0])/(na-1)
    delta_b = float(b_minmax[1]-b_minmax[0])/(nb-1)
    AV, BV = np.meshgrid(a_vals, b_vals, sparse=False, indexing='xy')
    return AV, BV, np.matrix(vfun(xs, ns, nys, a=AV, b=BV)), a_vals, b_vals, delta_a, delta_b


def make_pdf_grid(density, delta_a, delta_b):
    return density/density.sum()/delta_a/delta_b


def make_prob_grid(density):
    return density/density.sum()


def sample_posterior(x, y, pxy, nsamples=100):
    xypairs = np.array(list(itertools.product(x, y)))
    #randomly sample indices since np cannot sample an array of tuples
    return xypairs[np.random.choice(np.arange(len(xypairs)), nsamples, replace=False, p=np.ravel(pxy, order='F'))]


def plot_posterior_density(X, Y, Z, map_estimate=None, ax=None, xysamples=None):
    ax = ax or plt.gca()

    if xysamples is not None:
        # marker='+' doesnt render data in notebook
        ax.plot(xysamples[:,0], xysamples[:,1], color='k', marker='.',
                alpha=0.5, linewidth=0, label='Samples from posterior')

    ax.contour(X, Y, Z, 20, cmap='RdGy')
    ax.plot(map_estimate[0], map_estimate[1], 'ro', label='MAP')


    ax.set_xlabel("Intecept")
    ax.set_ylabel("Coefficient")
    ax.legend()
    plt.show()


def plot_posterior_samples(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    g = sns.jointplot(x="x", y="y", data=df, color="0.5")
    g.plot_joint(plt.scatter, c="k", s=15, linewidth=1, marker="+")
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("Intecept", "Coefficient")
    plt.show()


def plot_logistic(x, y, df, x_ld50=None, estimate_type=None,
                  alpha=1, ax=None, xlabel='', ylabel=''):
    """ Plot result(s) from logistic regression along with data """
    ax = ax or plt.gca()
    many = False
    label = 'Model'
    if len(y.shape) == 2 and y.shape[1] > 1:
        _label = ''
        many = True
    else:
        _label = label
    ax.plot(x, y, '-', c='0.', linewidth=0.5, alpha=alpha, label=_label)

    if many:
        ax.plot(x, y[:,0], '-', c='0.', linewidth=0.5, alpha=1., label=label)
        percentiles = np.percentile(y, q=[2.5, 50., 97.5], axis=1)
        ax.plot(x, percentiles[1,:], '-', c='seagreen', linewidth=2., label='Median')
        ax.fill_between(x, percentiles[0,:],  percentiles[2,:], alpha=0.6,
                        color="seagreen", label='95 % CI')

    if x_ld50 is not None:
        ax.plot(x_ld50, 0.5, 'ro', label='{} LD50 @ x={:5.2f}'.format(estimate_type, x_ld50))
        ax.plot([x_ld50, x_ld50], [0, 0.5], 'r--')
        ax.plot([x[0], x_ld50], [0.5, 0.5], 'r--')

    label = u'$p(\mathrm{dead}) = n_\mathrm{dead}/n_\mathrm{tot} |_\mathrm{dose}$'
    ax.scatter(df.x, df.pdead, marker='o', c='black',
               edgecolor='white', linewidth=1, s=35,
               label=label, zorder=9999)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()


def plot_ld50_hist(ab_vals):
    """ LD50 histogram for b > 0 """
    ld50_posterior = -ab_vals[:,0]/ab_vals[:,1]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    g = sns.distplot(ld50_posterior, kde=False, rug=False, norm_hist=True, axlabel="LD50", ax=axs[1])
    axs[1].set_ylabel('Density')
    return axs



def logit_sklean(X, y, feauture_names):
    print("\n*** sklearn ***")
    # logistic regression with no regularization. C is the inverse regularization strength
    model = LogisticRegression(C=1e9, solver='lbfgs').fit(X, y)

    print('Coefficients')
    fstr = '   {} {}'
    for i, feature_name in enumerate(feauture_names):
        print(fstr.format('Coef for ' + feature_name, model.coef_[0][i]))
    print(fstr.format('Intercept:', model.intercept_[0]))

    x_predict = np.linspace(-1, 1)
    y_predict = model.predict_proba(x_predict.reshape(-1, 1))[:,1]

    x_ld50 = -model.intercept_[0]/model.coef_[0][0]

    return x_predict, y_predict, x_ld50, model


def logit_statsmodels(X, y):
    print("\n*** statsmodels ***")
    # include intecepts
    model = sm.Logit(y, sm.add_constant(X, prepend=False)) 
    result = model.fit()
    print('params', result.params)
    print('cov', result.cov_params())
    print('SE', np.sqrt(np.diag(result.cov_params())))
    print(result.summary())
    return result.params, result.cov_params()


def calc_cov(X, a, b):
    """ 
    Calculate the parameter covariance from the Hessian 
    For the Hessian see e.g.
    https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function

    https://www.quora.com/How-can-I-show-that-the-Hessian-for-log-likelihood-for-logistic-regression-is-negative-definite

    https://www.youtube.com/watch?v=hWLdFMccpTY
    https://www.youtube.com/watch?v=jUwjbiBUR-k

    Hessian as covariance 
    https://www.cse.wustl.edu/~garnett/cse515t/spring_2019/files/lecture_notes/8.pdf
    Gelman
    """
    p = proba(X, a, b)[:,0]
    D = np.diag(p*(1 - p))
    X = sm.add_constant(X, prepend=False).T
    H = np.matmul(X, np.matmul(D, X.T))
    print('Hessian', H)
    print('COV', np.linalg.inv(H))


def get_data():
	feature_names = ['Dose log g/ml'] 
	data = {'Dose log g/ml': [-0.86, -0.30, -0.05, 0.73],
			'Number of animals': [5, 5, 5, 5],
			'Number of deaths': [0, 1, 3, 5]}
	return data, feature_names


def inflate_data(data):
	"""inflate data back to individuals. y=0: alive, y=1: dead """
	y_dead, y_alive = 1, 0
	X = np.atleast_2d([_x for idx, _x in enumerate(data['Dose log g/ml']) for _ in range(data['Number of animals'][idx])]).T
	y = [y_alive if _y <= idx_n else y_dead for idx, _y in enumerate(data['Number of deaths']) for idx_n in range(data['Number of animals'][idx])]
	return X, y
	

def group_data(X, y):
	""" Go from raw dataset to summary (superfluous but good practice) """
	df = pd.DataFrame({'x': X.flatten(), 'y': y})
	rename_cols = dict(size='ntot', sum='ndead')
	grouped = df.set_index('x').stack().groupby('x').agg(rename_cols.keys()).rename(columns=rename_cols)
	grouped['nalive'] = grouped.ntot - grouped.ndead
	grouped['pdead'] = grouped.ndead/grouped.ntot
	grouped = grouped.reset_index()
	return grouped


def sample_model(ab_vals):
    """ Posterior predictive death probability """
    dose = np.linspace(-1, 1)
    shape = dose.shape[0], len(ab_vals)
    p_dead = np.empty(shape)
    for idx, (a, b) in enumerate(ab_vals):
        p_dead[:, idx] = proba(dose, a, b)
    return dose, p_dead


def mcmc(xs, ns, nys, pars0, nsamples=1000, step_scale=1.7, nburn=None):
    """ Verbose and simple MCMC sample using flat improper priors """
    nburn = nburn or int(0.25*nsamples)
    nchains = len(pars0)
    npar = len(pars0[0])
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for ichain in range(nchains):
        acc = 0
        trace = np.empty((nsamples, npar), float)
        pars = pars0[ichain]
        print('Initial parameters {} chain {}'.format(pars, ichain + 1))
        for isample in range(nsamples):
            # step. here uniform for simplicity
            dx = np.random.uniform(-1, 1, size=npar)*step_scale
            # new trial position
            pars_trial = pars + dx 
            # Metropolis ratio (here its just the likelihood ratio due to flat improper priors)
            mr = log_likelihood(xs, ns, nys, *pars_trial) - log_likelihood(xs, ns, nys, *pars) 
            # reject/accept trial
            if min(np.exp(mr), 1) > np.random.uniform():
                acc += 1
                pars = pars_trial
            trace[isample,:] = pars
        print('Acceptance ratio', acc/nsamples)
        axs.plot(trace[nburn:,0], trace[nburn:,1], '-o', alpha=0.25)
    plt.show()