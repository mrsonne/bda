# Bayesian data analysis. Gelman et al. 2004. 
# Example 3.7 

import itertools
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from scipy.special import expit, logit
from scipy.integrate import simps
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
sns.set(color_codes=True)


def proba(x, a, b):
    """Probabiltiy in logistic regression"""
    return expit(a + b*x)


def likelihood(x, n, ny, a, b):
    """Likelihood"""
    _expit = expit(a + b*x)
    return _expit**ny * (1 - _expit)**(n-ny)


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


def sample_posterior(xygrid, pxy, nsamples=100):
    #randomly sample indices since np cannot sample an array of tuples
    return xygrid[np.random.choice(np.arange(len(xygrid)), nsamples, replace=False, p=pxy)]


def plot_posterior_density(X, Y, Z, map_estimate=None, ax=None):
    ax = ax or plt.gca()
    ax.contour(X, Y, Z, 20, cmap='RdGy')
    ax.plot(map_estimate[0], map_estimate[1], 'ro', label='MAP')
    ax.set_xlabel("Intecept")
    ax.set_ylabel("Coefficient")
    ax.legend()


def plot_posterior_samples(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    g = sns.jointplot(x="x", y="y", data=df, color="0.5")
    g.plot_joint(plt.scatter, c="k", s=15, linewidth=1, marker="+")
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("Intecept", "Coefficient")
    plt.show()


def plot_logistic(x, y, df, x_ld50=None, estimate_type=None,
                  alpha=1, ax=None, xlabel='', ylabel=''):
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
        percentiles = np.percentile(y, q=[2.5, 50, 97.5], axis=1)
        ax.plot(x, percentiles[1,:], '-', c='seagreen', linewidth=2., label='Median')
        ax.plot(x, percentiles[0,:], '--', c='seagreen', linewidth=2., label='95 % CI')
        ax.plot(x, percentiles[2,:], '--', c='seagreen', linewidth=2.)

    if x_ld50 is not None:
        ax.plot(x_ld50, 0.5, 'ro', label='{} LD50 @ x={:5.2f}'.format(estimate_type, x_ld50))
        ax.plot([x_ld50, x_ld50], [0, 0.5], 'r--')
        ax.plot([x[0], x_ld50], [0.5, 0.5], 'r--')

    ax.scatter(df.x, df.pdead, marker='o', c='black', edgecolor='white', linewidth=1, s=35, label='p(dead)', zorder=9999)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()


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
    print(result.summary())
    print(result.params)
    return result.params


feauture_names = ['Dose log g/ml'] 
data = {'Dose log g/ml': [-0.86, -0.30, -0.05, 0.73],
        'Number of animals': [5, 5, 5, 5],
        'Number of deaths': [0, 1, 3, 5]}


# flatten data back to individuals. y=0: alive, y=1: dead
y_dead, y_alive = 1, 0
X = np.atleast_2d([_x for idx, _x in enumerate(data['Dose log g/ml']) for _ in range(data['Number of animals'][idx])]).T
y = [y_alive if _y <= idx_n else y_dead for idx, _y in enumerate(data['Number of deaths']) for idx_n in range(data['Number of animals'][idx])]

# Go from raw dataset to summary (superfluous but good practice)
df = pd.DataFrame({'x': X.flatten(), 'y': y})
rename_cols = dict(size='ntot', sum='ndead')
grouped = df.set_index('x').stack().groupby('x').agg(rename_cols.keys()).rename(columns=rename_cols)
grouped['nalive'] = grouped.ntot - grouped.ndead
grouped['pdead'] = grouped.ndead/grouped.ntot
grouped = grouped.reset_index()
print(grouped)

# MLE logistic regression using sklearn
dose, p_dead, x_ld50, _ = logit_sklean(X, y, feauture_names)
plot_logistic(dose, p_dead, grouped, x_ld50, estimate_type='MLE',
              xlabel=feauture_names[0], ylabel='P(Dead)')

# MLE logistic regression using statsmodels
logit_statsmodels(X, y)

# Calculate posterior density on a grid
av, bv, density, a_vals, b_vals, delta_a, delta_b = posterior_density_grid(data['Dose log g/ml'],
                                                                           data['Number of animals'],
                                                                           data['Number of deaths'],
                                                                           na=500, nb=500)


pdf_grid = make_pdf_grid(density, delta_a, delta_b)
print('Total posterior probability (validation):', simps(simps(pdf_grid, b_vals), a_vals))
idxs_map = pdf_grid.argmax()
map_estimate = av.ravel()[idxs_map], bv.ravel()[idxs_map]
print('MAP estimate', map_estimate)
fig, ax = plt.subplots(nrows=1, ncols=1)
plot_posterior_density(av, bv, pdf_grid, map_estimate, ax)
plt.show()

# Samples from posterior distribution
prob_grid = make_prob_grid(density)
print('Total posterior probability (validation):', prob_grid.sum())
xypairs = np.array(list(itertools.product(a_vals, b_vals)))
xyvals = sample_posterior(xypairs, np.ravel(prob_grid), nsamples=1000)
plot_posterior_samples(xyvals[:,0], xyvals[:,1])

# LD50 histogram for b > 0
idxs = xyvals[:,1] > 0
posterior_ld50 = -xyvals[idxs,0]/xyvals[idxs,1]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
g = sns.distplot(posterior_ld50, kde=False, rug=False, norm_hist=True, axlabel="LD50", ax=axs[1])
axs[1].set_ylabel('Density')

# Posterior predictive death probability 
dose = np.linspace(-1, 1)
shape = dose.shape[0], len(idxs)
p_dead = np.empty(shape)
for idx, (a, b) in enumerate(xyvals[idxs]):
    p_dead[:, idx] = proba(dose, a, b)
x_ld50 = -map_estimate[0]/map_estimate[1]
plot_logistic(dose, p_dead, grouped, x_ld50, estimate_type='MAP', alpha=0.1,
              xlabel=feauture_names[0], ylabel='p(dead)', ax=axs[0])
