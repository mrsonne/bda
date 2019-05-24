from scipy.integrate import simps
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sec_3_7 as sec37

def run_sec37():
    data, feature_names = sec37.get_data()
    X, y = sec37.inflate_data(data)
    grouped = sec37.group_data(X, y)
    print(grouped)

    # MLE logistic regression using sklearn
    dose, p_dead, x_ld50, _ = sec37.logit_sklean(X, y, feature_names)
    sec37.plot_logistic(dose, p_dead, grouped, x_ld50, estimate_type='MLE',
                        xlabel=feature_names[0], ylabel='$p(Dead)$')

    # MLE logistic regression using statsmodels
    params, cov = sec37.logit_statsmodels(X, y)

    sec37.calc_cov(X, *params[::-1])

    # Calculate posterior density on a grid
    av, bv, density, a_vals, b_vals, delta_a, delta_b = sec37.posterior_density_grid(data['Dose log g/ml'],
                                                                                     data['Number of animals'],
                                                                                     data['Number of deaths'],
                                                                                     na=500, nb=500)
 

    # Normalize density
    pdf_grid = sec37.make_pdf_grid(density, delta_a, delta_b)
    print('Total posterior probability (validation):', simps(simps(pdf_grid, b_vals), a_vals))

    # Get MAP estimate
    idxs_map = pdf_grid.argmax()
    map_estimate = av.ravel()[idxs_map], bv.ravel()[idxs_map]
    print('MAP estimate', map_estimate)

    # Sample from posterior distribution
    prob_grid = sec37.make_prob_grid(density)
    print('Total posterior probability (validation):', prob_grid.sum())
    ab_vals = sec37.sample_posterior(a_vals, b_vals, prob_grid, nsamples=1000)

    # Plot the density with MAP
    sec37.plot_posterior_density(av, bv, pdf_grid, map_estimate, xysamples=ab_vals)

    sec37.plot_posterior_samples(ab_vals[:,0], ab_vals[:,1])

    # Only use parameters where b>0 
    ab_vals = ab_vals[ab_vals[:,1] > 0]

    # LD50 histogram for b > 0
    axs = sec37.plot_ld50_hist(ab_vals)

    dose, p_dead = sec37.sample_model(ab_vals)
    x_ld50_map = -map_estimate[0]/map_estimate[1]
    sec37.plot_logistic(dose, p_dead, grouped, x_ld50_map,
                        estimate_type='MAP', alpha=0.1,
                        xlabel=feature_names[0], ylabel='p(dead)',
                        ax=axs[0])

    sec37.normal_approximation(ab_vals, map_estimate, params, cov)

    nchains = 3
    pars0 = np.random.multivariate_normal(params[::-1], np.rot90(cov, k=2), nchains)
    sec37.mcmc(data['Dose log g/ml'],
               data['Number of animals'],
               data['Number of deaths'], 
               pars0, nsamples=25000)

def run_chaper3():
    run_sec37()


if __name__ == '__main__':
    run_chaper3()