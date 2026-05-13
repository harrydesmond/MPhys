import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
import matplotlib.collections as mcoll
import scipy.stats as stats
import numpyro
import numpyro.distributions as dist
from jax import random
import corner
import arviz as az
import jax.numpy as jnp
import ast
import matplotlib.lines as mlines
from numpyro.infer.util import log_likelihood



def NH_cuts(Chosen_galaxies_NH, galaxy_dataframe_NH, ratio_lim, gas_ratio_lim, galaxy_stellar_mass_lim, r_half_lim, alignment_lim, v_rot_over_sigma_v_lim):
    # NH Selection criteria 

    ratio = galaxy_dataframe_NH['ratio']
    gas_ratio = galaxy_dataframe_NH['gas_ratio']
    galaxy_stellar_mass = galaxy_dataframe_NH['mass']
    gal_number = galaxy_dataframe_NH['Gal_number']
    r_half = galaxy_dataframe_NH['r_half']
    gas_r_half = np.array(galaxy_dataframe_NH['gas_r_half'])
    specific_angular_mom = np.array(galaxy_dataframe_NH['specific_angular_mom'])
    h1_specific_angular_mom = np.array(galaxy_dataframe_NH['h1_specific_angular_mom'])


    mask = np.isin(gal_number, Chosen_galaxies_NH)

    ratio = np.array(ratio)[mask]
    gas_ratio = np.array(gas_ratio)[mask]
    galaxy_stellar_mass = np.array(galaxy_stellar_mass)[mask]
    gal_number = np.array(gal_number)[mask]
    r_half = np.array(r_half)[mask]
    gas_r_half = np.array(gas_r_half)[mask]
    specific_angular_mom = specific_angular_mom[mask]
    h1_specific_angular_mom = h1_specific_angular_mom[mask]


    # V_rot/sigma_v

    v_rot_over_sigma_v_csv = pd.read_csv('/mnt/users/darnej/MPhys/v_sig_ratios_NH.csv')

    v_rot_over_sigma_v = v_rot_over_sigma_v_csv['mean_v_sig_ratio']

    gal_number_v = v_rot_over_sigma_v_csv['gal_number']

    mask = np.isin(gal_number_v, Chosen_galaxies_NH)

    v_rot_over_sigma_v = np.array(v_rot_over_sigma_v)[mask]

    plt.hist(v_rot_over_sigma_v[v_rot_over_sigma_v > 0], bins = 100)
    plt.axvline(x = v_rot_over_sigma_v_lim, color = 'r', linestyle = '--', label = 'Cut at 2')

    # Gas and stellar allignment

    alignment_ratios = []
    for i in range(len(specific_angular_mom)):

        cleaned = specific_angular_mom[i].replace("np.float64", "") 
        lst = ast.literal_eval(cleaned)        
        specific_angular_mom[i] = np.array(lst, dtype=float) 

        cleaned = h1_specific_angular_mom[i].replace("unyt_quantity", "") 
        cleaned = cleaned.replace("kpc/s", "")
        cleaned = cleaned.replace(", ''", "")
        lst = ast.literal_eval(cleaned)
        h1_specific_angular_mom[i] = np.array(lst, dtype=float) 

        alignment = np.dot(specific_angular_mom[i], h1_specific_angular_mom[i]) / (np.linalg.norm(specific_angular_mom[i]) * np.linalg.norm(h1_specific_angular_mom[i]))
        alignment_ratios.append(alignment)

    #Defining NH cuts

    ratio_mask = ratio < ratio_lim

    gas_ratio_mask = gas_ratio < gas_ratio_lim

    galaxy_stellar_mass_mask = (galaxy_stellar_mass > galaxy_stellar_mass_lim)

    r_half_mask = (r_half < r_half_lim) & (r_half > 0)

    alignment_mask = np.array(alignment_ratios) > alignment_lim

    rotation_mask = v_rot_over_sigma_v > v_rot_over_sigma_v_lim

    total_mask = ratio_mask & galaxy_stellar_mass_mask & gas_ratio_mask & r_half_mask & rotation_mask & alignment_mask

    print('stellar mass cut', sum(galaxy_stellar_mass_mask))
    print('Stellar ratio cut', sum(ratio_mask & galaxy_stellar_mass_mask))
    print('Gas ratio cut', sum(gas_ratio_mask & ratio_mask & galaxy_stellar_mass_mask))
    print('R half cut', sum(r_half_mask & galaxy_stellar_mass_mask & gas_ratio_mask & ratio_mask))
    print('alignment mask cut', sum(alignment_mask & r_half_mask & galaxy_stellar_mass_mask & gas_ratio_mask & ratio_mask))
    print('vrot/sigma_v cut', sum(rotation_mask & alignment_mask & r_half_mask & galaxy_stellar_mass_mask & gas_ratio_mask & ratio_mask))

    Chosen_galaxies_NH = gal_number[total_mask]

    print('Number of galaxies after cuts is:',len(Chosen_galaxies_NH))

    #Removing specific galaxies from NH

    deffo_mergers = [128, 162, 575, 594]

    for x in deffo_mergers:
        if x in Chosen_galaxies_NH:
            Chosen_galaxies_NH = Chosen_galaxies_NH[Chosen_galaxies_NH != x]


    n_within_ten_perc_r_vir = [375, 427, 241, 413, 47, 49, 14]

    for x in n_within_ten_perc_r_vir:
        if x in Chosen_galaxies_NH:
            Chosen_galaxies_NH = Chosen_galaxies_NH[Chosen_galaxies_NH != x]

    print('Final number of chosen galaxies in NH is:',len(Chosen_galaxies_NH))
    np.savetxt('/mnt/users/darnej/MPhys/Chosen_galaxies_final_NH.txt',Chosen_galaxies_NH, delimiter=',')


def TNG_cuts(Chosen_galaxies_TNG, galaxy_dataframe_TNG, ratio_lim, gas_ratio_lim, galaxy_stellar_mass_lim, r_half_lim, alignment_lim, min_distance_lim, v_rot_over_sigma_v_lim):

    # TNG Selection criteria 

    ratio = galaxy_dataframe_TNG['ratio']
    gas_ratio = galaxy_dataframe_TNG['gas_ratio']
    galaxy_stellar_mass = galaxy_dataframe_TNG['stellar_mass']
    gal_number = galaxy_dataframe_TNG['gal_number']
    r_half = galaxy_dataframe_TNG['r_half']
    gas_r_half = np.array(galaxy_dataframe_TNG['gas_r_half'])
    gas_com = np.array(galaxy_dataframe_TNG['gas_COM'])
    specific_angular_mom = np.array(galaxy_dataframe_TNG['specific_angular_mom'])
    h1_specific_angular_mom = np.array(galaxy_dataframe_TNG['h1_specific_angular_mom'])

    mask = np.isin(gal_number, Chosen_galaxies_TNG)

    ratio = np.array(ratio)[mask]
    gas_ratio = np.array(gas_ratio)[mask]
    galaxy_stellar_mass = np.array(galaxy_stellar_mass)[mask]
    gal_number = np.array(gal_number)[mask]
    r_half = np.array(r_half)[mask]
    gas_r_half = np.array(gas_r_half)[mask]
    gas_com = gas_com[mask]
    specific_angular_mom = specific_angular_mom[mask]
    h1_specific_angular_mom = h1_specific_angular_mom[mask]


    # V_rot/sigma_v

    v_rot_over_sigma_v_csv = pd.read_csv('/mnt/users/darnej/MPhys/TNG-50/v_sig_ratios_TNG.csv')

    v_rot_over_sigma_v = v_rot_over_sigma_v_csv['mean_v_sig_ratio']


    plt.hist(v_rot_over_sigma_v[v_rot_over_sigma_v > 0], bins = 100)
    plt.xlabel('V_rot/sigma_v')
    plt.ylabel('Number of galaxies')
    plt.title('Histogram of V_rot/sigma_v for TNG galaxies')
    plt.axvline(x = v_rot_over_sigma_v_lim, color = 'r', linestyle = '--', label = 'Cut at 2.5')
    plt.show()


    # Gas and stellar allignment

    alignment_ratios = []
    for i in range(len(specific_angular_mom)):

        cleaned = specific_angular_mom[i].replace("np.float64", "") 
        lst = ast.literal_eval(cleaned)        
        specific_angular_mom[i] = np.array(lst, dtype=float) 

        cleaned = h1_specific_angular_mom[i].replace("np.float64", "") 
        lst = ast.literal_eval(cleaned)      
        h1_specific_angular_mom[i] = np.array(lst, dtype=float) 

        alignment = np.dot(specific_angular_mom[i], h1_specific_angular_mom[i]) / (np.linalg.norm(specific_angular_mom[i]) * np.linalg.norm(h1_specific_angular_mom[i]))
        alignment_ratios.append(alignment)


    #Defining TNG cuts

    ratio_mask = ratio < ratio_lim

    gas_ratio_mask = gas_ratio < gas_ratio_lim

    galaxy_stellar_mass_mask = (galaxy_stellar_mass > galaxy_stellar_mass_lim) 

    r_half_mask = (r_half < r_half_lim) & (r_half > 0)

    alignment_mask = np.array(alignment_ratios) > alignment_lim

    min_distance = min_distance_lim #kpc

    rotation_mask = v_rot_over_sigma_v > v_rot_over_sigma_v_lim

    total_mask = ratio_mask & galaxy_stellar_mass_mask & gas_ratio_mask & r_half_mask & rotation_mask & alignment_mask


    print('stellar mass cut', sum(galaxy_stellar_mass_mask))
    print('Stellar ratio cut', sum(ratio_mask & galaxy_stellar_mass_mask))
    print('Gas ratio cut', sum(gas_ratio_mask & ratio_mask & galaxy_stellar_mass_mask))
    print('R half cut', sum(r_half_mask & galaxy_stellar_mass_mask & gas_ratio_mask & ratio_mask))
    print('vrot/sigma_v cut', sum(rotation_mask & r_half_mask & galaxy_stellar_mass_mask & gas_ratio_mask & ratio_mask))
    print('alignment mask cut', sum(alignment_mask & rotation_mask & r_half_mask & galaxy_stellar_mass_mask & gas_ratio_mask & ratio_mask))



    Chosen_galaxies_TNG = gal_number[total_mask]

    print('Number of galaxies after cuts is:',len(Chosen_galaxies_TNG))

    chosen_gals_com = gas_com[total_mask]

    #Cleaning COM of gals

    for i in range(len(chosen_gals_com)):
        cleaned = chosen_gals_com[i].replace("np.float64", "") 


        lst = ast.literal_eval(cleaned)   
        chosen_gals_com[i] = np.array(lst, dtype=float) 



    #Removing galaxies within 100 kpc of each other

    gal_com = []
    for i in range(len(Chosen_galaxies_TNG)):
        gal_com.append(list(chosen_gals_com)[i])

    all_close_gals = defaultdict(list)

    for i in range(len(gal_com)):
        holder = gal_com
        holder = holder - gal_com[i]
        holder = np.linalg.norm(holder, axis=1)

        #DEFINE MINIMUM DISTANCE FOR CLOSE GALAXIES
        mask = (holder < min_distance) & (holder > 0)

        if np.sum(mask) > 0:
            close_gals = Chosen_galaxies_TNG[mask]
            for gal in close_gals:
                all_close_gals[gal].append(gal)



    close_gals = list(all_close_gals.keys())

    print('Number of gals within',str(min_distance) + 'Kpc is',len(close_gals))

    chosen_gals_final = []
    for gal in Chosen_galaxies_TNG:
        if gal not in close_gals:
            chosen_gals_final.append(gal)

    Chosen_galaxies_TNG = np.array(chosen_gals_final)
    print('Final number of chosen galaxies in TNG is:',len(Chosen_galaxies_TNG))
    np.savetxt('/mnt/users/darnej/MPhys/TNG-50/Chosen_galaxies_final_TNG.txt',Chosen_galaxies_TNG, delimiter=',')


def get_pytree_TNG_NH(Chosen_galaxies_TNG, Chosen_galaxies_NH):

    directory = '/mnt/users/darnej/MPhys/TNG-50/pytree_results'

    files = glob.glob(directory + '/*.csv')

    pytree_results_TNG = defaultdict(list)

    for i in Chosen_galaxies_TNG:
        i = int(i)
        file = directory + '/pytree_results_' + str(i) + '.csv'
        gal_number = int(i)
        csv = pd.read_csv(file, float_precision='round_trip')
        a_r = np.array(csv['mean_a_r_py'])
        a_tot = np.array(csv['mean_a_r_total_py'])
        mean_r = np.array(csv['mean_r'])
        pytree_results_TNG[gal_number] = [a_r, a_tot, mean_r]

    #PYTREEGRAV RESULTS NH

    directory = '/mnt/users/darnej/MPhys/pytree_results'

    files = glob.glob(directory + '/*.csv')

    pytree_results_NH = defaultdict(list)

    for i in Chosen_galaxies_NH:
        i = int(i)
        file = directory + '/pytree_results_' + str(i) + '.csv'
        gal_number = int(i)
        csv = pd.read_csv(file, float_precision='round_trip')
        a_r = np.array(csv['mean_a_r_py'])
        a_tot = np.array(csv['mean_a_r_total_py'])
        mean_r = np.array(csv['mean_r'])
        pytree_results_NH[gal_number] = [a_r, a_tot, mean_r]
    
    return pytree_results_TNG, pytree_results_NH


def Plot_pytree(pytree_results_TNG, gas_r_half_TNG, gal_indices_TNG, min_radius_TNG, max_radius_TNG):

    #PLOTTING TNG PYTREEGRAV RESULTS

    fig = plt.figure(figsize=(13, 8))

    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern Roman
    plt.rcParams["xtick.labelsize"] = 15   # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 15   # Increase y-axis tick font size


    g_tot_all_tree = []
    g_bar_all_tree = []

    for gal in pytree_results_TNG:
    
        g_tot = pytree_results_TNG[gal][1]
        g_bar = pytree_results_TNG[gal][0]
        mean_r = pytree_results_TNG[gal][2]

        r_half = gas_r_half_TNG[gal == gal_indices_TNG][0]
        
        r_over_r_half = mean_r / r_half

        #MASK TO SAY HOW MANY MULTIPLES OF GAS_R_HALF YOU WANT TO GO OUT TO 

        r_mask_tree_TNG = (r_over_r_half <= max_radius_TNG) & (r_over_r_half >= min_radius_TNG)

        g_tot = g_tot[r_mask_tree_TNG]
        g_bar = g_bar[r_mask_tree_TNG]
        mean_r = mean_r[r_mask_tree_TNG]

        mask = (g_bar > 10**(-14)) & (g_tot > 10**(-14))
        g_tot = g_tot[mask]
        g_bar = g_bar[mask]
        mean_r = mean_r[mask]



        plt.scatter(np.log10(g_bar), np.log10(g_tot), s = 2, alpha = 0.5, c = mean_r)

        g_tot_all_tree = g_tot_all_tree + list(g_tot)
        g_bar_all_tree = g_bar_all_tree + list(g_bar)


    g_tot_all_tree = np.array(g_tot_all_tree)
    g_bar_all_tree = np.array(g_bar_all_tree)

    print(len(g_tot_all_tree))
    print(len(g_bar_all_tree))



    mask = (g_bar_all_tree > 0) & (g_tot_all_tree > 0) 

    g_tot_all_tree = g_tot_all_tree[mask]
    g_bar_all_tree = g_bar_all_tree[mask]

    print(len(g_tot_all_tree))
    print(len(g_bar_all_tree))

    mean_TNG, bins, bin_number = stats.binned_statistic(np.log10(g_bar_all_tree), np.log10(g_tot_all_tree), bins=30, statistic='mean')

    bincenters_TNG = 0.5*(bins[1:]+bins[:-1])

    plt.scatter(bincenters_TNG, mean_TNG, label = 'Mean Acceleration', color = 'red', s = 40)
    #plt.plot(bincenters, mean, color = 'red', linewidth = 4)



    x = np.linspace(-16, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')


    key = list(pytree_results_TNG.keys())[0]
    colors = pytree_results_TNG[key][2]
    points = np.array([np.log10(pytree_results_TNG[key][0]), np.log10(pytree_results_TNG[key][1])]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.array(colors).min(), np.array(colors).max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(colors)

    cbar = fig.colorbar(lc, ax = plt.gca())
    cbar.set_label("Radial Distance [Kpc]", rotation=270, labelpad=25, fontsize = 17)



    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.legend(prop={'size': 15})
    plt.xlim(-12.5,-8)
    plt.title('PyTreeGrav Results', fontsize = 20)
    plt.ylim(-12.5,-8)

    plt.show()

    return g_bar_all_tree, g_tot_all_tree, bincenters_TNG, mean_TNG


def Power_law_MCMC(gbar, gtot):

    log_gbar = jnp.log10(gbar)
    log_gtot = jnp.log10(gtot)

    def model(x=None, y=None):
        A = numpyro.sample("A", dist.Uniform(-5, 5))
        k = numpyro.sample("k", dist.Uniform(-5, 5))
        sigma = numpyro.sample("sigma", dist.Uniform(0, 5))

        # log10(g_tot) = A + k*log10(gbar),  where A = log10(normalisation)
        log_gtot_pred = A + k * jnp.log10(gbar)

        numpyro.sample("obs", dist.Normal(log_gtot_pred, sigma), obs=log_gtot)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=2000)
    mcmc.run(rng_key_, x=gbar, y=log_gtot)

    mcmc.print_summary()
    samples = mcmc.get_samples()
    res = az.from_numpyro(mcmc)

    # ---- BIC computation (approximate MLE via best posterior draw) ----
    ll_dict = log_likelihood(model, samples, x=gbar, y=log_gtot)
    ll_per_draw = ll_dict["obs"].sum(axis=-1)
    logL_hat = float(ll_per_draw.max())

    n = int(log_gtot.shape[0])
    k_params = 3  # A, k, sigma
    bic_value = k_params * np.log(n) - 2.0 * logL_hat

    print(f"logL_hat (approx) = {logL_hat:.3f}")
    print(f"BIC = {bic_value:.3f}")

    # Extract posterior samples as a NumPy array (same as you did)
    samples_array = np.array([res.posterior[var].values.flatten()
                              for var in res.posterior.keys()]).T

    fig = corner.corner(
        samples_array,
        labels=[r'$\log_{10}(A)$', 'k', '$\sigma$'],
        show_titles=True,
        smooth=1,
        label_kwargs={"fontsize": 15}
    )
    plt.show()

    return samples_array, bic_value


def Power_law_MCMC(gbar, gtot):

    log_gbar = jnp.log10(gbar)
    log_gtot = jnp.log10(gtot)

    # g_tot = A*g_bar**k
    # log_g_tot = log_A + k*log_g_bar

    def model(x=None, y=None):

        A = numpyro.sample("A", dist.Uniform(-5, 5))
        k = numpyro.sample("k", dist.Uniform(-5, 5))
        sigma = numpyro.sample("sigma", dist.Uniform(0, 5))

        # Prediction of y — A = log10(normalisation), no fixed 1e-2 scaling
        log_gtot_pred = A + k * jnp.log10(gbar)

        # Gaussian likelihood
        numpyro.sample("obs", dist.Normal(log_gtot_pred, sigma), obs=log_gtot)

    # Run MCMC
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=2000)
    mcmc.run(rng_key_, x=gbar, y=log_gtot)

    mcmc.print_summary()
    samples = mcmc.get_samples()
    res = az.from_numpyro(mcmc)

    # Extract posterior samples as a NumPy array
    samples_array = np.array([res.posterior[var].values.flatten() for var in res.posterior.keys()]).T

    # Corner plot
    fig = corner.corner(
        samples_array,
        labels=[r'$\log_{10}(A)$', 'k', '$\sigma$'],  # Ensure correct variable names
        show_titles=True,
        smooth = 1,
        label_kwargs={"fontsize": 15}
    )
    plt.show()

    return samples_array


def Plot_pytree_power_law(samples_array, pytree_results, gas_r_half, gal_indices, min_radius, max_radius, bincenters, mean):

    #TNG PLOT WITH LINE FIT

    A = np.median(samples_array[:, 0])
    k = np.median(samples_array[:, 1])
    sigma_pytree = np.median(samples_array[:, 2])

    print('A:', A)
    print('k:', k)
    print('sigma:', sigma_pytree)

    g_bar = np.logspace(-8, -14, 40)

    log_g_tot_pred_pytree = A + k * np.log10(g_bar)

    plt.figure(figsize=(10,8))

    x = np.linspace(-15, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')

    for gal in pytree_results:
        g_tot = pytree_results[gal][1]
        g_bar_individual = pytree_results[gal][0]
        mean_r = pytree_results[gal][2]

        r_half = gas_r_half[gal == gal_indices][0]
        
        r_over_r_half = mean_r / r_half

        #MASK TO SAY HOW MANY MULTIPLES OF GAS_R_HALF YOU WANT TO GO OUT TO 

        r_mask_tree_TNG = (r_over_r_half <= max_radius) & (r_over_r_half >= min_radius)


        g_tot = g_tot[r_mask_tree_TNG]
        g_bar_individual = g_bar_individual[r_mask_tree_TNG]
        mean_r = mean_r[r_mask_tree_TNG]


        plt.scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)


    #plt.scatter(np.log10(g_bar), log_g_tot_pred_pytree, color='black', label='Pytree_fit', s=80)
    plt.plot(np.log10(g_bar), log_g_tot_pred_pytree, color='black', label='Pytree_fit', linewidth=4)


    #plt.plot(bincenters, mean, color = 'red', linewidth = 4)
    plt.scatter(bincenters, mean, color = 'red', s = 80)


    #plt.scatter(bincenters, mean, color='red', label='Mean Acceleration', alpha = 0.8, marker = 'D')
    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.xlim(-14, -8)
    plt.ylim(-14, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_tot_pred_pytree


def RAR_func_MCMC(gbar, gtot):

    log_gbar = jnp.log10(gbar)
    log_gtot = jnp.log10(gtot)


    def model(x=None, y=None):

        a0 = numpyro.sample("a0", dist.Uniform(0, 5))
        sigma = numpyro.sample("sigma", dist.Uniform(0, 5))
        shape = 1
        
        # Prediction of y
        log_gtot_pred = jnp.log10(gbar*(1-jnp.exp(-(gbar/(a0*1e-10))**(shape/2)))**(-1/shape))  

        # Gaussian likelihood
        numpyro.sample("obs", dist.Normal(log_gtot_pred, sigma), obs=log_gtot)

    # Run MCMC
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=2000)
    mcmc.run(rng_key_, x=gbar, y=log_gtot)

    mcmc.print_summary()
    samples = mcmc.get_samples()
    res = az.from_numpyro(mcmc)

    # Extract posterior samples as a NumPy array
    samples_array = np.array([res.posterior[var].values.flatten() for var in res.posterior.keys()]).T


    # Corner plot
    fig = corner.corner(
        samples_array,
        labels=[r'$a_0 \times 10^{10}$', '$\sigma$'],  # Ensure correct variable names
        show_titles=True,
        smooth = 1,
        label_kwargs={"fontsize": 15}
    )
    plt.show()

    return samples_array

def Plot_pytree_RAR_func(samples_array, pytree_results, gas_r_half, gal_indices, min_radius, max_radius, bincenters, mean):


    a0 = np.median(samples_array[:, 0])
    sigma_pytree = np.median(samples_array[:, 1])

    print('a0:', a0)
    shape = 1
    print('sigma:', sigma_pytree)

    g_bar = np.logspace(-8, -14, 40)

    log_g_tot_pred_pytree = np.log10(g_bar*(1-np.exp(-(g_bar/(a0*1e-10))**(shape/2)))**(-1/shape))

    plt.figure(figsize=(10,8))

    x = np.linspace(-15, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')

    for gal in pytree_results:
        g_tot = pytree_results[gal][1]
        g_bar_individual = pytree_results[gal][0]
        mean_r = pytree_results[gal][2]

                
        r_half = gas_r_half[gal == gal_indices][0]
        
        r_over_r_half = mean_r / r_half

        #MASK TO SAY HOW MANY MULTIPLES OF GAS_R_HALF YOU WANT TO GO OUT TO 

        r_mask_tree = (r_over_r_half <= max_radius) & (r_over_r_half >= min_radius)


        g_tot = g_tot[r_mask_tree]
        g_bar_individual = g_bar_individual[r_mask_tree]
        mean_r = mean_r[r_mask_tree]


        plt.scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)


    #plt.scatter(np.log10(g_bar), log_g_tot_pred_pytree, color='black', label='Pytree_fit', s=80)
    plt.plot(np.log10(g_bar), log_g_tot_pred_pytree, color='black', label='Pytree_fit', linewidth=4)


    #plt.plot(bincenters, mean, color = 'red', linewidth = 4)
    plt.scatter(bincenters, mean, color = 'red', s = 50)



    #plt.scatter(bincenters, mean, color='red', label='Mean Acceleration', alpha = 0.8, marker = 'D')
    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.xlim(-12.5, -8)
    plt.ylim(-12.5, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_tot_pred_pytree

def RAR_func_shape_MCMC(gbar, gtot):

    log_gbar = jnp.log10(gbar)
    log_gtot = jnp.log10(gtot)


    def model(x=None, y=None):

        a0 = numpyro.sample("a0", dist.Uniform(0, 5))
        sigma = numpyro.sample("sigma", dist.Uniform(0, 5))
        shape = numpyro.sample("shape", dist.Uniform(0.1, 3))
        
        # Prediction of y
        log_gtot_pred = jnp.log10(gbar*(1-jnp.exp(-(gbar/(a0*1e-10))**(shape/2)))**(-1/shape))  

        # Gaussian likelihood
        numpyro.sample("obs", dist.Normal(log_gtot_pred, sigma), obs=log_gtot)

    # Run MCMC
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=2000)
    mcmc.run(rng_key_, x=gbar, y=log_gtot)

    mcmc.print_summary()
    samples = mcmc.get_samples()
    res = az.from_numpyro(mcmc)

    # Extract posterior samples as a NumPy array
    samples_array = np.array([res.posterior[var].values.flatten() for var in res.posterior.keys()]).T


    # Corner plot
    fig = corner.corner(
        samples_array,
        labels=[r'$a_0 \times 10^{10}$', '$\delta$', '$\sigma$'],  # Ensure correct variable names
        show_titles=True,
        smooth = 1,
        label_kwargs={"fontsize": 15}
    )
    plt.show()

    return samples_array

def Plot_pytree_RAR_shape(samples_array, pytree_results, gas_r_half, gal_indices, min_radius, max_radius, bincenters, mean):


    a0 = np.median(samples_array[:, 0])
    sigma_pytree = np.median(samples_array[:, 2])
    shape = np.median(samples_array[:, 1])

    print('a0:', a0)
    print('sigma:', sigma_pytree)
    print('shape:', shape)

    g_bar = np.logspace(-8, -14, 40)

    log_g_tot_pred_pytree = np.log10(g_bar*(1-np.exp(-(g_bar/(a0*1e-10))**(shape/2)))**(-1/shape))

    plt.figure(figsize=(10,8))

    x = np.linspace(-15, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')

    for gal in pytree_results:
        g_tot = pytree_results[gal][1]
        g_bar_individual = pytree_results[gal][0]
        mean_r = pytree_results[gal][2]

                
        r_half = gas_r_half[gal == gal_indices][0]
        
        r_over_r_half = mean_r / r_half

        #MASK TO SAY HOW MANY MULTIPLES OF GAS_R_HALF YOU WANT TO GO OUT TO 

        r_mask_tree = (r_over_r_half <= max_radius) & (r_over_r_half >= min_radius)


        g_tot = g_tot[r_mask_tree]
        g_bar_individual = g_bar_individual[r_mask_tree]
        mean_r = mean_r[r_mask_tree]


        plt.scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)


    #plt.scatter(np.log10(g_bar), log_g_tot_pred_pytree, color='black', label='Pytree_fit', s=80)
    plt.plot(np.log10(g_bar), log_g_tot_pred_pytree, color='black', label='Pytree_fit', linewidth=4)


    #plt.plot(bincenters, mean, color = 'red', linewidth = 4)
    plt.scatter(bincenters, mean, color = 'red', s = 50)



    #plt.scatter(bincenters, mean, color='red', label='Mean Acceleration', alpha = 0.8, marker = 'D')
    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.xlim(-12.5, -8)
    plt.ylim(-12.5, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_tot_pred_pytree


def get_RAR_data(Chosen_galaxies, gas_r_half, gal_indices, min_radius, max_radius, sim):


    rar = defaultdict(list)

    if sim == 'TNG':
        directory = '/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG'

    else:
        directory = '/mnt/users/darnej/MPhys/RAR_for_all'

    files = glob.glob(os.path.join(directory, '*.csv'))

    files.sort()

    g_bar_all = []
    g_rot_all = []
    g_tot_all = []
    mean_r_all = []
    g_obs_all = []


    for i in Chosen_galaxies:
        i = int(i)

        file = directory + '/results_for_' + str(i) + '.csv'
        csv = pd.read_csv(file, float_precision='round_trip')
        mean_acc_bar = csv['mean_acc_bar']
        mean_acc_obs = csv['mean_acc_obs']
        mean_r = csv['mean_r']
        total_mean_acc_bar = csv['total_mean_acc_bar']
        gal_number = int(i)


        try:
            g_obs = csv['g_obs']
        
        except KeyError:
            g_obs = csv['mean_acc_obs']
            

        r_half = gas_r_half[gal_number == gal_indices][0]
        
        r_over_r_half = mean_r / r_half

        #MASK TO SAY HOW MANY MULTIPLES OF GAS_R_HALF YOU WANT TO GO OUT TO 

        r_mask_rar_TNG = (r_over_r_half <= max_radius) & (r_over_r_half >= min_radius)

        mean_acc_bar = mean_acc_bar[r_mask_rar_TNG]
        mean_acc_obs = mean_acc_obs[r_mask_rar_TNG]
        mean_r = mean_r[r_mask_rar_TNG]
        total_mean_acc_bar = total_mean_acc_bar[r_mask_rar_TNG]
        g_obs = g_obs[r_mask_rar_TNG]

        mask = (mean_acc_bar > 0) & (g_obs > 0)

        mean_acc_bar = mean_acc_bar[mask]
        mean_acc_obs = mean_acc_obs[mask]
        mean_r = mean_r[mask]
        total_mean_acc_bar = total_mean_acc_bar[mask]
        g_obs = g_obs[mask]

        g_bar_all = g_bar_all + list(mean_acc_bar)
        g_rot_all = g_rot_all + list(mean_acc_obs)
        g_tot_all = g_tot_all + list(total_mean_acc_bar)
        mean_r_all = mean_r_all + list(mean_r)
        g_obs_all = g_obs_all + list(g_obs)



        rar[gal_number] = [mean_acc_bar, mean_acc_obs, mean_r, total_mean_acc_bar, g_obs]


    g_tot_all = np.array(g_tot_all)
    g_bar_all = np.array(g_bar_all)
    g_rot_all = np.array(g_rot_all)
    mean_r_all = np.array(mean_r_all)
    g_obs_all = np.array(g_obs_all)

    #The g bar values contains some 0's therefore we need to remove them to not get -inf when take log 10

    print(len(g_bar_all))
    print(len(g_rot_all))
    print(len(g_tot_all))
    print(len(g_obs_all))


    #Creating mean

    mean_rot, bins_rot, bin_number = stats.binned_statistic(np.log10(g_bar_all), np.log10(g_rot_all), bins=30, statistic='mean')

    bincenters_rot = 0.5*(bins_rot[1:]+bins_rot[:-1])

    mean_tot, bins_tot, bin_number = stats.binned_statistic(np.log10(g_bar_all), np.log10(g_tot_all), bins=30, statistic='mean')

    bincenters_tot = 0.5*(bins_tot[1:]+bins_tot[:-1])


    mean_rar, bins_rar, bin_number = stats.binned_statistic(np.log10(g_bar_all), np.log10(g_obs_all), bins=30, statistic='mean')

    bincenters_rar = 0.5*(bins_rar[1:]+bins_rar[:-1])

    return g_bar_all, g_rot_all, g_tot_all, g_obs_all, bincenters_rot, mean_rot, bincenters_tot, mean_tot, bincenters_rar, mean_rar, rar

def Plot_RARs(rar, bincenters_rot, mean_rot, bincenters_tot, mean_tot, bincenters_rar, mean_rar):

    fig, ax = plt.subplots(1, 3, figsize = (23,10))
    #fig.suptitle('RAR for all galaxies', fontsize = 30)

    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern Roman
    plt.rcParams["xtick.labelsize"] = 15   # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 15   # Increase y-axis tick font size




    for gal in rar:

        g_bar = rar[gal][0]
        g_rot = rar[gal][1]
        g_obs = rar[gal][4]


        ax[0].scatter(np.log10(g_bar), np.log10(g_rot), s = 1, alpha = 0.5, c = rar[gal][2])

        g_bar = rar[gal][0]
        g_tot = rar[gal][3]

        ax[1].scatter(np.log10(g_bar), np.log10(g_tot), s = 1, alpha = 0.5, c = rar[gal][2])

        ax[2].scatter(np.log10(g_bar), np.log10(g_obs), s = 1, alpha = 0.5, c = rar[gal][2])

        key = gal

    x = np.linspace(-16, -8, 100)
    y = x
    ax[0].plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.4, label = 'y=x')
    ax[1].plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.4, label = 'y=x')  
    ax[2].plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.4, label = 'y=x')


    ax[0].scatter(bincenters_rot, mean_rot, label = 'Mean $g_{rot}$', color = 'red')

    ax[0].set_xlabel('$\log_{10}({g_{bar}})$', fontsize = 20)
    ax[0].set_ylabel('$\log_{10}({g_{rot}})$', fontsize = 20)
    ax[0].set_title('$g_{rot}$ VS $g_{bar}$', fontsize = 20)
    ax[0].set_xlim(-13, -8)
    ax[0].set_ylim(-13,-8)


    ax[1].scatter(bincenters_tot, mean_tot, color = 'blue', label = 'Mean $g_{tot}$')

    ax[1].set_xlabel('$\log_{10}({g_{bar}})$', fontsize = 20)
    ax[1].set_ylabel('$\log_{10}({g_{tot}})$', fontsize = 20)
    ax[1].set_title('$g_{tot}$ VS $g_{bar}$', fontsize = 20)
    ax[1].set_xlim(-13, -8)
    ax[1].set_ylim(-13,-8)


    ax[2].scatter(bincenters_rar, mean_rar, color = 'red', label = 'Mean $g_{obs}$')

    ax[2].set_xlabel('$\log_{10}({g_{bar}})$', fontsize = 20)
    ax[2].set_ylabel('$\log_{10}({g_{obs}})$', fontsize = 20)
    ax[2].set_title('$g_{obs}$ VS $g_{bar}$', fontsize = 20)
    ax[2].set_xlim(-13, -8)
    ax[2].set_ylim(-13,-8)



    colors = rar[key][2]
    points = np.array([np.log10(rar[key][0]), np.log10(rar[key][1])]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.array(colors).min(), np.array(colors).max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(colors)


    cbar = fig.colorbar(lc, ax=ax[:], location = 'bottom', shrink = 0.6, pad = -0.44)
    cbar.set_label("Radial Distance [Kpc]", labelpad=15, fontsize = 22)


    ax[0].legend(prop={'size': 19})
    ax[1].legend(prop={'size': 19})
    ax[2].legend(prop={'size': 19})
    plt.tight_layout()
    plt.show()

def Plot_RAR_power_law(Samples_array_obs, rar, bincenters_rar, mean_rar):

    A = np.median(Samples_array_obs[:, 0])
    k = np.median(Samples_array_obs[:, 1])
    sigma_obs = np.median(Samples_array_obs[:, 2])

    print('A:', A)
    print('k', k)
    print('sigma:', sigma_obs)

    g_bar = np.logspace(-8, -14, 40)

    log_g_obs_pred = A + k * np.log10(g_bar)


    fig = plt.figure(figsize=(10,8))

    x = np.linspace(-15, -8, 100)
    y = x


    for gal in rar:
        g_obs = rar[gal][4]
        g_bar_individual = rar[gal][0]
        mean_r = rar[gal][2]

        plt.scatter(np.log10(g_bar_individual), np.log10(g_obs), alpha = 0.5, c = mean_r, s = 2)

    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')

    #plt.plot(bincenters_rar, mean_rar, color = 'red', linewidth = 4, label = 'Mean $g_{obs}$')
    plt.scatter(bincenters_rar, mean_rar, color = 'red', s = 50, label = 'Mean $g_{obs}$')

    #plt.scatter(np.log10(g_bar), log_g_obs_pred, color='black', label='Predicted MCMC Function', s = 80)
    plt.plot(np.log10(g_bar), log_g_obs_pred, color='black', label='Power law function', linewidth = 4)

    key = list(rar.keys())[0]
    colors = rar[key][2]
    points = np.array([np.log10(rar[key][0]), np.log10(rar[key][1])]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.array(colors).min(), np.array(colors).max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(colors)

    cbar = fig.colorbar(lc, ax = plt.gca())
    cbar.set_label("Radial Distance [Kpc]", rotation=270, labelpad=25, fontsize = 17)



    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{obs})$', fontsize = 20)
    plt.xlim(-13, -8)
    plt.ylim(-13, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_obs_pred

def Plot_RAR_RAR_func(Samples_array_obs, rar, bincenters_rar, mean_rar):

    a0 = np.median(Samples_array_obs[:, 0])
    shape = 1
    sigma_obs = np.median(Samples_array_obs[:, 1])

    print('a0:', a0)
    print('shape:', shape)
    print('sigma:', sigma_obs)

    g_bar = np.logspace(-8, -14, 40)

    log_g_obs_pred = np.log10(g_bar*(1-np.exp(-(g_bar/(a0*1e-10))**(shape/2)))**(-1/shape))  



    fig = plt.figure(figsize=(10,8))

    x = np.linspace(-15, -8, 100)
    y = x



    for gal in rar:
        g_obs = rar[gal][4]
        g_bar_individual = rar[gal][0]
        mean_r = rar[gal][2]

        plt.scatter(np.log10(g_bar_individual), np.log10(g_obs), alpha = 0.5, c = mean_r, s = 2)

    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')

    #plt.plot(bincenters_rar, mean_rar, color = 'red', linewidth = 4, label = 'Mean $g_{obs}$')
    plt.scatter(bincenters_rar, mean_rar, color = 'red', s = 60, label = 'Mean $g_{obs}$')

    #plt.scatter(np.log10(g_bar), log_g_obs_pred, color='black', label='Predicted MCMC Function', s = 80)
    plt.plot(np.log10(g_bar), log_g_obs_pred, color='black', label='Predicted MCMC Function', linewidth = 4)

    key = list(rar.keys())[0]
    colors = rar[key][2]
    points = np.array([np.log10(rar[key][0]), np.log10(rar[key][1])]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.array(colors).min(), np.array(colors).max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(colors)

    cbar = fig.colorbar(lc, ax = plt.gca())
    cbar.set_label("Radial Distance [Kpc]", rotation=270, labelpad=25, fontsize = 17)



    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{obs})$', fontsize = 20)
    plt.xlim(-13, -8)
    plt.ylim(-13, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_obs_pred

def Plot_RAR_RAR_shape(Samples_array_obs, rar, bincenters_rar, mean_rar):

    a0 = np.median(Samples_array_obs[:, 0])
    shape = np.median(Samples_array_obs[:, 1])
    sigma_obs = np.median(Samples_array_obs[:, 2])

    print('a0:', a0)
    print('shape:', shape)
    print('sigma:', sigma_obs)

    g_bar = np.logspace(-8, -14, 40)

    log_g_obs_pred = np.log10(g_bar*(1-np.exp(-(g_bar/(a0*1e-10))**(shape/2)))**(-1/shape))  



    fig = plt.figure(figsize=(10,8))

    x = np.linspace(-15, -8, 100)
    y = x



    for gal in rar:
        g_obs = rar[gal][4]
        g_bar_individual = rar[gal][0]
        mean_r = rar[gal][2]

        plt.scatter(np.log10(g_bar_individual), np.log10(g_obs), alpha = 0.5, c = mean_r, s = 2)

    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')

    #plt.plot(bincenters_rar, mean_rar, color = 'red', linewidth = 4, label = 'Mean $g_{obs}$')
    plt.scatter(bincenters_rar, mean_rar, color = 'red', s = 60, label = 'Mean $g_{obs}$')

    #plt.scatter(np.log10(g_bar), log_g_obs_pred, color='black', label='Predicted MCMC Function', s = 80)
    plt.plot(np.log10(g_bar), log_g_obs_pred, color='black', label='Predicted MCMC Function', linewidth = 4)

    key = list(rar.keys())[0]
    colors = rar[key][2]
    points = np.array([np.log10(rar[key][0]), np.log10(rar[key][1])]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.array(colors).min(), np.array(colors).max())

    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(colors)
    cbar = fig.colorbar(lc, ax = plt.gca())
    cbar.set_label("Radial Distance [Kpc]", rotation=270, labelpad=25, fontsize = 17)
    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{obs})$', fontsize = 20)
    plt.xlim(-13, -8)
    plt.ylim(-13, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_obs_pred

def Plot_tot_power_law(Samples_array_tot, rar, bincenters_tot, mean_tot):

    A = np.median(Samples_array_tot[:, 0])
    k = np.median(Samples_array_tot[:, 1])
    sigma_tot = np.median(Samples_array_tot[:, 2])

    print('A:', A)
    print('k:', k)
    print('sigma:', sigma_tot)

    g_bar = np.logspace(-8, -14, 40)

    log_g_tot_pred = A + k * np.log10(g_bar)

    plt.figure(figsize=(10,8))


    x = np.linspace(-15, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')



    for gal in rar:
        g_tot = rar[gal][3]
        g_bar_individual = rar[gal][0]
        mean_r = rar[gal][2]

        plt.scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)

    #plt.plot(bincenters_tot, mean_tot, color = 'blue', linewidth = 4, label = 'Mean $g_{tot}$')
    plt.scatter(bincenters_tot, mean_tot, color = 'blue', s = 50, label = 'Mean $g_{tot}$')


    #plt.scatter(np.log10(g_bar), log_g_tot_pred, color='black', label='Predicted MCMC Function', s = 80)
    plt.plot(np.log10(g_bar), log_g_tot_pred, color='black', label='Power Law Fit', linewidth = 4)


    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.xlim(-13, -8)
    plt.ylim(-13, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_tot_pred

def Plot_tot_RAR_func(Samples_array_tot, rar, bincenters_tot, mean_tot):

    a0 = np.median(Samples_array_tot[:, 0])
    shape = 1
    sigma_tot = np.median(Samples_array_tot[:, 1])

    g_bar = np.logspace(-8, -14, 40)

    log_g_tot_pred = np.log10(g_bar*(1-np.exp(-(g_bar/(a0*1e-10))**(shape/2)))**(-1/shape))  

    plt.figure(figsize=(10,8))


    x = np.linspace(-15, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')



    for gal in rar:
        g_tot = rar[gal][3]
        g_bar_individual = rar[gal][0]
        mean_r = rar[gal][2]

        plt.scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)

    #plt.plot(bincenters_tot, mean_tot, color = 'blue', linewidth = 4, label = 'Mean $g_{tot}$')
    plt.scatter(bincenters_tot, mean_tot, color = 'blue', s = 50, label = 'Mean $g_{tot}$')


    #plt.scatter(np.log10(g_bar), log_g_tot_pred, color='black', label='Predicted MCMC Function', s = 80)
    plt.plot(np.log10(g_bar), log_g_tot_pred, color='black', label='Predicted MCMC Function', linewidth = 4)


    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.xlim(-13, -8)
    plt.ylim(-13, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_tot_pred


def Plot_tot_RAR_shape(Samples_array_tot, rar, bincenters_tot, mean_tot):

    a0 = np.median(Samples_array_tot[:, 0])
    shape = np.median(Samples_array_tot[:, 1])
    sigma_tot = np.median(Samples_array_tot[:, 2])

    g_bar = np.logspace(-8, -14, 40)

    log_g_tot_pred = np.log10(g_bar*(1-np.exp(-(g_bar/(a0*1e-10))**(shape/2)))**(-1/shape))  

    plt.figure(figsize=(10,8))


    x = np.linspace(-15, -8, 100)
    y = x
    plt.plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.8, label = 'y=x')



    for gal in rar:
        g_tot = rar[gal][3]
        g_bar_individual = rar[gal][0]
        mean_r = rar[gal][2]

        plt.scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)

    #plt.plot(bincenters_tot, mean_tot, color = 'blue', linewidth = 4, label = 'Mean $g_{tot}$')
    plt.scatter(bincenters_tot, mean_tot, color = 'blue', s = 50, label = 'Mean $g_{tot}$')


    #plt.scatter(np.log10(g_bar), log_g_tot_pred, color='black', label='Predicted MCMC Function', s = 80)
    plt.plot(np.log10(g_bar), log_g_tot_pred, color='black', label='Predicted MCMC Function', linewidth = 4)


    plt.xlabel('$log_{10}(g_{bar})$', fontsize = 20)
    plt.ylabel('$log_{10}(g_{tot})$', fontsize = 20)
    plt.xlim(-13, -8)
    plt.ylim(-13, -8)
    plt.legend(prop = {'size': 15})
    plt.tight_layout()
    plt.show()

    return log_g_tot_pred

def Get_Mixed_RAR_TNG_NH(Chosen_galaxies, gas_r_half, gal_indices, max_radius, min_radius, sim):

    mixed = defaultdict(list)

    if sim == 'TNG':

        directory_rar = '/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG'
        directory_tree = '/mnt/users/darnej/MPhys/TNG-50/pytree_results'

    else:

        directory_rar = '/mnt/users/darnej/MPhys/RAR_for_all'
        directory_tree = '/mnt/users/darnej/MPhys/pytree_results'

    mean_r_all = []
    g_obs_all = []
    g_bar_all = []
    a_r_all = []
    a_tot_all = []
    g_tot_all = []


    for i in Chosen_galaxies:
        i = int(i)

        file = directory_rar + '/results_for_' + str(i) + '.csv'
        csv = pd.read_csv(file, float_precision='round_trip')
        mean_r = csv['mean_r']
        mean_acc_bar = csv['mean_acc_bar']
        total_mean_acc_bar = csv['total_mean_acc_bar']
        gal_number = int(i)

        try:
            g_obs = csv['g_obs']
        
        except KeyError:
            g_obs = csv['mean_acc_obs']

        file = directory_tree + '/pytree_results_' + str(i) + '.csv'
        gal_number = int(i)
        csv = pd.read_csv(file, float_precision='round_trip')
        a_r = np.array(csv['mean_a_r_py'])
        a_tot = np.array(csv['mean_a_r_total_py'])
        mean_r_py = np.array(csv['mean_r'])


        r_half = gas_r_half[gal_number == gal_indices][0]
        
        r_over_r_half = mean_r / r_half

        #MASK TO SAY HOW MANY MULTIPLES OF GAS_R_HALF YOU WANT TO GO OUT TO 

        r_mask_rar = (r_over_r_half <= max_radius) & (r_over_r_half >= min_radius)

        mean_r = mean_r[r_mask_rar]
        g_obs = g_obs[r_mask_rar]
        mean_acc_bar = mean_acc_bar[r_mask_rar]
        a_r = a_r[r_mask_rar]
        a_tot = a_tot[r_mask_rar]
        total_mean_acc_bar = total_mean_acc_bar[r_mask_rar]


        mask = (g_obs > 0) & (a_r > 0)

        mean_r = mean_r[mask]
        g_obs = g_obs[mask]
        mean_acc_bar = mean_acc_bar[mask]
        a_r = a_r[mask]
        a_tot = a_tot[mask]
        total_mean_acc_bar = total_mean_acc_bar[mask]


        mean_r_all = mean_r_all + list(mean_r)
        g_obs_all = g_obs_all + list(g_obs)
        g_bar_all = g_bar_all + list(mean_acc_bar)
        g_tot_all = g_tot_all + list(total_mean_acc_bar)
        a_r_all = a_r_all + list(a_r)
        a_tot_all = a_tot_all + list(a_tot)


        mixed[gal_number] = [mean_r, g_obs, mean_acc_bar, total_mean_acc_bar, a_r, a_tot]

    return mixed, np.array(g_bar_all), np.array(g_obs_all), np.array(g_tot_all), np.array(a_r_all), np.array(a_tot_all), np.array(mean_r_all)


def Plotting_mixed(mixed, g_bar_tree_plot_all, g_obs_plot_all, bincenters, mean, bincenters_tot, mean_tot, rar, log_g_tot_pred_pytree, log_g_tot_pred, log_g_mixed_pred):

    fig, ax = plt.subplots(1, 3, figsize = (20,10), sharey = True, sharex = True)

    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern Roman
    plt.rcParams["xtick.labelsize"] = 20  # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 20   # Increase y-axis tick font size

    x = np.linspace(-14, -8, 100)
    y = x

    for gal in mixed:

        g_obs = mixed[gal][1]
        g_bar_individual = mixed[gal][4]
        mean_r = mixed[gal][0]

        ax[0].scatter(np.log10(g_bar_individual), np.log10(g_obs), alpha = 0.5, c = mean_r, s = 2)

        g_bar_individual = mixed[gal][2]
        g_tot = mixed[gal][3]

        ax[2].scatter(np.log10(g_bar_individual), np.log10(g_tot), alpha = 0.5, c = mean_r, s = 2)

        g_bar_individual = mixed[gal][4]
        a_tot = mixed[gal][5]

        ax[1].scatter(np.log10(g_bar_individual), np.log10(a_tot), alpha = 0.5, c = mean_r, s = 2)

    mean_mixed, bins_mixed, bin_number = stats.binned_statistic(np.log10(g_bar_tree_plot_all), np.log10(g_obs_plot_all), bins=30, statistic='mean')

    bincenters_mixed = 0.5*(bins_mixed[1:]+bins_mixed[:-1])


    ax[0].scatter(bincenters_mixed, mean_mixed, color = 'red', alpha = 0.6, label = 'Binned Mean')
    ax[0].plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.4, label='$y=x$')
    #ax[0].plot(np.log10(a_r_all), np.log10(g_obs_all), linestyle = 'none', marker = '.', color = 'grey', alpha = 0.5, label = 'All Data Points', markersize = 2)
    ax[0].set_xlabel('$\log_{10}({g_{bar,tree}[\mathrm{ms^{-2}}]})$', fontsize = 30)
    ax[0].set_ylabel('$\log_{10}({g_{obs}[\mathrm{ms^{-2}}]})$', fontsize = 30)
    ax[0].set_xlim(-12.5, -8.5)
    ax[0].set_ylim(-12.5,-8.5)

    ax[2].scatter(bincenters_tot, mean_tot, color = 'red', alpha = 0.6)
    ax[2].plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.4)
    #ax[2].plot(np.log10(g_bar_all), np.log10(g_tot_all), linestyle = 'none', marker = '.', color = 'grey', alpha = 0.5, label = 'All Data Points', markersize = 2)
    ax[2].set_xlabel('$\log_{10}({g_{bar,sph}[\mathrm{ms^{-2}}]})$', fontsize = 30)
    ax[2].set_ylabel('$\log_{10}({g_{tot,sph}[\mathrm{ms^{-2}}]})$', fontsize = 30)
    ax[2].set_xlim(-12.5, -8.5)
    ax[2].set_ylim(-12.5,-8.5)

    ax[1].scatter(bincenters, mean, color = 'red', alpha = 0.6)
    ax[1].plot(x, y, linestyle = 'dashed', color = 'black', alpha = 0.4)
    #ax[1].plot(np.log10(a_r_all), np.log10(a_tot_all), linestyle = 'none', marker = '.', color = 'grey', alpha = 0.5, label = 'All Data Points', markersize = 2)
    ax[1].set_xlabel('$\log_{10}({g_{bar,tree}[\mathrm{ms^{-2}}]})$', fontsize = 30)
    ax[1].set_ylabel('$\log_{10}({g_{tot,tree}[\mathrm{ms^{-2}}]})$', fontsize = 30)
    ax[1].set_xlim(-12.5, -8.5)
    ax[1].set_ylim(-12.5,-8.5)

    key = list(rar.keys())[0]
    colors = rar[key][2]
    points = np.array([np.log10(rar[key][0]), np.log10(rar[key][1])]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(np.array(colors).min(), np.array(colors).max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(colors)

    cbar = fig.colorbar(lc, ax=ax[:], location = 'bottom', shrink = 0.6, pad = -0.44)
    cbar.set_label("Radial Distance [Kpc]", labelpad=15, fontsize = 26)


    g_bar = np.logspace(-8, -14, 40)

    ax[0].plot(np.log10(g_bar), log_g_mixed_pred, color='blue', linewidth = 3, alpha = 0.9, label = "MCMC Fit")

    ax[2].plot(np.log10(g_bar), log_g_tot_pred, color='blue', linewidth = 3, alpha = 0.9)

    ax[1].plot(np.log10(g_bar), log_g_tot_pred_pytree, color='blue', linewidth = 3, alpha = 0.9)

    plt.tight_layout()
    ax[0].legend(prop={'size': 25}, loc = 'lower right', frameon = False, fontsize = 30)
    plt.show()

def Plot_all_corners_pl(samples_SPARC, samples_array_mixed_TNG, samples_array_tot_TNG, samples_array_TNG, samples_array_mixed_NH, samples_array_tot_NH, samples_array_NH):


    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern Roman
    plt.rcParams["xtick.labelsize"] = 20   # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 20   # Increase y-axis tick font size

    fig = plt.figure(figsize=(26, 8))
    subfigs = fig.subfigures(1, 3)


    fig1 = corner.corner(samples_SPARC, 
                        labels=[r'$a_0 \times 10^{10} [\mathrm{ms^{-2}}]$ ', '$\sigma_{int}$'],
                        show_titles=True,
                        title_kwargs={"fontsize": 36, 'color': 'blue'}, 
                        label_kwargs={"fontsize": 32},
                        fill_contours=True,
                        color='blue',
                        plot_datapoints=False,
                        smooth = 1,
                        fig = subfigs[0])


    sample_labels = ['$\log({g_{obs}})$ - $\log({g_{bar,tree}})$',
                    '$\log({g_{tot,sph}})$ - $\log({g_{bar,sph}})$',
                    '$\log({g_{tot,tree}})$ - $\log({g_{bar,tree}})$']

    colors = ['red', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']



    fig2 = corner.corner(samples_array_mixed_TNG, labels = [r'$A \times 10^{2}$', 'k', '$\sigma$'], show_titles = True, title_kwargs={"fontsize": 30, "color": colors[0], 'fontweight': 'bold'}, label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[0], smooth = 1, fig = subfigs[1])

    fig2 = corner.corner(samples_array_tot_TNG, labels = [r'$A \times 10^{2}$', 'k', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[1], smooth = 1, fig = subfigs[1])

    fig2 = corner.corner(samples_array_TNG, labels = [r'$A \times 10^{2}$', 'k', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[2], smooth = 1, fig = subfigs[1])



    fig3 = corner.corner(samples_array_mixed_NH, labels = [r'$A \times 10^{2}$', 'k', '$\sigma$'], show_titles = True, title_kwargs={"fontsize": 30, "color": colors[0], 'fontweight': 'bold'}, label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[0], smooth = 1, fig = subfigs[2])

    fig3 = corner.corner(samples_array_tot_NH, labels = [r'$A \times 10^{2}$', 'k', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[1], smooth = 1, fig = subfigs[2])

    fig3 = corner.corner(samples_array_NH, labels = [r'$A \times 10^{2}$', 'k', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[2], smooth = 1, fig = subfigs[2])

    plt.legend(handles=[
        mlines.Line2D([], [], color = colors[i], label=sample_labels[i])
        for i in range(3)], fontsize=32, frameon=False, loc = (0,1.6)
    )

    fig1.suptitle('SPARC', fontsize = 34, y = 1.08)
    fig2.suptitle('TNG-50', fontsize = 34, y = 1.08)
    fig3.suptitle('NH', fontsize = 34, y = 1.08)

    plt.show()

def Plot_all_corners_RAR(samples_SPARC, samples_array_mixed_TNG, samples_array_tot_TNG, samples_array_TNG, samples_array_mixed_NH, samples_array_tot_NH, samples_array_NH):

    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern Roman
    plt.rcParams["xtick.labelsize"] = 20   # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 20   # Increase y-axis tick font size

    fig = plt.figure(figsize=(26, 8))
    subfigs = fig.subfigures(1, 3)


    fig1 = corner.corner(samples_SPARC, 
                        labels=[r'$a_0 \times 10^{10} [\mathrm{ms^{-2}}]$ ', '$\sigma_{int}$'],
                        show_titles=True,
                        title_kwargs={"fontsize": 36, 'color': 'blue'}, 
                        label_kwargs={"fontsize": 32},
                        fill_contours=True,
                        color='blue',
                        plot_datapoints=False,
                        smooth = 1,
                        fig = subfigs[0])


    sample_labels = ['$\log({g_{obs}})$ - $\log({g_{bar,tree}})$',
                    '$\log({g_{tot,sph}})$ - $\log({g_{bar,sph}})$',
                    '$\log({g_{tot,tree}})$ - $\log({g_{bar,tree}})$']

    colors = ['red', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']



    fig2 = corner.corner(samples_array_mixed_TNG, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\sigma$'], show_titles = True, title_kwargs={"fontsize": 36, "color": colors[0], 'fontweight': 'bold'}, label_kwargs={"fontsize": 32}, fill_contours=True, color=colors[0], smooth = 1, fig = subfigs[1])

    fig2 = corner.corner(samples_array_tot_TNG, labels = [r'$a_0 \times 10^{10}$', '$\sigma$'], label_kwargs={"fontsize": 32}, fill_contours=True, color=colors[1], smooth = 1, fig = subfigs[1])

    fig2 = corner.corner(samples_array_TNG, labels = [r'$a_0 \times 10^{10} [\mathrm{ms^{-2}}]$', '$\sigma$'], label_kwargs={"fontsize": 32}, fill_contours=True, color=colors[2], smooth = 1, fig = subfigs[1])



    fig3 = corner.corner(samples_array_mixed_NH, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\sigma$'], show_titles = True, title_kwargs={"fontsize": 36, "color": colors[0], 'fontweight': 'bold'}, label_kwargs={"fontsize": 32}, fill_contours=True, color=colors[0], smooth = 1, fig = subfigs[2])

    fig3 = corner.corner(samples_array_tot_NH, labels = [r'$a_0 \times 10^{10}$', '$\sigma$'], label_kwargs={"fontsize": 32}, fill_contours=True, color=colors[1], smooth = 1, fig = subfigs[2])

    fig3 = corner.corner(samples_array_NH, labels = [r'$a_0 \times 10^{10} [\mathrm{ms^{-2}}]$', '$\sigma$'], label_kwargs={"fontsize": 32}, fill_contours=True, color=colors[2], smooth = 1, fig = subfigs[2])

    plt.legend(handles=[
        mlines.Line2D([], [], color = colors[i], label=sample_labels[i])
        for i in range(3)], fontsize=27, frameon=False, loc = (-0.05,1.3)
    )

    fig1.suptitle('SPARC', fontsize = 34, y = 1.08)
    fig2.suptitle('TNG-50', fontsize = 34, y = 1.08)
    fig3.suptitle('NH', fontsize = 34, y = 1.08)

    plt.show()


def Plot_all_corners_shape(samples_SPARC, samples_array_mixed_TNG, samples_array_tot_TNG, samples_array_TNG, samples_array_mixed_NH, samples_array_tot_NH, samples_array_NH):

    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10'  # Computer Modern Roman
    plt.rcParams["xtick.labelsize"] = 20   # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 20   # Increase y-axis tick font size

    fig = plt.figure(figsize=(26, 8))
    subfigs = fig.subfigures(1, 3)


    fig1 = corner.corner(samples_SPARC, 
                        labels=[r'$a_0 \times 10^{10} [\mathrm{ms^{-2}}]$ ', '$\sigma_{int}$'],
                        show_titles=True,
                        title_kwargs={"fontsize": 36, 'color': 'blue'}, 
                        label_kwargs={"fontsize": 32},
                        fill_contours=True,
                        color='blue',
                        plot_datapoints=False,
                        smooth = 1,
                        fig = subfigs[0])


    sample_labels = ['$\log({g_{obs}})$ - $\log({g_{bar,tree}})$',
                    '$\log({g_{tot,sph}})$ - $\log({g_{bar,sph}})$',
                    '$\log({g_{tot,tree}})$ - $\log({g_{bar,tree}})$']

    colors = ['red', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']



    fig2 = corner.corner(samples_array_mixed_TNG, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\delta$', '$\sigma$'], show_titles = True, title_kwargs={"fontsize": 30, "color": colors[0], 'fontweight': 'bold'}, label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[0], smooth = 1, fig = subfigs[1])

    fig2 = corner.corner(samples_array_tot_TNG, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\delta$', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[1], smooth = 1, fig = subfigs[1])

    fig2 = corner.corner(samples_array_TNG, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\delta$', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[2], smooth = 1, fig = subfigs[1])



    fig3 = corner.corner(samples_array_mixed_NH, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\delta$', '$\sigma$'], show_titles = True, title_kwargs={"fontsize": 30, "color": colors[0], 'fontweight': 'bold'}, label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[0], smooth = 1, fig = subfigs[2])

    fig3 = corner.corner(samples_array_tot_NH, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\delta$', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[1], smooth = 1, fig = subfigs[2])

    fig3 = corner.corner(samples_array_NH, labels = [r'$a_0 \times 10^{10}[\mathrm{ms^{-2}}]$', '$\delta$', '$\sigma$'], label_kwargs={"fontsize": 25}, fill_contours=True, color=colors[2], smooth = 1, fig = subfigs[2])

    plt.legend(handles=[
        mlines.Line2D([], [], color = colors[i], label=sample_labels[i])
        for i in range(3)], fontsize=32, frameon=False, loc = (0,1.6)
    )

    fig1.suptitle('SPARC', fontsize = 34, y = 1.08)
    fig2.suptitle('TNG-50', fontsize = 34, y = 1.08)
    fig3.suptitle('NH', fontsize = 34, y = 1.08)

    plt.show()