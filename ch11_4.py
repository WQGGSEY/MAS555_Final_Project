import scipy.stats, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, pyreadr, os, sys, warnings, pickle, cvxpy as cp, pickle, statsmodels.api as sm
from tqdm import tqdm, trange
warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True) # Set numpy print options for better readability
np.random.seed(42) # for reproducibility

save = True # save the figure or not
img_save_path = 'Images'

def load_data()-> pd.DataFrame:
    """
    Load math standard data from a .dat file.
    county | metstandard | percentms
    """
    parsed_data_list = []
    column_names_from_file = []     
    file_path = os.path.join(os.path.dirname(__file__), 'mathstandard.dat')
    data_stream = open(file_path, 'r')
        
    # 첫 번째 줄을 읽어 헤더로 사용
    header_line = data_stream.readline().strip()
    if header_line:
        column_names_from_file = header_line.split()
    else:
        raise ValueError("파일이 비어 있거나 헤더를 읽을 수 없습니다.")

    # 나머지 데이터 줄 처리
    for line_number, line_content in enumerate(data_stream, start=2): # 헤더 다음 줄부터 시작 
        stripped_line = line_content.strip()
        if not stripped_line: # 빈 줄 건너뛰기
            continue
        parts = stripped_line.split() # 공백 기준으로 모든 부분을 나눔
        percentms_val = float(parts[-1])
        metstandard_val = int(parts[-2])
        county_name_val = " ".join(parts[:-2])
        parsed_data_list.append([county_name_val, metstandard_val, percentms_val])

    if not column_names_from_file: # 만약 위에서 헤더를 제대로 못 읽었다면 기본 컬럼명 사용
        column_names_from_file = ['county', 'metstandard', 'percentms']
        
    df = pd.DataFrame(parsed_data_list, columns=column_names_from_file)
    return df

def ad_hoc_analysis(df: pd.DataFrame):
    data_dict = {} # key: county name, value: np.array of [metstandard, percentms], y<-metstardart, x<-percentms
    beta_dict_valid = {}
    for county in df['county'].unique():
        data_dict[county] = np.array(df[df['county'] == county].reset_index(drop=True)[['metstandard', 'percentms']])
    
    # for each county, find the maximizer, beta
    beta_dict = {}
    for county in data_dict:
        data = data_dict[county]
        x_data = data[:, 1] # Use a different variable name to avoid conflict with cp.Variable x if it were named x
        y_data = data[:, 0] # Use a different variable name
        
        beta = cp.Variable(2, name=f"beta_{county.replace(' ', '_')}") # Naming variables can help in debugging
        
        # eta is the linear predictor
        gamma = beta[0] + beta[1] * x_data
        
        # Log-likelihood for logistic regression
        log_likelihood_terms = cp.multiply(y_data, gamma) - cp.logistic(gamma)
        
        objective = cp.Maximize(cp.sum(log_likelihood_terms))
        prob = cp.Problem(objective)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if beta.value is not None:
                beta_dict[county] = beta.value
                if len(data) >= 10: 
                    beta_dict_valid[county] = beta.value
            else:
                print(f"Solution not found or beta.value is None for county {county} (status: {prob.status})")
                beta_dict[county] = None
        except cp.error.SolverError as e:
            print(f"Solver error for county {county}: {e}")
            beta_dict[county] = None
        except Exception as e: # Catch other potential errors
            print(f"An unexpected error occurred for county {county}: {e}")
            beta_dict[county] = None

    # show every county in a single figure
    logistics_drawing(beta_dict, file_name = 'ch11_4b_all.png', title='All Counties')
    logistics_drawing(beta_dict_valid, file_name = 'ch11_4b_only10.png', title='counties with 10 or more schools')
    
    return beta_dict, beta_dict_valid

def line_drawing(beta_dict, file_name: str, title):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 10)) # 그림 크기는 범례 위치에 따라 조정될 수 있음
    valid_beta_items = [(county, beta) for county, beta in beta_dict.items() if beta is not None]
    plotted_counties_count = len(valid_beta_items)
    colors = sns.color_palette("rocket_r", n_colors=plotted_counties_count)
    for i, (county, beta) in enumerate(valid_beta_items):
        x_plot = np.linspace(70, 130, 200)
        y_plot = beta[0] + beta[1] * x_plot
        
        current_color = colors[i] if colors else None
        
        ax.plot(x_plot, y_plot, label=f'{county} ($\\beta_0={beta[0]:.3f}, \\beta_1={beta[1]:.4f}$)', 
                color=current_color, linewidth=2)

    ax.set_title(f'Line Plotting for {title} ({plotted_counties_count} Counties)', fontsize=18, pad=20)
    ax.set_xlabel('percentms (x)', fontsize=14, labelpad=15)
    ax.set_ylabel('beta0+beta1*x', fontsize=14, labelpad=15)
    ax.set_ylim(0, 2.5)

    legend_title = "Counties"
    if plotted_counties_count > 0:
        if plotted_counties_count > 10: # 10개를 초과하면 (예: 12개) 범례를 그래프 밖에 표시
            # bbox_to_anchor의 x값을 약간 늘려 그래프와의 간격을 확보할 수 있습니다.
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title=legend_title, frameon=True, shadow=False)
        elif plotted_counties_count > 5: # 6-10개 항목은 내부에 'small' 폰트로
            ax.legend(fontsize='small', title=legend_title, frameon=True, shadow=False)
        else: # 1-5개 항목은 내부에 'medium' 폰트로
            ax.legend(fontsize='medium', title=legend_title, frameon=True, shadow=False)

    if plotted_counties_count > 10 and plotted_counties_count > 0 :
        fig.subplots_adjust(right=0.78) # 이 값은 범례의 너비에 따라 조절 필요
    else:
        fig.tight_layout()
    if save: # 'save' 변수가 외부에서 정의되어 있다고 가정
        save_path = os.path.join(os.path.dirname(__file__), img_save_path, file_name)
        plt.savefig(save_path, dpi=300)
    plt.close(fig) # 메모리 해제를 위해 현재 그림 닫기

def logistics_drawing(beta_dict, file_name: str, title):
     # show every county in a single figure
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 10)) # 그림 크기는 범례 위치에 따라 조정될 수 있음
    valid_beta_items = [(county, beta) for county, beta in beta_dict.items() if beta is not None]
    plotted_counties_count = len(valid_beta_items)
    colors = sns.color_palette("rocket_r", n_colors=plotted_counties_count)
    for i, (county, beta) in enumerate(valid_beta_items):
        x_plot = np.linspace(0, 100, 200)
        y_plot = scipy.special.expit(beta[0] + beta[1] * x_plot)
        
        current_color = colors[i] if colors else None
        
        ax.plot(x_plot, y_plot, label=f'{county} ($\\beta_0={beta[0]:.3f}, \\beta_1={beta[1]:.4f}$)', 
                color=current_color, linewidth=2)

    ax.set_title(f'Logistic Plotting for {title} ({plotted_counties_count} Counties)', fontsize=18, pad=20)
    ax.set_xlabel('percentms (x)', fontsize=14, labelpad=15)
    ax.set_ylabel('P(metstandard = 1)', fontsize=14, labelpad=15)
    ax.set_ylim(-0.05, 1.05)

    legend_title = "Counties"
    if plotted_counties_count > 0:
        if plotted_counties_count > 10: # 10개를 초과하면 (예: 12개) 범례를 그래프 밖에 표시
            # bbox_to_anchor의 x값을 약간 늘려 그래프와의 간격을 확보할 수 있습니다.
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title=legend_title, frameon=True, shadow=False)
        elif plotted_counties_count > 5: # 6-10개 항목은 내부에 'small' 폰트로
            ax.legend(fontsize='small', title=legend_title, frameon=True, shadow=False)
        else: # 1-5개 항목은 내부에 'medium' 폰트로
            ax.legend(fontsize='medium', title=legend_title, frameon=True, shadow=False)

    if plotted_counties_count > 10 and plotted_counties_count > 0 :
        fig.subplots_adjust(right=0.78) # 이 값은 범례의 너비에 따라 조절 필요
    else:
        fig.tight_layout()
    if save: # 'save' 변수가 외부에서 정의되어 있다고 가정
        save_path = os.path.join(os.path.dirname(__file__), img_save_path, file_name)
        plt.savefig(save_path, dpi=300)
    plt.close(fig) # 메모리 해제를 위해 현재 그림 닫기

def estimate_parameters(data:pd.DataFrame, theta_est: np.ndarray, sigma_est: np.ndarray, m:int, load=False, iter=50_000):
    """
    Perform Metropolis-Hastings sampling to estimate the posterior distribution of beta.
    step1) generate sample theta ~ MVN(theta_m, sigma_m)
    step2) generate sample Sigma_inv ~ InvWishart(4+m, (sigma_est+sigma_theta)^{-1})
    step3) gene
    """
    if load and os.path.exists(os.path.join(os.path.dirname(__file__), 'ch11_4_samples.pkl')):
        with open(os.path.join(os.path.dirname(__file__), 'ch11_4_samples.pkl'), 'rb') as f:
            sample_dict = pickle.load(f)
        return sample_dict

    data_dict = {}
    for j, county in enumerate(data['county'].unique()):
        data_dict[j] = np.array(data[data['county'] == county].reset_index(drop=True)[['metstandard', 'percentms']])

    betas = scipy.stats.multivariate_normal.rvs(mean=theta_est, cov=sigma_est, size=m) # Initial sample of betas
    sigma = scipy.stats.invwishart.rvs(df=4, scale=sigma_est) # Initial sample of Sigma_inv, Note that the scipy.stats.invwishart scale argument is not the inverse one.
    theta_samples = []
    sigma_samples = []
    betas_samples = []
    accept_count_list = np.array([0 for j in range(m)]) # Count of accepted samples for each beta

    for i in trange(iter, desc="Sampling"):
        # Step 1: Sample theta from MVN
        sigma_m = np.linalg.inv(np.linalg.inv(sigma_est) + m * np.linalg.inv(sigma)) # sigma_m = (sigma_est^{-1} + m * sigma^{-1})^{-1}
        mu_m = sigma_m @ (np.linalg.inv(sigma_est) @ theta_est + m * np.linalg.inv(sigma) @ np.mean(betas, axis=0)) # mu_m = sigma_m @ (sigma_est^{-1} @ theta_est + m * sigma^{-1} @ mean(betas))
        theta = scipy.stats.multivariate_normal.rvs(mean=mu_m, cov=sigma_m)
        # Step 2: Sample Sigma_inv from InvWishart
        sigma_theta = np.array([np.outer(betas[j] - theta, betas[j] - theta) for j in range(m)]).sum(axis=0)
        scale_matrix = sigma_est + sigma_theta
        sigma = scipy.stats.invwishart.rvs(df=4+m, scale=scale_matrix)
        # Step 3: Sample betas from MVN => use metropolis-hastings sampling
        for j in range(m):
            proposed_beta_j = scipy.stats.multivariate_normal.rvs(mean=betas[j], cov=2*sigma) # Proposed sample for beta_j
            current_beta_j = betas[j]

            # Calculate acceptance ratio
            y_j = data_dict[j][:,0]
            x_j = data_dict[j][:,1]
            proposed_log_likelihood_term = np.sum(y_j * (proposed_beta_j[0] + proposed_beta_j[1] * x_j) - np.log(1 + np.exp(proposed_beta_j[0] + proposed_beta_j[1] * x_j)))
            current_log_likelihood_term = np.sum(y_j * (current_beta_j[0] + current_beta_j[1] * x_j) - np.log(1 + np.exp(current_beta_j[0] + current_beta_j[1] * x_j)))
            log_likelihood_term = proposed_log_likelihood_term - current_log_likelihood_term
            log_beta_term = scipy.stats.multivariate_normal.logpdf(proposed_beta_j, mean=theta, cov=sigma) - scipy.stats.multivariate_normal.logpdf(current_beta_j, mean=theta, cov=sigma)

            log_J_term = 0 # if we use symmetric proposal distribution, this term is 0
            acceptance_ratio = np.exp(log_likelihood_term + log_beta_term + log_J_term)
            accept = np.random.rand() < acceptance_ratio
            if accept:
                betas[j] = proposed_beta_j
                accept_count_list[j] += 1
            else:
                betas[j] = current_beta_j

        if i % 1000 == 0:
            print("acceptance rates so far:", accept_count_list / (i + 1))
            print("overall acceptance rate:", np.mean(accept_count_list / (i + 1)))
        # Store the samples
        theta_samples.append(theta.copy())
        sigma_samples.append(sigma.copy())
        betas_samples.append(betas.copy()) # Store a copy of the current betas
    
    acceptance_rates = accept_count_list / iter
    # Save the samples to a dictionary
    sample_dict = {
        'theta': np.array(theta_samples),
        'sigma': np.array(sigma_samples),
        'betas': np.array(betas_samples),
        'acceptance_rates': acceptance_rates,
    }  
    with open(os.path.join(os.path.dirname(__file__), 'ch11_4_samples.pkl'), 'wb') as f:
        pickle.dump(sample_dict, f)
    return sample_dict

def plot_variables(sample_dict: dict):
    """
    Plot the samples of theta and sigma.: Total 5 variables. vs iteration
    """
    # Seaborn 스타일 적용 (선택 사항, 전역적으로 이미 설정되어 있을 수 있음)
    # 이전 대화에서 seaborn 스타일을 선호하셨으므로, 여기서도 일관성을 위해 적용하거나
    # 이미 전역적으로 설정된 스타일을 따르도록 할 수 있습니다.
    # sns.set_theme(style="whitegrid") # 필요하다면 여기서 특정 스타일 지정

    theta_samples = sample_dict['theta']
    Sigma_samples = sample_dict['sigma']
    
    num_samples = theta_samples.shape[0]  # Number of samples

    iterations = np.arange(1, num_samples + 1)

    params_to_plot = [
        ("theta_0", theta_samples[:, 0], r"$\theta_0$"),
        ("theta_1", theta_samples[:, 1], r"$\theta_1$"),
        ("Sigma_00", Sigma_samples[:, 0, 0], r"$\Sigma_{00}$ (Var($\beta_0$))"),
        ("Sigma_11", Sigma_samples[:, 1, 1], r"$\Sigma_{11}$ (Var($\beta_1$))"),
        ("Sigma_01", Sigma_samples[:, 0, 1], r"$\Sigma_{01}$ (Cov($\beta_0, \beta_1$))")
    ]


    # 4. 각 파라미터에 대해 플롯 생성 및 저장
    for i, (param_key_name, param_series, param_latex_label) in enumerate(params_to_plot):
        fig, ax = plt.subplots(figsize=(12, 5)) # 플롯 크기
        
        ax.plot(iterations, param_series, linewidth=1.2, alpha=0.8)
        
        ax.set_title(f"MCMC Trace Plot: {param_latex_label}", fontsize=15)
        ax.set_xlabel(f"Iteration : {num_samples}", fontsize=12)
        ax.set_ylabel("Sampled Value", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6) # Seaborn 스타일에 따라 격자 자동 적용될 수 있음
        ax.set_ylim(param_series.min() - 0.1 * (param_series.max() - param_series.min()), 
                   param_series.max() + 0.1 * (param_series.max() - param_series.min()))
        
        # y축 범위 자동 조절 또는 필요시 명시적 설정
        # padding = (param_series.max() - param_series.min()) * 0.1
        # ax.set_ylim(param_series.min() - padding, param_series.max() + padding)

        plt.tight_layout() # 레이아웃 최적화
        
        # 파일명 설정: "ch11_4d{숫자}.png" (숫자는 1부터 5까지)
        file_name = f"ch11_4d_{i+1}.png"
        full_save_path = os.path.join(os.path.dirname(__file__), img_save_path, file_name)
        if save:
            plt.savefig(full_save_path, dpi=300)
            
        plt.close(fig) # 메모리 해제를 위해 현재 그림 닫기)

def calculate_ess(series, max_lag_prop=0.5):
    n = len(series)
    max_lag = min(n - 1, int(n * max_lag_prop))

    try:
        acf_vals = sm.tsa.stattools.acf(series, nlags=max_lag, fft=(n > 1000), alpha=None)
        if np.isnan(acf_vals).any(): # ACF 결과에 NaN이 포함된 경우
            return float(n)
    except Exception as e:
        return float(n) # ACF 계산 실패 시 N으로 대체
    current_sum_of_pairs = 0.0
    for k_odd_lag in range(1, max_lag, 2): # k_odd_lag = 1, 3, 5, ...
        if k_odd_lag + 1 >= len(acf_vals): # 배열 범위를 벗어나지 않도록 확인
            break
        
        rho_pair_sum = acf_vals[k_odd_lag] + acf_vals[k_odd_lag + 1]
        
        if rho_pair_sum > 0:
            current_sum_of_pairs += rho_pair_sum
        else:
            break
            
    denominator = 1 + 2 * current_sum_of_pairs

    if denominator <= 0:
        rho_sum_simple = 0.0
        for k_simple in range(1, len(acf_vals)): # lag 1부터 합산
            if acf_vals[k_simple] > 0:
                rho_sum_simple += acf_vals[k_simple]
            else:
                break # 첫 번째 음수 ACF에서 중단
        denominator = 1 + 2 * rho_sum_simple
        if denominator <= 0:
             # 여전히 문제가 있다면, ESS는 매우 낮을 수 있음 (최악의 경우 1)
             return 1.0

    ess = n / denominator
    return max(1.0, min(ess, float(n)))


def main():
    data = load_data()
    m = len(data['county'].unique())
    ###################### Problem (b) ######################
    print(f"Number of counties: {m}")
    ad_hoc_beta_dict, ad_hoc_beta_dict_valid = ad_hoc_analysis(data)
    beta_array = np.array([ad_hoc_beta_dict_valid[county] for county in ad_hoc_beta_dict_valid])
    theta_est = np.mean(beta_array, axis=0)
    sigma_est = np.cov(beta_array.T, ddof=1)  # ddof=1 for sample covariance
    print(f"ad hoc theta: {theta_est}")
    print(f"ad hoc sigma: \n{sigma_est}")
    ###################### Problem (c) ######################
    sample_dict = estimate_parameters(data, theta_est, sigma_est, m, load=True, iter=50_000)
    print(f"Overall Acceptance Rate: {np.mean(sample_dict['acceptance_rates'])}")
    ###################### Problem (d) ######################
    plot_variables(sample_dict)
    print("Effective Sample Sizes (ESS):")
    print("="*50)
    ess_dict = {}
    for i, (param_key_name, param_series, param_latex_label) in enumerate([
        ("theta_0", sample_dict['theta'][:, 0], r"$\theta_0$"),
        ("theta_1", sample_dict['theta'][:, 1], r"$\theta_1$"),
        ("Sigma_00", sample_dict['sigma'][:, 0, 0], r"$\Sigma_{00}$ (Var($\beta_0$))"),
        ("Sigma_11", sample_dict['sigma'][:, 1, 1], r"$\Sigma_{11}$ (Var($\beta_1$))"),
        ("Sigma_01", sample_dict['sigma'][:, 0, 1], r"$\Sigma_{01}$ (Cov($\beta_0, \beta_1$))")
    ]):
        ess = calculate_ess(param_series)
        ess_dict[param_key_name] = ess
        print(f"{param_latex_label} ESS: {ess:.2f}")
    ####################### Problem (e) ######################
    # find posterior beta_j for each j
    # use only 10th samples after 10,000 iterations
    betas_samples = sample_dict['betas']
    betas_samples = betas_samples[10000::10]  # Skip the first 10,000 samples and take every 10th sample
    betas_mean = np.mean(betas_samples, axis=0)
    beta_dict = {data['county'].unique()[j]: betas_mean[j] for j in range(m)}
    ad_hoc_beta_dict_valid = {county: beta_dict[county] for county in beta_dict if county in ad_hoc_beta_dict_valid}
    logistics_drawing(beta_dict, file_name='ch11_4e_all.png', title='All Counties')
    logistics_drawing(ad_hoc_beta_dict_valid, file_name='ch11_4e_only10.png', title='counties with 10 or more schools')

    line_drawing(beta_dict, file_name='ch11_4e_line_bayesian.png', title='Bayesian Parameters')
    line_drawing(ad_hoc_beta_dict, file_name='ch11_4e_line_ad_hoc.png', title='Ad-Hoc Parameters')

    ####################### Problem (f) ######################
    # We will use the posterior theta and sigma 
    # use only 10th samples after 10,000 iterations
    theta_posterior_samples = sample_dict['theta'][10000::10]
    sigma_posterior_samples = sample_dict['sigma'][10000::10]
    nu0_prior_sigma = 4 
    # Number of samples for empirical prior of Sigma_01
    N_prior_sigma_samples = 10000 
    # Generate prior samples for Sigma once, to get Sigma_01 prior samples
    # Prior for Sigma is Inverse-Wishart(df=nu0_prior_sigma, scale=sigma_est)
    if sigma_est is not None and np.all(np.linalg.eigvals(sigma_est) > 1e-9): # Ensure sigma_est is positive definite
        try:
            prior_sigma_dist_samples = scipy.stats.invwishart.rvs(df=nu0_prior_sigma, 
                                                                  scale=sigma_est, 
                                                                  size=N_prior_sigma_samples)
            prior_sigma01_samples_for_kde = prior_sigma_dist_samples[:, 0, 1]
        except np.linalg.LinAlgError:
            print("Warning: sigma_est might not be positive definite for sampling from Inverse-Wishart prior. Skipping Sigma_01 prior KDE.")
            prior_sigma01_samples_for_kde = None
    else:
        print("Warning: sigma_est is None or not positive definite. Skipping Sigma_01 prior KDE.")
        prior_sigma01_samples_for_kde = None


    params_for_f_plots = [
        {
            "name": "theta_0", "posterior": theta_posterior_samples[:, 0], "label": r"$\theta_0$",
            "prior_type": "norm", "prior_mean": theta_est[0], "prior_std": np.sqrt(sigma_est[0,0]) if sigma_est[0,0]>0 else None,
            "adhoc_est": theta_est[0]
        },
        {
            "name": "theta_1", "posterior": theta_posterior_samples[:, 1], "label": r"$\theta_1$",
            "prior_type": "norm", "prior_mean": theta_est[1], "prior_std": np.sqrt(sigma_est[1,1]) if sigma_est[1,1]>0 else None,
            "adhoc_est": theta_est[1]
        },
        {
            "name": "Sigma_00", "posterior": sigma_posterior_samples[:, 0, 0], "label": r"$\Sigma_{00}$ (Var($\beta_0$))",
            "prior_type": "invgamma", "prior_a": (nu0_prior_sigma - 2 + 1) / 2.0, # (nu0 - p + 1)/2 where p=2
            "prior_scale": sigma_est[0,0] / 2.0 if sigma_est[0,0]>0 else None,
            "adhoc_est": sigma_est[0,0]
        },
        {
            "name": "Sigma_11", "posterior": sigma_posterior_samples[:, 1, 1], "label": r"$\Sigma_{11}$ (Var($\beta_1$))",
            "prior_type": "invgamma", "prior_a": (nu0_prior_sigma - 2 + 1) / 2.0,
            "prior_scale": sigma_est[1,1] / 2.0 if sigma_est[1,1]>0 else None,
            "adhoc_est": sigma_est[1,1]
        },
        {
            "name": "Sigma_01", "posterior": sigma_posterior_samples[:, 0, 1], "label": r"$\Sigma_{01}$ (Cov($\beta_0, \beta_1$))",
            "prior_type": "kde_empirical", "prior_samples": prior_sigma01_samples_for_kde,
            "adhoc_est": sigma_est[0,1]
        }
    ]

    for i, param_info in enumerate(params_for_f_plots):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Posterior
        sns.histplot(param_info["posterior"], bins=50, kde=True, ax=ax, stat="density", label="Posterior Density (KDE)", alpha=0.7)
        
        # Determine x-range for prior plotting based on posterior plot
        xmin, xmax = ax.get_xlim()
        # Extend range slightly for better visualization of prior if it's wider
        x_plot_range_extension = (xmax - xmin) * 0.1 
        x_prior_plot = np.linspace(xmin - x_plot_range_extension, xmax + x_plot_range_extension, 300)

        # Plot Prior
        prior_plotted = False
        if param_info["prior_type"] == "norm":
            if param_info["prior_std"] is not None and param_info["prior_std"] > 1e-9 : # Check for valid std dev
                y_prior_pdf = scipy.stats.norm.pdf(x_prior_plot, loc=param_info["prior_mean"], scale=param_info["prior_std"])
                ax.plot(x_prior_plot, y_prior_pdf, color='red', linestyle='--', linewidth=2, label='Prior Density (Normal)')
                prior_plotted = True
        elif param_info["prior_type"] == "invgamma":
            if param_info["prior_scale"] is not None and param_info["prior_scale"] > 1e-9 and param_info["prior_a"] > 0:
                # Ensure x_prior_plot is positive for invgamma
                x_prior_positive = x_prior_plot[x_prior_plot > 1e-9] 
                if len(x_prior_positive) > 0:
                    y_prior_pdf = scipy.stats.invgamma.pdf(x_prior_positive, a=param_info["prior_a"], scale=param_info["prior_scale"])
                    ax.plot(x_prior_positive, y_prior_pdf, color='red', linestyle='--', linewidth=2, label='Prior Density (InvGamma)')
                    prior_plotted = True
        elif param_info["prior_type"] == "kde_empirical":
            if param_info["prior_samples"] is not None:
                sns.kdeplot(param_info["prior_samples"], color='red', linestyle='--', linewidth=2, label='Prior Density (KDE from IW)', ax=ax)
                prior_plotted = True
        
        if not prior_plotted:
            print(f"Warning: Prior for {param_info['name']} could not be plotted due to invalid parameters or data.")

        # Plot Ad-hoc Estimate
        ax.axvline(param_info["adhoc_est"], color='green', linestyle=':', linewidth=2.5, label=f'Ad-hoc Est: {param_info["adhoc_est"]:.3f}')
        
        ax.set_title(f"Marginal Posterior & Prior: {param_info['label']}", fontsize=16)
        ax.set_xlabel("Value", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
        
        plt.tight_layout()
        
        file_name = f"ch11_4f_{i+1}.png" # Add param name to file for clarity
        # Example: ch11_4f_1_theta_0.png
        if img_save_path and not os.path.exists(os.path.join(os.path.dirname(__file__), img_save_path)):
            os.makedirs(os.path.join(os.path.dirname(__file__), img_save_path), exist_ok=True)

        full_save_path = os.path.join(os.path.dirname(__file__), img_save_path, file_name)
        if save:
            plt.savefig(full_save_path, dpi=300)
            
        plt.close(fig)
    # 95% Credible Intervals for each parameter
    print("="*80)
    print("\n95% Credible Intervals for each parameter:")
    for param_info in params_for_f_plots:
        posterior_samples = param_info["posterior"]
        lower_bound = np.percentile(posterior_samples, 2.5)
        upper_bound = np.percentile(posterior_samples, 97.5)
        print(f"{param_info['label']} Mean: {np.mean(posterior_samples):.4f}, Variance: {np.var(posterior_samples):.4f},",sep=' ')
        print(f"95% Credible Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
        # Printing the expected value (mean) for each parameter
        expected_value = np.mean(posterior_samples)
    print("="*80)
    print("Add-hoc beta dictionary:")
    print(ad_hoc_beta_dict)
    print("=" * 80)
    print("Posterior beta dictionary:")
    print(beta_dict)



if __name__ == "__main__":
    main()