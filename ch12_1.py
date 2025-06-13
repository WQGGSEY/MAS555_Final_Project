import scipy.stats, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, pyreadr, os, sys, warnings, pickle, cvxpy as cp, pickle, statsmodels.api as sm
from tqdm import tqdm, trange
from statsmodels.graphics.tsaplots import plot_acf
warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True) # Set numpy print options for better readability
np.random.seed(42) # for reproducibility

save = True
img_save_path = 'images'

def load_data():
    """
    Load the dataset from a specified path.
    female vocab female.vocab prayer
    """
    data_path = os.path.join(os.path.dirname(__file__), 'prayer.dat')
    data = np.loadtxt(data_path, skiprows=1)
    X = data[:, :-1] # 946 x 3
    y = data[:, -1] # 946 x 1
    return X, y

def mh_sampling(X:np.ndarray, y:np.ndarray, iter=10_000, load=True):
    """
    Perform Metropolis-Hastings sampling.
    X: feature matrix
    y: target variable
    iter: number of iterations
    load: whether to load existing samples
    """
    if load and os.path.exists(os.path.join(os.path.dirname(__file__), 'ch12_1_samples.pkl')):
        with open(os.path.join(os.path.dirname(__file__), 'ch12_1_samples.pkl'), 'rb') as f:
            sample_dict = pickle.load(f)
        return sample_dict
    n = X.shape[0]
    z = y  # initializing z with y for compatibility, note that y in R(y)
    beta_samples = []
    z_samples = []
    for _ in trange(iter, desc='Sampling'):
        # Sample beta from the posterior distribution
        post_beta_mean = n / (n + 1) * np.linalg.inv(X.T @ X) @ X.T @ z
        post_beta_cov = n / (n + 1) * np.linalg.inv(X.T @ X)
        beta = scipy.stats.multivariate_normal.rvs(mean=post_beta_mean, cov=post_beta_cov)
        z_mean = X @ beta
        for i in range(n):
            # Sample z_i from the conditional distribution
            # a_i = max{z_j: y_j < y_i}
            a_i = np.max(z[y < y[i]]) if np.any(y < y[i]) else -float('inf')
            # b_i = min{z_j: y_j > y_i}
            b_i = np.min(z[y > y[i]]) if np.any(y > y[i]) else float('inf')
            psi_a_i = scipy.stats.norm.cdf(a_i, loc=z_mean[i], scale=1)
            psi_b_i = scipy.stats.norm.cdf(b_i, loc=z_mean[i], scale=1)
            u = np.random.uniform(psi_a_i, psi_b_i)
            z[i] = z_mean[i] + scipy.stats.norm.ppf(u, loc=0, scale=1) # Recover z_i
        # Store samples
        beta_samples.append(beta.copy())
        z_samples.append(z.copy())

    sample_dict = {
        'beta': np.array(beta_samples),
        'z': np.array(z_samples)
    }

    with open('ch12_1_samples.pkl', 'wb') as f:
        pickle.dump(sample_dict, f)
    return sample_dict

def data_analysis(X:np.ndarray, y:np.ndarray):
    """
    Perform data analysis on the dataset.
    X: feature matrix, n by 4 matrix [1, x1, x2, x3]
    y: target variable, n by 1 vector
    """
    df = pd.DataFrame({
    'x1_sex': X[:, 1].astype(int), # 정수형으로 명확히 하고, 범례에서 더 잘 보이도록 함
    'x2_vocab': X[:, 2],
    'x3_other': X[:, 3], # 사용되진 않지만, 원래 데이터 구조를 보여주기 위함
    'y_prayer': y
    })  
    # plot y vs x2
    # plot difference colors by x1, not considering x3, not use colorbar, just legend it
    # x1 = 0 or 1 by sex
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    plt.figure(figsize=(12, 7)) # figsize 조정

    # Seaborn의 scatterplot 사용
    # hue에 따라 색상이 구분되고, 범례가 자동으로 생성됨
    # palette를 명시적으로 지정하여 0과 1에 대한 색상을 정의할 수 있음
    # 예: palette={0: "blue", 1: "red"} 또는 seaborn 팔레트 사용
    # 여기서는 set_theme에서 "muted"를 사용했으므로, seaborn이 알아서 할당
    # 명시적으로 두 가지 색상을 원하면 palette=['#1f77b4', '#ff7f0e'] 등으로 지정 가능 (muted의 첫 두 색상)
    scatter_plot = sns.scatterplot(
        data=df,
        x='x2_vocab',
        y='y_prayer',
        hue='x1_sex', # x1 (sex) 값에 따라 색상 구분
        palette={0: '#1f77b4', 1: '#ff7f0e'}, # muted 팔레트의 첫 두 색상을 명시적으로 사용하거나 다른 색상 지정
        alpha=0.7,
        edgecolors='w', # 점 테두리 색상
        s=60,          # 점 크기
        legend='full'  # 범례를 항상 표시 (기본값이지만 명시)
    )

    plt.title('Prayer Count vs. Vocabulary Score by Sex', fontsize=18, fontweight='bold')
    plt.xlabel('x2 (Vocabulary Score)', fontsize=14)
    plt.ylabel('y (Prayer Count)', fontsize=14)

    # 범례 제목 수정 (선택 사항)
    handles, labels = scatter_plot.get_legend_handles_labels()
    # x1_sex의 값이 0, 1이므로, 레이블을 더 명확하게 변경할 수 있음
    # 예시: labels=['Sex: 0', 'Sex: 1'] 또는 ['Male', 'Female'] (데이터의 실제 의미에 따라)
    # 여기서는 x1_sex의 원본 값을 그대로 사용하므로 labels는 '0', '1'이 됨
    # 좀 더 명확한 레이블을 원한다면:
    new_labels = [f'Sex: {label}' for label in labels] # 또는 { '0': 'Sex A', '1': 'Sex B'}.get(label, label)
    plt.legend(handles=handles, labels=new_labels, title='x1 (Sex)', fontsize=12, title_fontsize=13)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7) # 그리드 스타일 미세 조정

    if save:
        save_filename = f'{img_save_path}/ch12_1_scatter_plot_by_sex.png'
        plt.savefig(save_filename, dpi=300, bbox_inches='tight') # bbox_inches='tight'로 잘림 방지

def plot_beta_acf(beta_samples: np.ndarray,
                  lags_to_show: int = 50):
    num_params = beta_samples.shape[1]
    if num_params == 0:
        print("ACF를 플롯할 Beta 파라미터가 없습니다.")
        return

    # 서브플롯 레이아웃 결정 (예: 2x2, 파라미터 개수에 따라 유동적으로)
    ncols = 2
    nrows = (num_params + ncols - 1) // ncols # 필요한 행의 수 계산

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten() # 다차원 axes 배열을 1차원으로 만듦

    fig.suptitle(f'Beta Coefficients ACF (Lags: {lags_to_show})', fontsize=16, fontweight='bold')

    for i in range(num_params):
        ax = axes[i]
        plot_acf(beta_samples[:, i], ax=ax, lags=lags_to_show,
                                      title=f'ACF of Beta {i}')
        ax.set_xlabel("Lag", fontsize=12)
        ax.set_ylabel("ACF", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    # 사용되지 않은 서브플롯 숨기기
    for j in range(num_params, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle과의 간격 조정

    if save:
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        save_filename = os.path.join(img_save_path, 'ch12_1_beta_acf_plots.png')
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Beta ACF plots saved to {save_filename}")

def main():
    X, y = load_data()
    X = sm.add_constant(X)  # Add intercept term
    sample_dict = mh_sampling(X, y, iter=100_000, load=True)
    data_analysis(X, y)
    # check the convergence of the beta
    # Plot the trace of beta samples
    sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1) # font_scale로 전체 폰트 크기 조절 가능

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10)) # figsize 조정으로 더 보기 좋게
    axes = axes.flatten() 

    fig.suptitle('Trace Plots of Beta Coefficients (MCMC Samples)', fontsize=18, fontweight='bold')

    num_params = sample_dict['beta'].shape[1]

    for i in range(num_params):
        if i < len(axes): # 생성된 subplot 개수 내에서만 그리도록 방어 코드 추가
            ax = axes[i]
            ax.plot(sample_dict['beta'][:, i], label=f'Beta {i}', alpha=0.8, linewidth=1.5)
            ax.set_title(f'Trace of Beta {i}', fontsize=14)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel(f'Value of Beta {i}', fontsize=12)
            ax.legend(fontsize=10)
            ax.tick_params(labelsize=10) # 눈금 레이블 크기 조절
        else:
            print(f"Warning: Beta {i} will not be plotted as there are not enough subplots.")

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # suptitle과의 간격 조정을 위해 rect 사용

    save_filename = f'{img_save_path}/ch12_1_trace_plots.png'
    plt.savefig(save_filename, dpi=300)
    ######################### TRACE ACF PLOT ########################
    # Use only after 40,000 iterations with 10th sample
    beta_samples = sample_dict['beta'][40000::10, :]  # Take every 10th sample after 40,000 iterations
    plot_beta_acf(beta_samples, lags_to_show=50)
    ######################## DRAW MGN PLOT ########################

    # draw kde and its prior in each figure
    # prior of beta ~ MVN(0, n(X^T X)^-1), beta in R^4
    beta0 = beta_samples[:, 0]
    beta1 = beta_samples[:, 1]
    beta2 = beta_samples[:, 2]
    beta3 = beta_samples[:, 3]
    for i, beta in enumerate([beta0, beta1, beta2, beta3]):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(beta, label='Posterior', color='blue', fill=True, alpha=0.5)
        # Prior distribution
        n = X.shape[0]
        prior_mean = 0
        prior_cov = n * np.linalg.inv(X.T @ X)
        # to marignalize it
        prior_cov = prior_cov[i, i]
        prior_dist = scipy.stats.norm(loc=prior_mean, scale=np.sqrt(prior_cov))
        x = np.linspace(-3, 5, 1000)
        prior_pdf = prior_dist.pdf(x)
        plt.plot(x, prior_pdf, label='Prior', color='red', linestyle='--')
        
        plt.title(f'Posterior and Prior of Beta {i}', fontsize=16)
        plt.xlabel(f'Beta {i} Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        save_filename = f'{img_save_path}/ch12_1_beta_{i}_plots.png'
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    # print confidence intervals for each beta with the same sample settings
    print("#"*80)
    for i in range(num_params):
        beta_i = sample_dict['beta'][40000::10, i]
        print(f'Beta {i} Mean: {np.mean(beta_i):.4f}, Std: {np.std(beta_i):.4f}', end=' ')
        ci_lower = np.percentile(beta_i, 2.5)
        ci_upper = np.percentile(beta_i, 97.5)
        print(f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    print("#"*80)
    
    
    

if __name__ == "__main__":
    main()



