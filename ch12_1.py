import scipy.stats, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, pyreadr, os, sys, warnings, pickle, cvxpy as cp, pickle, statsmodels.api as sm
from tqdm import tqdm, trange
warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True) # Set numpy print options for better readability
np.random.seed(42) # for reproducibility

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



