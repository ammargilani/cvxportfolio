import torch
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk

plt.style.use("cyberpunk")
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor)
torch.autograd.set_detect_anomaly(False)

# Debugging
debugging_size = 10

# Hyperparemters
gamma_trade = torch.tensor(1.)
gamma_hold = torch.tensor(1.)
gamma_risk = torch.tensor(5.)
leverage_limit = torch.tensor(1e3)
turnover_limit = torch.tensor(3.)
min_cash = torch.tensor(-1.)
w_min = torch.tensor(-1.)
w_max = torch.tensor(1.)

PARAMETRIC_CONSTRAINTS_START_IDX = 2
GAMMA_TRADE_IDX = 0
GAMMA_HOLD_IDX = 1
GAMMA_RISK_IDX = 2
CONSTRAINTS_HP_START_IDX = 3
LEVERAGE_LIMIT_IDX = 3
TURNOVER_LIMIT_IDX = 4
MIN_PORTFOLIO_IDX = 5
MAX_PORTFOLIO_IDX = 6

# Configurations
silent_warnings = False
inactive_constraint_tol = 1e-5
