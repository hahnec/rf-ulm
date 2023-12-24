import pandas as pd
import numpy as np
from scipy.signal import lfiltic, lfilter
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib import rcParams
rcParams['text.usetex'] = True


def numpy_ewma_vectorized(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def ewma(x, alpha):
    '''
    Returns the exponentially weighted moving average of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n,n)) * (1-alpha)
    p = np.vstack([np.arange(i,i-n,-1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0**p,0)

    # Calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)

def ewma_linear_filter(array, window):
    alpha = 2 /(window + 1)
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, array[0:1], [0])
    return lfilter(b, a, array, zi=zi)[0]

def ema(p:np.ndarray, a:float) -> np.ndarray:
    o = np.empty(p.shape, p.dtype)
    #   (1-α)^0, (1-α)^1, (1-α)^2, ..., (1-α)^n
    np.power(1.0 - a, np.arange(0.0, p.shape[0], 1.0, p.dtype), o)
    #   α*P0, α*P1, α*P2, ..., α*Pn
    np.multiply(a, p, p)
    #   α*P0/(1-α)^0, α*P1/(1-α)^1, α*P2/(1-α)^2, ..., α*Pn/(1-α)^n
    np.divide(p, o, p)
    #   α*P0/(1-α)^0, α*P0/(1-α)^0 + α*P1/(1-α)^1, ...
    np.cumsum(p, out=p)
    #   (α*P0/(1-α)^0)*(1-α)^0, (α*P0/(1-α)^0 + α*P1/(1-α)^1)*(1-α)^1, ...
    np.multiply(p, o, o)
    return o

# Specify the path to your CSV file
csv_file_path = 'val_rf12_sig.csv'

# Use pandas to read the CSV file, handling quotes
df = pd.read_csv(csv_file_path, delimiter=',', quoting=1, skiprows=1)

# Convert the DataFrame to a NumPy array
data = df.to_numpy(dtype=float)
rolling_mean = df.rolling(window=90).mean().to_numpy(dtype=float)[:, 1:]
#rolling_mean = df.ewm(span=20, adjust=False)[:, 1:]

# Assuming the first column represents x values, extract it
x = data[:, 0]

# Extract the remaining columns as y values for different graphs
ys = data[:, 1:]

list_idcs = [6,0,3] #[5,11,17]
labels = ['$\sigma=1$ (const.)','$\sigma=2$ (const.)','$\sigma=[3, ..., 1]$']
linestyles = [':','-.','-']
colors = ['#1b9e77', '#d95f02', '#7570b3']

ys_proc = []
for idx in list_idcs:
    ys_proc.append(rolling_mean.T[idx])

drop = 20

fig, ax = plt.subplots(figsize=(9,3))
for i, (r, l) in enumerate(zip(ys_proc, labels)):
    #plt.semilogy(np.arange(len(y)), y, label=str(i))
    ax.semilogy(np.arange(len(r))[::drop], r[::drop], label=str(l), linestyle=linestyles[i], linewidth=3, color=colors[i])

ax.set_xlim((0, len(r)))
ax.set_ylim((1/3, 4))
xlabel = ax.set_xlabel('Validation step [\#]', fontsize=20)
ylabel = ax.set_ylabel('Loss $\mathcal{L}\left(\mathbf{X},\mathbf{Y}\\right)$ [a.u.]', fontsize=20)
ylabel.set_y(0.5)

# switch off frame lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.set_major_locator(mticker.LogLocator(subs=[1, 4]))
#ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=20, subs="auto"))

# Set the tick label size for both x and y axes
ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=16)
ax.set_yticklabels(ax.get_yticks().astype(float), fontsize=16)

ax.legend(fontsize=18, shadow=False, frameon=True, edgecolor='black', loc='upper right', bbox_to_anchor=(1.0, 1.07))
plt.tight_layout()
plt.savefig('sigma_loss_plot.eps')
plt.show()
