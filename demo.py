from nnenum import nnenum
import numpy as np
import onnx
import onnxruntime
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from nnenum.specification import Specification
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from nnenum.settings import Settings

p_range = [-6, -3]
theta_range = [0, 30]
p_num_bin = 24
theta_num_bin = 74
p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
p_lbs = np.array(p_bins[:-1],dtype=np.float32)
p_ubs = np.array(p_bins[1:], dtype=np.float32)

theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

#p_choices = [25, 32, 41, 46, 53, 68, 71, 89, 100, 121]
#theta_choices = [34, 61, 83, 28, 56, 79, 94, 13, 21, 67]
p_choices = 2
theta_choices = 1

p_lb = p_lbs[p_choices]
p_ub = p_ubs[p_choices]

theta_lb = theta_lbs[theta_choices]
theta_ub = theta_ubs[theta_choices]


model_path_1 = "./system_model_1_1.onnx"

init_box = [[-0.8, 0.8], [-0.8, 0.8]]
init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
init_box = np.array(init_box, dtype=np.float32)

nnenum.set_exact_settings()
Settings.ONNX_WHITELIST.append("TaxiNetDynamics")

Settings.CONTRACT_ZONOTOPE_LP = False # contract zonotope using LPs (even more accurate prefilter, but even slower)
Settings.CONTRACT_LP_OPTIMIZED = False # use optimized lp contraction
Settings.RESULT_SAVE_STARS = True

try:
    network = nnenum.load_onnx_network_optimized(model_path_1)
except:
    network = nnenum.load_onnx_network(model_path_1)



from nnenum.lp_star import LpStar
from nnenum.util import Freezable, compress_init_box
init_bm, init_bias, init_box = compress_init_box(init_box)
star = LpStar(init_bm, init_bias, init_box)

result = nnenum.enumerate_network(star, network)

substar = result.stars[2]
substar.a_mat = substar.a_mat[2:4, :]
substar.bias = substar.bias[2:4]

# check the star set's interval enclosure
p_ub = substar.minimize_output(0, True)
p_lb = substar.minimize_output(0, False)
theta_ub = substar.minimize_output(1, True)
theta_lb = substar.minimize_output(1, False)


# get the candidate grids


# intersection check





fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
verts = substar.verts()
codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
path = Path(verts, codes)
patch = patches.PathPatch(path, edgecolor='None', facecolor='blue', alpha=0.2)
ax.add_patch(patch)

## plot grids
for p_lb in p_lbs:
    X = [p_lb, p_lb]
    Y = [theta_bins[0], theta_bins[-1]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs:
    Y = [theta_lb, theta_lb]
    X = [p_bins[0], p_bins[-1]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

ax.set_xticks([i for i in np.linspace(-6, -3.4, 14)])
ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
ax.set_xlim([-6, -3.5])
ax.set_ylim([0, 15])
ax.set_xlabel(r"$p$ (m)")
ax.set_ylabel(r"$\theta$ (degrees)")

plt.savefig("demo.png")

