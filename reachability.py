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
model_path_2 = "./system_model_2_1.onnx"
model_path_3 = "./system_model_3_1.onnx"

init_box = [[-0.8, 0.8], [-0.8, 0.8]]
init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
init_box = np.array(init_box, dtype=np.float32)

## plot the axis
fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

from nnenum.settings import Settings
nnenum.set_exact_settings()
#nnenum.set_control_settings()
#Settings.RESULT_SAVE_STARS = True
Settings.ONNX_WHITELIST.append("TaxiNetDynamics")
#Settings.NUM_PROCESSES = 0
#Settings.LP_SOLVER = "Gurobi"

Settings.CONTRACT_ZONOTOPE_LP = False # contract zonotope using LPs (even more accurate prefilter, but even slower)
Settings.CONTRACT_LP_OPTIMIZED = False # use optimized lp contraction
Settings.GLPK_TIMEOUT = 10
#Settings.GLPK_FIRST_PRIMAL = False # first try primal LP... if that fails do dual
#Settings.SKIP_CONSTRAINT_NORMALIZATION = True # disable constraint normalization in LP (may reduce stability) 
#Settings.UNDERFLOW_BEHAVIOR = 'warn' # np.seterr behavior for floating-point underflow

#Settings.CONTRACT_LP_TRACK_WITNESSES = False # track box bounds witnesses to reduce LP solving
#Settings.EAGER_BOUNDS = False
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

#mat = np.array([[0., 0., -1., 0.]])
#rhs = np.array([3.2])
#spec = Specification(mat, rhs)
#result = nnenum.enumerate_network(star, network, spec)

for idx, substar in enumerate(result.stars):
    print(idx)
    #substar = result.stars[2665]
    substar.a_mat = substar.a_mat[2:4, :]
    substar.bias = substar.bias[2:4]
    
    verts = substar.verts()
    if len(verts) < 2:
        continue
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, edgecolor='None', facecolor='blue', alpha=0.2)
    ax.add_patch(patch)

# second step
try:
    network = nnenum.load_onnx_network_optimized(model_path_2)
except:
    network = nnenum.load_onnx_network(model_path_2)

from nnenum.lp_star import LpStar
from nnenum.util import Freezable, compress_init_box
init_bm, init_bias, init_box = compress_init_box(init_box)
star = LpStar(init_bm, init_bias, init_box)

result = nnenum.enumerate_network(star, network)

#mat = np.array([[0., 0., -1., 0.]])
#rhs = np.array([3.2])
#spec = Specification(mat, rhs)
#result = nnenum.enumerate_network(star, network, spec)

for idx, substar in enumerate(result.stars):
    print(idx)
    #substar = result.stars[2665]
    substar.a_mat = substar.a_mat[2:4, :]
    substar.bias = substar.bias[2:4]
    
    verts = substar.verts()
    if len(verts) < 2:
        continue
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, edgecolor='None', facecolor='blue', alpha=0.2)
    ax.add_patch(patch)



# third step
try:
    network = nnenum.load_onnx_network_optimized(model_path_3)
except:
    network = nnenum.load_onnx_network(model_path_3)

from nnenum.lp_star import LpStar
from nnenum.util import Freezable, compress_init_box
init_bm, init_bias, init_box = compress_init_box(init_box)
star = LpStar(init_bm, init_bias, init_box)

result = nnenum.enumerate_network(star, network)

#mat = np.array([[0., 0., -1., 0.]])
#rhs = np.array([3.2])
#spec = Specification(mat, rhs)
#result = nnenum.enumerate_network(star, network, spec)

for idx, substar in enumerate(result.stars):
    print(idx)
    #substar = result.stars[2665]
    substar.a_mat = substar.a_mat[2:4, :]
    substar.bias = substar.bias[2:4]
    
    verts = substar.verts()
    if len(verts) < 2:
        continue
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, edgecolor='None', facecolor='blue', alpha=0.2)
    ax.add_patch(patch)

# simulation
## simulation
shared_library = "libcustom_dynamics.so"
so = onnxruntime.SessionOptions()
so.register_custom_ops_library(shared_library)

samples = 1000
ps = np.random.uniform(p_lb, p_ub, (samples, 1))
thetas = np.random.uniform(theta_lb, theta_ub, (samples, 1))
zs = np.random.uniform(-0.8, 0.8, (samples, 2))

session_1 = onnxruntime.InferenceSession(model_path_1, so)
session_2 = onnxruntime.InferenceSession(model_path_2, so)
session_3 = onnxruntime.InferenceSession(model_path_3, so)
input_shape = session_1.get_inputs()[0].shape

input_name = session_1.get_inputs()[0].name
input_shape = session_1.get_inputs()[0].shape
output_name_1 = session_1.get_outputs()[0].name
output_name_2 = session_2.get_outputs()[0].name
output_name_3 = session_3.get_outputs()[0].name

inputs = np.hstack((zs, ps, thetas)).astype(np.float32)
states_1 = []
states_2 = []
states_3 = []
for input in inputs:
    input = input.reshape(input_shape)
    s_1 = session_1.run([output_name_1], {input_name: input})
    states_1.append(s_1[0].reshape(-1)[2:4])
    s_2 = session_2.run([output_name_2], {input_name: input})
    states_2.append(s_2[0].reshape(-1)[2:4])
    s_3 = session_3.run([output_name_3], {input_name: input})
    states_3.append(s_3[0].reshape(-1)[2:4])

states_0 = inputs[:, 2:4]
scatter0 = ax.scatter(states_0[:, 0], states_0[:, 1], c='red', s=1, alpha=0.1)

states_1 = np.array(states_1)
scatter1 = ax.scatter(states_1[:, 0], states_1[:, 1], c='red', s=1, alpha=0.1)

states_2 = np.array(states_2)
scatter2 = ax.scatter(states_2[:, 0], states_2[:, 1], c='red', s=1, alpha=0.1)

states_3 = np.array(states_3)
scatter3 = ax.scatter(states_3[:, 0], states_3[:, 1], c='red', s=1, alpha=0.1)

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

plt.savefig("reachability_3steps_check.png")