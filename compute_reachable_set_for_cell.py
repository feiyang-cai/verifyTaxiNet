import argparse
import numpy as np
from nnenum import nnenum
from nnenum.settings import Settings
from nnenum.lp_star import LpStar
from nnenum.util import compress_init_box
import onnxruntime as ort
import time
import math

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--p_lb', type=float, default=0.0, help='Lower bound for p')
    parser.add_argument('--p_ub', type=float, default=0.1, help='Upper bound for p')
    parser.add_argument('--theta_lb', type=float, default=0.0, help='Lower bound for theta')
    parser.add_argument('--theta_ub', type=float, default=0.1, help='Upper bound for theta')
    parser.add_argument('--reachability_steps', type=int, default=2, help='Number of reachability steps')
    parser.add_argument('--p_range_lb', type=float, default=-11.0, help='Lower bound for p_range')
    parser.add_argument('--p_range_ub', type=float, default=+11.0, help='Upper bound for p_range')
    parser.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p')
    parser.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta_range')
    parser.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta_range')
    parser.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    p_lb = args.p_lb
    p_ub = args.p_ub
    theta_lb = args.theta_lb
    theta_ub = args.theta_ub
    reachability_steps = args.reachability_steps

    p_range_lb = args.p_range_lb
    p_range_ub = args.p_range_ub
    p_num_bin = args.p_num_bin
    theta_range_lb = args.theta_range_lb
    theta_range_ub = args.theta_range_ub
    theta_num_bin = args.theta_num_bin

    p_bins = np.linspace(p_range_lb, p_range_ub, p_num_bin+1, endpoint=True)
    p_lbs = np.array(p_bins[:-1],dtype=np.float32)
    p_ubs = np.array(p_bins[1:], dtype=np.float32)

    theta_bins = np.linspace(theta_range_lb, theta_range_ub, theta_num_bin+1, endpoint=True)
    theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
    theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

    lbs = np.array(np.meshgrid(p_lbs, theta_lbs)).T.reshape(-1,2)
    ubs = np.array(np.meshgrid(p_ubs, theta_ubs)).T.reshape(-1,2)
    assert p_ub - p_lb == p_ubs[0] - p_lbs[0]
    assert theta_ub - theta_lb == theta_ubs[0] - theta_lbs[0]

    def compute_interval_enclosure(star):
        p_ub = star.minimize_output(0, True)
        p_lb = star.minimize_output(0, False)
        theta_ub = star.minimize_output(1, True)
        theta_lb = star.minimize_output(1, False)

        # the cells may be out of the range (unsafe), filter them out
        if p_lb < p_lbs[0] or p_ub > p_ubs[-1]:
            return [[-1, -1], [-1, -1]]

        # get the cell index
        p_lb_idx = math.floor((p_lb - p_lbs[0])/(p_ubs[0]-p_lbs[0])) # floor
        p_ub_idx = math.ceil((p_ub - p_lbs[0])/(p_ubs[0]-p_lbs[0])) # ceil

        theta_lb_idx = math.floor((theta_lb - theta_lbs[0])/(theta_ubs[1]-theta_lbs[0])) # floor
        theta_ub_idx = math.ceil((theta_ub - theta_lbs[0])/(theta_ubs[0]-theta_lbs[0])) # ceil

        assert p_lb_idx >= 0 and p_ub_idx <= len(p_lbs)

        theta_lb_idx = max(theta_lb_idx, 0)
        theta_ub_idx = min(theta_ub_idx, len(theta_lbs))

        return [[p_lb_idx, p_ub_idx], [theta_lb_idx, theta_ub_idx]]

    def check_intersection(star, p_idx, theta_idx):
        p_lb = p_lbs[p_idx]
        p_ub = p_ubs[p_idx]
        theta_lb = theta_lbs[theta_idx]
        theta_ub = theta_ubs[theta_idx]

        p_bias = star.bias[0]
        theta_bias = star.bias[1]

        if "ita" not in star.lpi.names:
            p_mat = star.a_mat[0, :]
            theta_mat = star.a_mat[1, :]

            # add the objective variable 'ita'
            star.lpi.add_cols(['ita'])

            # add constraints

            ## p_mat * p - ita <= p_ub - p_bias
            p_mat_1 = np.hstack((p_mat, -1))
            star.lpi.add_dense_row(p_mat_1, p_ub - p_bias)

            ## -p_mat * p - ita <= -p_lb + p_bias
            p_mat_2 = np.hstack((-p_mat, -1))
            star.lpi.add_dense_row(p_mat_2, -p_lb + p_bias)

            ## theta_mat * theta - ita <= theta_ub - theta_bias
            theta_mat_1 = np.hstack((theta_mat, -1))
            star.lpi.add_dense_row(theta_mat_1, theta_ub - theta_bias)

            ## -theta_mat * theta - ita <= -theta_lb + theta_bias
            theta_mat_2 = np.hstack((-theta_mat, -1))
            star.lpi.add_dense_row(theta_mat_2, -theta_lb + theta_bias)

        else:
            rhs = star.lpi.get_rhs()
            rhs[-4] = p_ub - p_bias
            rhs[-3] = -p_lb + p_bias
            rhs[-2] = theta_ub - theta_bias
            rhs[-1] = -theta_lb + theta_bias
            star.lpi.set_rhs(rhs)

        direction_vec = [0] * star.lpi.get_num_cols()
        direction_vec[-1] = 1
        rv = star.lpi.minimize(direction_vec)
        return rv[-1] <= 0.0

    def get_reachable_cells(stars, reachable_cells):

        for star in stars:
            # get the p and theta
            star.a_mat = star.a_mat[2:4, :]
            star.bias = star.bias[2:4]

            # compute the interval enclosure for the star set
            interval_enclosure = compute_interval_enclosure(star)

            ## if p is out of the range (unsafe), then clear the reachable cells
            if interval_enclosure == [[-1, -1], [-1, -1]]:
                reachable_cells = set()
                reachable_cells.add((-2, -2, -2, -2))
                break
            
            # if the theta out of the range, discard the star
            if interval_enclosure[1][0] >= len(theta_lbs) or interval_enclosure[1][1] <= 0:
                reachable_cells.add((-3, -3, -3, -3))
                print("warning: theta out of the range")
                continue

            assert interval_enclosure[0][0] <= interval_enclosure[0][1] - 1
            assert interval_enclosure[1][0] <= interval_enclosure[1][1] - 1

            ## if only one candidate cell, then skip
            if interval_enclosure[0][0] == interval_enclosure[0][1] - 1 and interval_enclosure[1][0] == interval_enclosure[1][1] - 1:
                reachable_cells.add((p_lbs[interval_enclosure[0][0]], p_ubs[interval_enclosure[0][0]],
                                     theta_lbs[interval_enclosure[1][0]], theta_ubs[interval_enclosure[1][0]]))
                continue

            # intersection check for the candidate cells
            for p_idx in range(interval_enclosure[0][0], interval_enclosure[0][1]):
                for theta_idx in range(interval_enclosure[1][0], interval_enclosure[1][1]):
                    if (p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx]) not in reachable_cells:
                        if check_intersection(star, p_idx, theta_idx):
                            reachable_cells.add(((p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx])))

        return reachable_cells


    # set nneum settings
    nnenum.set_exact_settings()
    Settings.ONNX_WHITELIST.append("TaxiNetDynamics")
    #Settings.CONTRACT_ZONOTOPE_LP = False # contract zonotope using LPs (even more accurate prefilter, but even slower)
    #Settings.CONTRACT_LP_OPTIMIZED = False # use optimized lp contraction
    Settings.GLPK_TIMEOUT = 10
    Settings.PRINT_OUTPUT = False
    Settings.TIMING_STATS = False
    Settings.RESULT_SAVE_STARS = True

    shared_library = "./libcustom_dynamics.so"
    so = ort.SessionOptions()
    so.register_custom_ops_library(shared_library)

    onnx_file = f"./models/system_model_{reachability_steps}_1.onnx"
    network = nnenum.load_onnx_network(onnx_file)
    session = ort.InferenceSession(onnx_file, so)

    t_start_cell = time.time()    
    reachable_cells = set()
    init_box = [[-0.8, 0.8], [-0.8, 0.8]]
    init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
    init_box = np.array(init_box, dtype=np.float32)
    init_bm, init_bias, init_box = compress_init_box(init_box)
    star = LpStar(init_bm, init_bias, init_box)
    print(f"Computing reachable set for p_lb={p_lb}, theta_lb={theta_lb}")

    # simulations to get the reachable set to avoid some intersection checking
    samples = 5000
    z = np.random.uniform(-0.8, 0.8, size=(samples, 2)).astype(np.float32)
    p = np.random.uniform(p_lb, p_ub, size=(samples, 1)).astype(np.float32)
    theta = np.random.uniform(theta_lb, theta_ub, size=(samples, 1)).astype(np.float32)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    t_start_sim = time.time()
    for z_i, p_i, theta_i in zip(z, p, theta):
        assert p_i<=p_ub and p_i>=p_lb
        assert theta_i<=theta_ub and theta_i>=theta_lb
        input_0 = np.concatenate([z_i, p_i, theta_i]).astype(np.float32).reshape(input_shape)
        res = session.run([output_name], {input_name: input_0})
        _, _, p_, theta_ = res[0][0]
        # initial check the reachable set
        ## the cells may be out of the range, filter them out
        if p_ < p_lbs[0] or p_ > p_ubs[-1]:
            reachable_cells = set()
            reachable_cells.add((-2, -2, -2, -2))
            break

        if theta_ < theta_lbs[0] or theta_ > theta_ubs[-1]:
            reachable_cells.add((-3, -3, -3, -3))
            continue

        # get the cell index
        p_idx = math.floor((p_ - p_lbs[0])/(p_ubs[0]-p_lbs[0])) # floor
        theta_idx = math.floor((theta_ - theta_lbs[0])/(theta_ubs[0]-theta_lbs[0])) # floor
        assert 0 <= p_idx < len(p_lbs), "p_idx out of range"
        assert 0 <= theta_idx < len(theta_lbs), "theta_idx out of range"

        reachable_cells.add((p_lbs[p_idx], p_ubs[p_idx], theta_lbs[theta_idx], theta_ubs[theta_idx]))
    t_sim = time.time() - t_start_sim

    t_enumerate_nn = dict()
    t_get_reachable_cells = None
    if not reachable_cells == {(-2, -2, -2, -2)}:
        for split_tolerance in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            print(f"split_tolerance={split_tolerance}")
            Settings.SPLIT_TOLERANCE = split_tolerance # small outputs get rounded to zero when deciding if splitting is possible
            result = nnenum.enumerate_network(star, network)
            t_enumerate_nn[split_tolerance] = result.total_secs
            print(result.result_str)
            if result.result_str != "error":
                break
        if result.result_str == "error":
            reachable_cells = set()
            reachable_cells.add((-1, -1, -1, -1))
        else:
            t_start_get_reachable_cells = time.time()
            reachable_cells = get_reachable_cells(result.stars, reachable_cells)
            t_get_reachable_cells = time.time() - t_start_get_reachable_cells
    total_cell_time = time.time() - t_start_cell
    print(reachable_cells)
    print(t_sim, t_enumerate_nn, t_get_reachable_cells, total_cell_time)
    print("--------------------------------------------------------")


if __name__ == '__main__':
    main()