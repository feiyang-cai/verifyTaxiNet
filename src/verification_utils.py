import numpy as np
import os
from nnenum import nnenum
from nnenum.settings import Settings
from nnenum.lp_star import LpStar
from nnenum.util import compress_init_box
import math
from collections import defaultdict
import pickle

class Verification():
    def __init__(self, onnx_filepath="./models/", reachable_set_path="./reachable_sets", p_range=[-10, 10], p_num_bin=128, theta_range=[-30, 30], theta_num_bin=128, reachability_steps=2, server_id=1, server_total_num=16) -> None:
        self.p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
        self.p_lbs = np.array(self.p_bins[:-1],dtype=np.float32)
        self.p_ubs = np.array(self.p_bins[1:], dtype=np.float32)

        self.theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
        self.theta_lbs = np.array(self.theta_bins[:-1],dtype=np.float32)
        self.theta_ubs = np.array(self.theta_bins[1:], dtype=np.float32)

        self.lbs = np.array(np.meshgrid(self.p_lbs, self.theta_lbs)).T.reshape(-1,2)
        self.ubs = np.array(np.meshgrid(self.p_ubs, self.theta_ubs)).T.reshape(-1,2)

        self.reachability_steps = reachability_steps
        self.reachable_set_path = reachable_set_path

        self.server_id = server_id
        self.server_total_num = server_total_num
        assert 0<self.server_id <= self.server_total_num

        # mkdir if not exist
        os.mkdir(self.reachable_set_path) if not os.path.exists(self.reachable_set_path) else None

        # set nneum settings
        nnenum.set_exact_settings()
        Settings.ONNX_WHITELIST.append("TaxiNetDynamics")
        Settings.CONTRACT_ZONOTOPE_LP = False # contract zonotope using LPs (even more accurate prefilter, but even slower)
        Settings.CONTRACT_LP_OPTIMIZED = False # use optimized lp contraction
        Settings.GLPK_TIMEOUT = 10
        Settings.PRINT_OUTPUT = False
        Settings.TIMING_STATS = False
        Settings.RESULT_SAVE_STARS = True

        # check if onnx file exists and load them
        self.networks = []
        for step in range(1, self.reachability_steps+1):
            onnx_file = os.path.join(onnx_filepath, f"system_model_{step}_1.onnx")
            if not os.path.exists(onnx_file):
                raise FileNotFoundError(f"File {onnx_file} does not exist.")
            
            self.networks.append(nnenum.load_onnx_network(onnx_file))
        
    def compute_interval_enclosure(self, star):
        p_ub = star.minimize_output(0, True)
        p_lb = star.minimize_output(0, False)
        theta_ub = star.minimize_output(1, True)
        theta_lb = star.minimize_output(1, False)

        # the cells may be out of the range, filter them out
        if p_ub < self.p_lbs[0] or p_lb > self.p_ubs[-1]:
            return [[-1, -1], [-1, -1]]

        # get the cell index
        p_lb_idx = math.floor((p_lb - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # floor
        p_ub_idx = math.ceil((p_ub - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # ceil

        theta_lb_idx = math.floor((theta_lb - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # floor
        theta_ub_idx = math.ceil((theta_ub - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # ceil

        # filter out the cells out of the range
        p_lb_idx = max(p_lb_idx, 0)
        p_ub_idx = min(p_ub_idx, len(self.p_lbs))

        theta_lb_idx = max(theta_lb_idx, 0)
        theta_ub_idx = min(theta_ub_idx, len(self.theta_lbs))

        return [[p_lb_idx, p_ub_idx], [theta_lb_idx, theta_ub_idx]]
    
    def check_intersection(self, star, p_idx, theta_idx):
        p_lb = self.p_lbs[p_idx]
        p_ub = self.p_ubs[p_idx]
        theta_lb = self.theta_lbs[theta_idx]
        theta_ub = self.theta_ubs[theta_idx]

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
    
    def get_reachable_cells(self, stars):
        reachable_cells = set()

        for star in stars:
            # get the p and theta
            star.a_mat = star.a_mat[2:4, :]
            star.bias = star.bias[2:4]

            # compute the interval enclosure for the star set
            interval_enclosure = self.compute_interval_enclosure(star)

            ## if the star is out of the range, then clear the reachable cells
            if interval_enclosure[0][0] == -1:
                reachable_cells = set()
                reachable_cells.add((-2, -2, -2, -2))
                break
            
            # if the theta out of the range, discard the star
            if interval_enclosure[1][0] >= len(self.theta_lbs) or interval_enclosure[1][1] <= 0:
                continue

            assert interval_enclosure[0][0] <= interval_enclosure[0][1] - 1
            assert interval_enclosure[1][0] <= interval_enclosure[1][1] - 1

            ## if only one candidate cell, then skip
            if interval_enclosure[0][0] == interval_enclosure[0][1] - 1 and interval_enclosure[1][0] == interval_enclosure[1][1] - 1:
                reachable_cells.add((self.p_lbs[interval_enclosure[0][0]], self.p_ubs[interval_enclosure[0][0]],
                                     self.theta_lbs[interval_enclosure[1][0]], self.theta_ubs[interval_enclosure[1][0]]))
                continue

            # intersection check for the candidate cells
            for p_idx in range(interval_enclosure[0][0], interval_enclosure[0][1]):
                for theta_idx in range(interval_enclosure[1][0], interval_enclosure[1][1]):
                    if self.check_intersection(star, p_idx, theta_idx):
                        reachable_cells.add(((self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])))

        return reachable_cells
    
    def compute_reachable_set(self):
        reachable_set_multiple_steps = dict()
        for step in range(1, self.reachability_steps+1):
            # try to load reachable set
            reachable_set_file = os.path.join(self.reachable_set_path, 
                                              f"reachable_set_analysis_step_{step}_{int(self.p_lbs[0])}_{int(self.p_ubs[-1])}_{len(self.p_lbs)}_{int(self.theta_lbs[0])}_{int(self.theta_ubs[-1])}_{len(self.theta_lbs)}_{self.server_id}_{self.server_total_num}.pkl")
            try:
                with open(reachable_set_file, "rb") as f:
                    reachable_set = pickle.load(f)
                reachable_set_multiple_steps[step] = reachable_set
                print(f"Reachable set for step {step} already exists.")
            except:
                network = self.networks[step-1]
                reachable_set = defaultdict(set)

                count = 0
                assert len(self.p_lbs) % self.server_total_num == 0
                start_point = len(self.p_lbs) // self.server_total_num * (self.server_id-1)
                end_point = len(self.p_lbs) // self.server_total_num * (self.server_id)
                for p_idx, (p_lb, p_ub) in enumerate(zip(self.p_lbs[start_point:end_point], self.p_ubs[start_point:end_point])):
                    for theta_idx, (theta_lb, theta_ub) in enumerate(zip(self.theta_lbs, self.theta_ubs)):
                        p_idx += start_point
                        count += 1
                        # if any reachable set with less steps is empty (means out of the range), then skip
                        if step > 1 and \
                            (len(reachable_set_multiple_steps[step-1][(self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])]) == 0 or \
                                reachable_set_multiple_steps[step-1][(self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])] == {-1, -1, -1, -1}):
                            reachable_set[(self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])] = set()
                            continue
                        
                        init_box = [[-0.8, 0.8], [-0.8, 0.8]]
                        init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
                        init_box = np.array(init_box, dtype=np.float32)
                        init_bm, init_bias, init_box = compress_init_box(init_box)
                        star = LpStar(init_bm, init_bias, init_box)
                        print(f"Computing reachable set for p_idx={p_idx}, theta_idx={theta_idx}, reachable_step={step}")
                        
                        for split_tolerance in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                            print(f"split_tolerance={split_tolerance}")
                            Settings.SPLIT_TOLERANCE = split_tolerance # small outputs get rounded to zero when deciding if splitting is possible
                            result = nnenum.enumerate_network(star, network)
                            if result.result_str != "error":
                                break
                        if result.result_str == "error":
                            reachable_cells = set()
                            reachable_cells.add((-1, -1, -1, -1))
                        else:
                            reachable_cells = self.get_reachable_cells(result.stars)
                        print(reachable_cells)
                        reachable_set[(self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx])] = reachable_cells

                        # save the reachable set
                        if count % 100 == 0:
                            temp_reachable_set_file = os.path.join(self.reachable_set_path, 
                                                                    f"reachable_set_analysis_step_{step}_{int(self.p_lbs[0])}_{int(self.p_ubs[-1])}_{len(self.p_lbs)}_{int(self.theta_lbs[0])}_{int(self.theta_ubs[-1])}_{len(self.theta_lbs)}_{self.server_id}_{self.server_total_num}_temp_{count}.pkl")
                            with open(temp_reachable_set_file, "wb") as f:
                                pickle.dump(reachable_set, f)

                reachable_set_multiple_steps[step] = reachable_set
                with open(reachable_set_file, "wb") as f:
                    pickle.dump(reachable_set, f)
    """
    def compute_not_solved_reachable_set(self):
        reachable_set_multiple_steps = dict()
        for step in range(1, self.reachability_steps+1):
            # try to load reachable set
            reachable_set_file = os.path.join(self.reachable_set_path, 
                                              f"reachable_set_analysis_step_{step}_{int(self.p_lbs[0])}_{int(self.p_ubs[-1])}_{len(self.p_lbs)}_{int(self.theta_lbs[0])}_{int(self.theta_ubs[-1])}_{len(self.theta_lbs)}.pkl")
            reachable_set_new_file = os.path.join(self.reachable_set_path, 
                                              f"reachable_set_new_analysis_step_{step}_{int(self.p_lbs[0])}_{int(self.p_ubs[-1])}_{len(self.p_lbs)}_{int(self.theta_lbs[0])}_{int(self.theta_ubs[-1])}_{len(self.theta_lbs)}.pkl")
            assert os.path.exists(reachable_set_file), f"Reachable set for step {step} does not exist."
            with open(reachable_set_file, "rb") as f:
                reachable_set = pickle.load(f)
            reachable_set_multiple_steps[step] = reachable_set
            print(f"Reachable set for step {step} already exists.")

            network = self.networks[step-1]
            reachable_set = defaultdict(set)
            for idx, (cell, sets) in enumerate(reachable_set_multiple_steps[step].items()):
                if sets == {(-1, -1, -1, -1)}:
                    p_lb = cell[0]
                    p_ub = cell[1]
                    theta_lb = cell[2]
                    theta_ub = cell[3]
                    init_box = [[-0.8, 0.8], [-0.8, 0.8]]
                    init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
                    init_box = np.array(init_box, dtype=np.float32)
                    init_bm, init_bias, init_box = compress_init_box(init_box)
                    star = LpStar(init_bm, init_bias, init_box)
                    print(f"Recomputing reachable set using tolerance split for p_lb={p_lb}, theta_lb={theta_lb}, reachable_step={step}")
                    for split_tolerance in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                        print(f"split_tolerance={split_tolerance}")
                        Settings.SPLIT_TOLERANCE = split_tolerance # small outputs get rounded to zero when deciding if splitting is possible
                        result = nnenum.enumerate_network(star, network)
                        if result.result_str != "error":
                            break
                    if result.result_str == "error":
                        reachable_cells = set()
                        reachable_cells.add((-1, -1, -1, -1))
                    else:
                        reachable_cells = self.get_reachable_cells(result.stars)
                    reachable_set[cell] = reachable_cells

            reachable_set_multiple_steps[step] = reachable_set
            with open(reachable_set_new_file, "wb") as f:
                pickle.dump(reachable_set, f)
        """
        

if __name__ == "__main__":
    veri = Verification(p_range=[-10.0, 10.0], theta_range=[-30.0, 30.0], reachability_steps=3, p_num_bin=128, theta_num_bin=128, server_total_num=16, server_id=1)
    veri.compute_reachable_set()