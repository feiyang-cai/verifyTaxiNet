import numpy as np
import os
from nnenum import nnenum
from nnenum.settings import Settings
from nnenum.lp_star import LpStar
from nnenum.util import compress_init_box
import math
from collections import defaultdict
import pickle
import onnxruntime as ort

class Verification():
    def __init__(self, onnx_filepath="./models/", reachable_set_path="./reachable_sets", p_range=[-11, 11], p_num_bin=128, theta_range=[-30, 30], theta_num_bin=128, reachability_steps=2, server_id=1, server_total_num=16) -> None:
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

        shared_library = "./libcustom_dynamics.so"
        so = ort.SessionOptions()
        so.register_custom_ops_library(shared_library)

        # check if onnx file exists and load them
        self.networks = []
        self.sessions = []
        for step in range(1, self.reachability_steps+1):
            onnx_file = os.path.join(onnx_filepath, f"system_model_{step}_1.onnx")
            if not os.path.exists(onnx_file):
                raise FileNotFoundError(f"File {onnx_file} does not exist.")
            
            self.networks.append(nnenum.load_onnx_network(onnx_file))
            self.sessions.append(ort.InferenceSession(onnx_file, so))
        
    def compute_interval_enclosure(self, star):
        p_ub = star.minimize_output(0, True)
        p_lb = star.minimize_output(0, False)
        theta_ub = star.minimize_output(1, True)
        theta_lb = star.minimize_output(1, False)

        # the cells may be out of the range (unsafe), filter them out
        if p_lb < self.p_lbs[0] or p_ub > self.p_ubs[-1]:
            return [[-1, -1], [-1, -1]]

        # get the cell index
        p_lb_idx = math.floor((p_lb - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # floor
        p_ub_idx = math.ceil((p_ub - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # ceil

        theta_lb_idx = math.floor((theta_lb - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # floor
        theta_ub_idx = math.ceil((theta_ub - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # ceil
        
        assert p_lb_idx >= 0 and p_ub_idx <= len(self.p_lbs)

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
    
    def get_reachable_cells(self, stars, reachable_cells):

        for star in stars:
            # get the p and theta
            star.a_mat = star.a_mat[2:4, :]
            star.bias = star.bias[2:4]

            # compute the interval enclosure for the star set
            interval_enclosure = self.compute_interval_enclosure(star)

            ## if p is out of the range (unsafe), then clear the reachable cells
            if interval_enclosure == [[-1, -1], [-1, -1]]:
                reachable_cells = set()
                reachable_cells.add((-2, -2, -2, -2))
                break
            
            # if the theta out of the range, discard the star
            if interval_enclosure[1][0] >= len(self.theta_lbs) or interval_enclosure[1][1] <= 0:
                reachable_cells.add((-3, -3, -3, -3))
                print("warning: theta out of the range")
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
                    if (self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]) not in reachable_cells:
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
                session = self.sessions[step-1]
                reachable_set = defaultdict(set)

                count = 0
                assert len(self.p_lbs) % self.server_total_num == 0
                start_point = len(self.p_lbs) // self.server_total_num * (self.server_id-1)
                end_point = len(self.p_lbs) // self.server_total_num * (self.server_id)
                for _, (p_lb, p_ub) in enumerate(zip(self.p_lbs[start_point:end_point], self.p_ubs[start_point:end_point])):
                    for _, (theta_lb, theta_ub) in enumerate(zip(self.theta_lbs, self.theta_ubs)):
                        count += 1
                        # if any reachable set with less steps is empty (means out of the range), then skip
                        if step > 1: 
                            assert len(reachable_set_multiple_steps[step-1][(p_lb, p_ub, theta_lb, theta_ub)]) > 0, "reachable set is empty"
                            if reachable_set_multiple_steps[step-1][(p_lb, p_ub, theta_lb, theta_ub)] == {(-1, -1, -1, -1)}:
                                reachable_set[(p_lb, p_ub, theta_lb, theta_ub)].add((-1, -1, -1, -1))
                                continue
                            elif reachable_set_multiple_steps[step-1][(p_lb, p_ub, theta_lb, theta_ub)] == {(-2, -2, -2, -2)}:
                                reachable_set[(p_lb, p_ub, theta_lb, theta_ub)].add((-2, -2, -2, -2))
                                continue
                        
                        reachable_cells = set()
                        init_box = [[-0.8, 0.8], [-0.8, 0.8]]
                        init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
                        init_box = np.array(init_box, dtype=np.float32)
                        init_bm, init_bias, init_box = compress_init_box(init_box)
                        star = LpStar(init_bm, init_bias, init_box)
                        print(f"Computing reachable set for p_lb={p_lb}, theta_lb={theta_lb}, reachable_step={step}")

                        # simulations to get the reachable set to avoid some intersection checking
                        samples = 5000
                        z = np.random.uniform(-0.8, 0.8, size=(samples, 2)).astype(np.float32)
                        p = np.random.uniform(p_lb, p_ub, size=(samples, 1)).astype(np.float32)
                        theta = np.random.uniform(theta_lb, theta_ub, size=(samples, 1)).astype(np.float32)
                        input_name = session.get_inputs()[0].name
                        input_shape = session.get_inputs()[0].shape
                        output_name = session.get_outputs()[0].name
                        for z_i, p_i, theta_i in zip(z, p, theta):
                            input_0 = np.concatenate([z_i, p_i, theta_i]).astype(np.float32).reshape(input_shape)
                            res = session.run([output_name], {input_name: input_0})
                            _, _, p, theta = res[0][0]
                            # initial check the reachable set
                            ## the cells may be out of the range, filter them out
                            if p < self.p_lbs[0] or p > self.p_ubs[-1]:
                                reachable_cells = set()
                                reachable_cells.add((-2, -2, -2, -2))
                                break

                            if theta < self.theta_lbs[0] or theta > self.theta_ubs[-1]:
                                reachable_cells.add((-3, -3, -3, -3))
                                continue

                            # get the cell index
                            p_idx = math.floor((p - self.p_lbs[0])/(self.p_ubs[0]-self.p_lbs[0])) # floor
                            theta_idx = math.floor((theta - self.theta_lbs[0])/(self.theta_ubs[0]-self.theta_lbs[0])) # floor
                            assert 0 <= p_idx < len(self.p_lbs), "p_idx out of range"
                            assert 0 <= theta_idx < len(self.theta_lbs), "theta_idx out of range"

                            reachable_cells.add((self.p_lbs[p_idx], self.p_ubs[p_idx], self.theta_lbs[theta_idx], self.theta_ubs[theta_idx]))

                        if not reachable_cells == {(-2, -2, -2, -2)}:
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
                                reachable_cells = self.get_reachable_cells(result.stars, reachable_cells)
                        reachable_set[(p_lb, p_ub, theta_lb, theta_ub)] = reachable_cells
                        print(reachable_cells)

                        # save the reachable set
                        if count % 100 == 0:
                            temp_reachable_set_file = os.path.join(self.reachable_set_path, 
                                                                    f"reachable_set_analysis_step_{step}_{int(self.p_lbs[0])}_{int(self.p_ubs[-1])}_{len(self.p_lbs)}_{int(self.theta_lbs[0])}_{int(self.theta_ubs[-1])}_{len(self.theta_lbs)}_{self.server_id}_{self.server_total_num}_temp_{count}.pkl")
                            with open(temp_reachable_set_file, "wb") as f:
                                pickle.dump(reachable_set, f)

                reachable_set_multiple_steps[step] = reachable_set
                with open(reachable_set_file, "wb") as f:
                    pickle.dump(reachable_set, f)
        

if __name__ == "__main__":
    veri = Verification(p_range=[-11.0, 11.0], theta_range=[-30.0, 30.0], reachability_steps=3, p_num_bin=128, theta_num_bin=128, server_total_num=16, server_id=1)
    veri.compute_reachable_set()