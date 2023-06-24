from src.verification_utils import Verification
import argparse


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add the arguments
    parser.add_argument('--p_range_lb', type=float, default=-10.0, help='Lower bound for p_range')
    parser.add_argument('--p_range_ub', type=float, default=+10.0, help='Upper bound for p_range')
    parser.add_argument('--p_num_bin', type=int, default=128, help='Number of bins for p')
    parser.add_argument('--theta_range_lb', type=float, default=-30.0, help='Lower bound for theta_range')
    parser.add_argument('--theta_range_ub', type=float, default=+30.0, help='Upper bound for theta_range')
    parser.add_argument('--theta_num_bin', type=int, default=128, help='Number of bins for theta')
    parser.add_argument('--reachability_steps', type=int, default=3, help='Number of reachability steps')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the argument values
    p_range_lb = args.p_range_lb
    p_range_ub = args.p_range_ub
    p_num_bin = args.p_num_bin
    theta_range_lb = args.theta_range_lb
    theta_range_ub = args.theta_range_ub
    theta_num_bin = args.theta_num_bin
    reachability_steps = args.reachability_steps

    # Use the argument values in your program
    veri = Verification(p_range=[p_range_lb, p_range_ub],
                        theta_range=[theta_range_lb, theta_range_ub],
                        reachability_steps=reachability_steps,
                        p_num_bin=p_num_bin, 
                        theta_num_bin=theta_num_bin)
    veri.compute_reachable_set()

if __name__ == '__main__':
    main()