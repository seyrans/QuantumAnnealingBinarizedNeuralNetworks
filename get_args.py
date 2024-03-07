# WARNING: Only mutate args in this file.
import argparse
from enum import Enum
import numpy as np
parser = argparse.ArgumentParser(description='BNN Verification')


class IterableArgs:
    pass


iterable_args = IterableArgs()
# ========== BNN Training 1 ==========
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

args = parser.parse_args()
args.optimizer = 'adadelta'  # Available options: ('adadelta', 'sgd', 'binop')
args.use_scheduler = False

# ========== Dataset ============
np.random.seed(args.seed)
args.batch_size = 64
iterable_args.selected_targets = [(0, 1)]  # Only 2 numbers supported
args.flatten_dataset = True
args.pad_flattened_dataset = True  # must flatten_dataset be True to pad the dataset
args.remove_contradicting = False
args.shuffle = False
args.partial_training_dataset = 10
# ========== Down-scaling ==========
args.downscale = True
args.downscale_batch_size = 100_000
args.dataset_spin_form = True

args.use_adaptive = True  # adaptive vs non-adaptive Convolutional AvgPool Layer

# non-adaptive
args.avg_pool_kernel_size = 4  # 4-> (7x7), 5 -> (9,9)
args.avg_pool_stride = 1
# args.shape_2d = (7, 7)  # if not matching adaptive_avg_pool_shape/avg_pool_kernel_size won't work!

# adaptive
# iterable_args.adaptive_avg_pool_shape = range(4, 9)  # 5-> (5x5), 6-> (6x6)
iterable_args.adaptive_avg_pool_shape = [8]

args.shape_2d = (28, 28)  # default
# ========== BNN Architecture ==========
iterable_args.num_layers = range(2, 6)
# iterable_args.num_layers = [2]
iterable_args.hidden_dim = [3, 7, 15]   # must be in the form of 2**n-1
# iterable_args.hidden_dim = [3]
args.use_binary_net2 = False
args.only_train = False

# ========== QUBO Reformulation ==========
# iterable_args.bnn_as_quso_ = [False] # FIXME QUSO is not supporting new objectives yet.
iterable_args.bnn_as_quso_ = [False]
args.bnn_as_quso_ = False
args.bnn_as_qubo_dimod_ = False

iterable_args.LAMBDA = [{
        'sum_taus': 0.1,  # Relative to the other lambdas. 0.1 means 10% of the other lambdas.
        'output': 1,
        'hard_constraints': 1,
        'perturbation_bound_constraint': 1,
        'epsilon': 1
    }
]
# iterable_args.objectives = ['zero', 'sum_taus', 'output']
# iterable_args.include_perturbation_bound_constraint = [False, True]
iterable_args.objectives = ['zero']
iterable_args.include_perturbation_bound_constraint = [True]  # sum(taus) < epsilon

# iterable_args.epsilon = [3, 7, 15, 31, 63]  # if set to None, epsilon will be optimized by the annealer
iterable_args.epsilon = [15]

# If pixels_to_perturb is None, We automatically find important pixels based on how frequently that pixel being used
# for either of classes.
args.pixels_to_perturb = None

# args.pixels_to_perturb = range(0, 31)
# args.pixels_to_perturb = [6, 7, 8, 11, 12, 13, 16, 17]  # 5x5
# args.pixels_to_perturb = [1, 2, 5, 6, 9, 10, 12, 13]  # 4x4
# args.pixels_to_perturb = [8, 9, 10, 14, 15, 16, 20]  # 6x6
# args.pixels_to_perturb = [8, 9, 10, 14, 15, 16, 20]  # 7x7
# args.pixels_to_perturb = [8, 9, 10, 14, 15, 16, 20]  # 8x8
# iterable_args.pixels_to_perturb_len = [3, 7, 15, 31, 63]
# iterable_args.pixels_to_perturb_len = [2**i-1 for i in range(2, 10)]
iterable_args.pixels_to_perturb_len = [15]


# ========== QUBO Solvers ==========
class Solvers(Enum):
    BRUTEFORCE = 0
    DWAVE = 1
    SIMULATED = 2
    QUBOVERT = 3
    GUROBI = 4
    KERBEROS = 5  # Hybrid
    HUB_STRA = 6


# iterable_args.solver = range(0, 9) # All solvers
# iterable_args.solver = [Solvers.SIMULATED, Solvers.SIMULATED_EMBEDDED]
# iterable_args.solver = [Solvers.SIMULATED, Solvers.SIMULATED_EMBEDDED, Solvers.DWAVE, Solvers.KERBEROS]
iterable_args.solver = [Solvers.SIMULATED]
args.solve_embedded = False

args.num_reads = 1000  # SIMULATED/DWAVE
args.annealing_time = 50  # DWAVE
args.num_anneals = 1000   # QUBOVERT
args.chain_strength = None
args.gurobi_threads = 16
args.kerberos_max_iter = 100
args.kerberos_convergence = 3
args.no_solver = False  # If True, we will not solve the problem

# HUB_STRA
args.hub_stra_epochs = 1000
args.hub_stra_lr = 0.01
args.hub_stra_log_every = 10
args.hub_stra_use_adam = False

args.num_images_to_solve = 1
args.starting_image_index = 0  # None to start from 0 index

# args.hub_stra_gradient_solver = Solvers.DWAVE
args.hub_stra_gradient_solver = Solvers.SIMULATED

# ========== Embedding ==========
args.embedding_parameters = {
    'verbose': 1,  # 3 for maximum log
    'interactive': True,
    'chainlength_patience': 1024*1,  # important to get better embedding (smaller chains)
    'max_no_improvement': 32
}
args.normalize_bqm = False
args.normalize_bqm_lbound = 1
args.normalize_bqm_ubound = 4

# ========== Result ==========
args.visualize_dataset = True
args.visualize_result_ = True
args.sample_output = False
args.check_solutions_match_tensors = True
args.report_to_wandb = True
args.wandb_project_name = "qavbnn-Jun12-v1"
args.dwave_inspector = True
args.export_H_as_pickle = True

"""
---
- Use args or sth like that for logging the parameters
- Use results={} to keep all the results
- log to a file each time running  the code. 
- the code should commit the changes before running the experiments
- 

"""