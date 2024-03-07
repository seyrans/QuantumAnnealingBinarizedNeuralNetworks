from neal import SimulatedAnnealingSampler
import matplotlib.pyplot as plt
from binarized_neural_net import *
from get_args import Solvers
from utils import to_spin, to_boolean
from minorminer import find_embedding
from dwave.system.composites import FixedEmbeddingComposite
import torch
from bnn_as_qubo_grb import set_up_grb_model_qubo
from bnn_as_qubo import le_zero_constraint_to_eq_zero_constraint
from dimod import BinaryQuadraticModel, Vartype
from dwave.system import DWaveSampler
import networkx as nx
import qubovert as qv
from hybrid.reference.kerberos import KerberosSampler
import wandb
import time
import random
from functools import partial
from qubovert.sim import anneal_qubo


def log_degrees(dwave_qubo):
    G = nx.Graph()
    G.add_weighted_edges_from([(u[0], u[1], v) for u, v in dwave_qubo.items()])
    degree_dict = dict(G.degree(G.nodes()))
    degrees = sorted(degree_dict.values(), reverse=True)
    print('Top 20 Max Degrees')
    print(degrees[:20])
    wandb.log({'degrees': degrees})
    return degrees


def time_execution(func, label='sampling_time'):
    execution_start = time.time()
    res = func()
    wandb.log({label: time.time() - execution_start})
    return res


def solve_using_simulated(H, args):
    H_qubo = H.to_qubo().Q
    sim_sampler = SimulatedAnnealingSampler()
    samples = sim_sampler.sample_qubo(H_qubo, num_reads=args.num_reads, seed=args.seed)
    return samples


def find_embedding_wrapper(dwave_qubo, args):
    d_sampler = DWaveSampler()
    _, edge_list, adjacency = d_sampler.structure
    embedding = time_execution(partial(find_embedding, dwave_qubo, edge_list,
                                       random_seed=args.seed, **args.embedding_parameters)
                               , 'embedding_time')
    return embedding, adjacency


class DWaveFixedEmbeddingSolver:
    def __init__(self, dwave_qubo, args):
        self.embedding, self.adjacency = find_embedding_wrapper(dwave_qubo, args)

    def solve(self, args, dwave_qubo):
        sampler = DWaveSampler()
        sampler = FixedEmbeddingComposite(sampler, self.embedding)
        samples = time_execution(partial(sampler.sample_qubo, dwave_qubo, num_reads=args.num_reads,
                                         annealing_time=args.annealing_time, seed=args.seed))
        return samples


def get_p_constraints(H):
    p_constraints = []
    for constraint in H._constraints['eq']:
        p_constraints.append(constraint)
    for constraint in H._constraints['lt']:
        new_constraint = le_zero_constraint_to_eq_zero_constraint(constraint)
        p_constraints.append(new_constraint)
    return p_constraints


def solve_hub_stra(H, args):
    p_constraints = get_p_constraints(H)  # p must be equal to zero.
    variables = set()
    for p_constraint in p_constraints:
        variables |= p_constraint.variables
    variables = list(variables)
    num_variables = len(variables)
    num_p_constraints = len(p_constraints)

    v = {i: random.random() for i in range(num_p_constraints)}
    h = {i: random.random()/10 for i in range(num_variables)}
    print(f'initial v: {v}')
    H_solution = None

    dwave_fixed_embedding_solver = None

    for epoch in range(args.hub_stra_epochs):
        H_new = qv.PCBO()
        for k in range(num_variables):
            H_new += h[k] * qv.boolean_var(variables[k])

        for i in range(num_p_constraints):
            H_new += - v[i] * p_constraints[i]

        if args.hug_stra_gradient_solver == Solvers.SIMULATED:
            samples = solve_using_simulated(H_new, args)
        elif args.hub_stra_gradient_solver == Solvers.DWAVE:
            dwave_qubo = H_new.to_qubo().Q
            if not dwave_fixed_embedding_solver:
                dwave_fixed_embedding_solver = DWaveFixedEmbeddingSolver(dwave_qubo, args)
            dwave_fixed_embedding_solver.solve(dwave_qubo, args)

        exp_val_fs = []
        constraint_values = []
        for i in range(num_p_constraints):
            exp_val_f_i = 0
            for sample in samples:
                H_solution = H_new.convert_solution(sample)
                f_i_val = p_constraints[i].value(H_solution)
                exp_val_f_i += f_i_val
            exp_val_f_i = int(exp_val_f_i / len(samples))
            exp_val_fs.append(exp_val_f_i)
            H_solution = H_new.convert_solution(samples.first.sample)
            constraint_values.append(int(p_constraints[i].value(H_solution)))
            v[i] = v[i] - args.hub_stra_lr * exp_val_f_i
        if epoch != 0 and epoch % args.hub_stra_log_every == 0:
            print(f'Epoch {epoch}', exp_val_fs, constraint_values, v)

    return H_solution


def exec_solver(H, args, H_dimod=None):
    solver = args.solver
    qv_qubo = H.to_qubo()   # Get the QUBO form of the H
    dwave_qubo = qv_qubo.Q  # D-Wave accept QUBOs in a different format than Qubovert's QUBO format
    dwave_bqm = BinaryQuadraticModel(Vartype.BINARY).from_qubo(dwave_qubo, H.offset)
    if H_dimod:
        dwave_bqm = H_dimod
    log_degrees(dwave_qubo)
    if args.solve_embedded or solver == Solvers.DWAVE:  # D-wave solver always require embedding
        import dwave.embedding
        embedding, adjacency = find_embedding_wrapper(dwave_qubo, args)
        dwave_bqm = dwave.embedding.embed_bqm(dwave_bqm, embedding, adjacency, chain_strength=args.chain_strength)
        if args.normalize_bqm:
            dwave_bqm.normalize(args.normalize_bqm_lbound, args.normalize_bqm_ubound)
        print('# embedded variables', dwave_bqm.num_variables)
        print('# emb quad term:', dwave_bqm.num_interactions)
        dwave_qubo = dwave_bqm.to_qubo()[0]
        log_degrees(dwave_qubo)

    samples = None
    # either generate samples or H_solution
    if solver == Solvers.HUB_STRA:
        H_solution = solve_hub_stra(H, args)
    elif solver == Solvers.BRUTEFORCE:
        H_solution = H.solve_bruteforce()
    elif solver == Solvers.KERBEROS:
        sampler = KerberosSampler()
        samples = time_execution(partial(sampler.sample, dwave_bqm, max_iter=args.kerberos_max_iter,
                                         convergence=args.kerberos_convergence))
    elif solver == Solvers.SIMULATED:
        sim_sampler = SimulatedAnnealingSampler()
        qubo_ = dwave_bqm.to_qubo()[0]
        samples = time_execution(partial(sim_sampler.sample_qubo, qubo_, num_reads=args.num_reads))
    elif solver == Solvers.DWAVE:
        sampler = DWaveSampler()
        sampler = FixedEmbeddingComposite(sampler, embedding)
        samples = time_execution(partial(sampler.sample_qubo, dwave_qubo, num_reads=args.num_reads,
                                         annealing_time=args.annealing_time))
        if args.dwave_inspector:
            import dwave.inspector
            dwave.inspector.show(samples)
        qubo_solution = samples.first.sample
    elif solver == Solvers.QUBOVERT:
        qv_samples = time_execution(partial(anneal_qubo, H, num_anneals=args.num_anneals))
        H_solution = qv_samples.best.state
    elif solver == Solvers.GUROBI:
        import gurobipy as grb
        grb_model, gurobi_vars, grb_loss = set_up_grb_model_qubo(dwave_qubo, const=qv_qubo[()], args=args)  # FIXME
        grb_model.setObjective(grb_loss, grb.GRB.MINIMIZE)
        samples = time_execution(partial(grb_model.optimize))
        grb_dwave_qubo_solution = {}
        for i in grb_model.getVars():
            if i.VarName != 'x':
                grb_dwave_qubo_solution[int(i.VarName)] = i.x
        H_solution = H.convert_solution(grb_dwave_qubo_solution)
    else:
        raise ValueError

    if args.solve_embedded or solver == Solvers.DWAVE:
        samples = dwave.embedding.unembed_sampleset(samples, embedding, dwave_bqm)
        qubo_solution = samples.first.sample
        # embedding_rev = {v_: k for k, v in embedding.items() for v_ in v}
        # qubo_solution = {embedding_rev[k]: v for k, v in qubo_solution.items()}

    if H_dimod:
        H_solution = qubo_solution

    pct_valid_solutions = 0
    avg_H_value = 0
    if samples and not H_dimod:
        for sample in samples:
            H_sample_solution = H.convert_solution(sample)
            pct_valid_solutions += H.is_solution_valid(H_sample_solution)
            avg_H_value += H.value(H_sample_solution)
        pct_valid_solutions /= len(samples)
        avg_H_value /= len(samples)
        qubo_best_solution = samples.first.sample
        H_solution = H.convert_solution(qubo_best_solution)
        H_best_value = H.value(H_solution)

    return H_solution, H_best_value, pct_valid_solutions, avg_H_value


def check_solutions_match_tensors(model, old_class, new_class, H_solution, tau_tensor, sample_input_target, qubo_vars, all_constraints_satisfied, args):
    num_layers = len(model.fc_list)
    matrix_product_sizes = [old_class.nn_weights[i].T.shape[1] for i in range(num_layers-1)]
    aux_matrix_product_sizes = [(old_class.nn_weights[i].shape[0], math.floor(math.log2(old_class.nn_weights[i].shape[1]))) for i in range(num_layers)]
    partial_matrix_product_sizes = [tuple(old_class.nn_weights[i].T.shape) for i in range(num_layers)]

    aux_keys = [k for k in H_solution if '__a' in k]
    aux_keys_sorted = sorted(aux_keys, key=lambda a: int(a.replace('__a', '')))
    aux = [H_solution[k] for k in aux_keys_sorted]
    aux_tensor = torch.tensor(aux)

    matrix_products_tensors = {}
    for layer in range(num_layers-1):
        col_size = matrix_product_sizes[layer]
        matrix_products_tensors[layer] = torch.zeros(col_size)
        for col_ind in range(col_size):
            m_i = H_solution[f'matrix_product_{layer}_{col_ind}']
            if args.bnn_as_quso_:
                m_i = to_boolean(m_i)
            matrix_products_tensors[layer][col_ind] = m_i

    partial_matrix_products_tensors = {}
    for layer in range(num_layers):
        row_size, col_size = partial_matrix_product_sizes[layer]
        partial_matrix_products_tensors[layer] = torch.zeros(row_size, col_size)
        for row_ind in range(row_size):
            for col_ind in range(col_size):
                if f'partial_matrix_product_{layer}_{row_ind}_{col_ind}' in H_solution:
                    partial_matrix_products_tensors[layer][row_ind, col_ind] = H_solution[f'partial_matrix_product_{layer}_{row_ind}_{col_ind}']
                else:
                    partial_matrix_products_tensors[layer][row_ind, col_ind] = qubo_vars[f'partial_matrix_product_{layer}_{row_ind}_{col_ind}']

    if not args.bnn_as_quso_:
        aux_matrix_products_tensors = {}
        for layer in range(num_layers):
            row_size, col_size = aux_matrix_product_sizes[layer]
            aux_matrix_products_tensors[layer] = torch.zeros(row_size, col_size)
            for row_ind in range(row_size):
                for col_ind in range(col_size):
                    aux_matrix_products_tensors[layer][row_ind, col_ind] = \
                        H_solution[f'aux_matrix_product_{layer}_{row_ind}_{col_ind}']

    tensors = {'aux_tensor': aux_tensor,
               'tau_tensor': tau_tensor,
               'matrix_products_tensors': matrix_products_tensors,
               'partial_matrix_products_tensors': partial_matrix_products_tensors}
    if not args.bnn_as_quso_:
        tensors['aux_matrix_products_tensors'] = aux_matrix_products_tensors

    H_layout = [matrix_products_tensors[k].tolist() for k in matrix_products_tensors]
    H_layout.append(int(not sample_input_target.item()))
    nn_layout = [to_boolean(o).detach().squeeze().tolist() for o in new_class.nn_lay_outs]
    if all_constraints_satisfied and H_layout != nn_layout:
        print('-------------H and nn layers output don\'t match---------')
        print(H_layout,  nn_layout)
        raise Exception('H and nn layers output don\'t match')


def run_solver(model, sample_input, sample_input_target, H, args, H_dimod, qubo_vars):
    H_solution, H_best_value, pct_valid_solutions, avg_H_value = exec_solver(H, args, H_dimod)
    all_constraints_satisfied = H.is_solution_valid(H_solution)
    print("Constraints satisfied?", all_constraints_satisfied)
    tau_vec_size = sample_input.numel()
    tau_tensor = torch.zeros(tau_vec_size)
    for col_ind in range(tau_vec_size):
        if col_ind not in args.pixels_to_perturb:
            tau_i = 0
        else:
            tau_i = H_solution[f'tau_{col_ind}']
        if args.bnn_as_quso_:
            tau_i = to_boolean(-1*tau_i)  # FIXME why -1?
        tau_tensor[col_ind] = tau_i
    sum_taus = sum(tau_tensor).item()
    print('sum(taus)', sum_taus)
    if sum_taus == 0:
        print('------------UH-OH! sum(taus) == 0! ----------')
    # FIXME For quso
    new_class = model(-1 * sample_input * to_spin(tau_tensor), keep_intermediate=True)
    orig_class = int(sample_input_target)
    old_class = model(sample_input, keep_intermediate=True)
    print('target class:', orig_class, 'old class:', old_class.item(), 'new class:', new_class.item())
    is_adversarial = old_class.item() != new_class
    if args.check_solutions_match_tensors:
        check_solutions_match_tensors(model, old_class, new_class, H_solution, tau_tensor, sample_input_target,
                                      qubo_vars, all_constraints_satisfied, args)
    if all_constraints_satisfied and not is_adversarial:
        raise Exception('Adversarial example found, but it is not adversarial')
    perturbation_is_bounded = sum_taus < args.epsilon
    print('perturbation_is_bounded', perturbation_is_bounded)

    all_checks_passed = all_constraints_satisfied and is_adversarial and perturbation_is_bounded
    print('*** all_checks_passed ***', all_checks_passed)
    wandb.log({'all_constraints_satisfied': all_constraints_satisfied,
               'H_best_value': H_best_value,
               'pct_valid_solutions': pct_valid_solutions,
               'sum_taus': sum_taus,
               'avg_H_value': avg_H_value,
               'is_adversarial': is_adversarial,
               'perturbation_is_bounded': perturbation_is_bounded,
               'all_checks_passed': all_checks_passed})
    return H_solution, all_checks_passed


def visualize_result(model, sample_input_spin, sample_input_target, H_solution, shape, image_index, args):
    tau_list = []
    for i in range(len(sample_input_spin)):
        if i in args.pixels_to_perturb:
            tau_list.append(H_solution[f'tau_{i}'])
        else:
            tau_list.append(0)
    tau_tensor_ = torch.Tensor(tau_list)
    tau_tensor = tau_tensor_[:args.shape_2d[0] * args.shape_2d[1]]

    perturbation = tau_tensor.reshape(args.shape_2d)

    print('sum tau after cutting', sum(tau_tensor).item())
    sample_input_boolean_ = to_boolean(sample_input_spin)
    sample_input_boolean = sample_input_boolean_[:args.shape_2d[0] * args.shape_2d[1]]

    perturbed_image_ = torch.logical_xor(sample_input_boolean_, tau_tensor_).type(torch.FloatTensor) * 2 - 1
    perturbed_image = torch.logical_xor(sample_input_boolean, tau_tensor).type(torch.FloatTensor).reshape(
        args.shape_2d) * 2 - 1
    print('original image - ground truth:', int(sample_input_target))

    plt.imshow(sample_input_boolean.reshape(args.shape_2d), cmap='gray')
    plt.savefig(f'./imgs/zz_{args.run_id}-{image_index}-original.png')
    print('perturbation')
    plt.imshow(perturbation, cmap='gray')
    plt.savefig(f'./imgs/zz_{args.run_id}-{image_index}-perturbation.png')
    print('perturbed image - perturbed class:', model(perturbed_image_).item())
    plt.imshow(perturbed_image, cmap='gray')
    plt.savefig(f'./imgs/zz_{args.run_id}-{image_index}-perturbed.png')
