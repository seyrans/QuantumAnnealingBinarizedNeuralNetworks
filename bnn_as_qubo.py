# !pip install qubovert -q
import qubovert as qv
from qubovert import boolean_var
from IPython.display import display, Math
from binarized_neural_net import binarize_to_spin
from utils import to_boolean
from qubovert._pcbo import _special_constraints_le_zero, _get_bounds, num_bits
from qubovert.utils._warn import QUBOVertWarning
import wandb
import math


def log_H(H, name):
    H_qubo = H.to_qubo()
    log_dict = {
        f'num_variables_{name}': len(H.variables),
        f'qubo_num_terms_{name}': H_qubo.num_terms,
        f'qubo_num_binary_variables_{name}': H_qubo.num_binary_variables,
        f'qubo_num_quad_terms_{name}': H_qubo.num_terms - H_qubo.num_binary_variables
    }
    print(log_dict)
    wandb.log(log_dict)


def latex(model):
    text = model.pretty_str().replace('x', r'').replace('(', '').replace(')', '').replace('__a', 'a_')
    return display(Math(text))


# FIXME: Implement the following for quso
def le_zero_constraint_to_eq_zero_constraint(P, log_trick=True, bounds=None, suppress_warnings=False):
    # Inspired by PCBO.add_constraint_le_zero implementation.
    model = qv.PCBO()
    P = qv.PUBO(P)

    bounds = min_val, max_val = _get_bounds(P, bounds)
    if _special_constraints_le_zero(model, P, 1, log_trick, bounds):
        return model

    if min_val > 0:
        if not suppress_warnings:
            QUBOVertWarning.warn("Constraint cannot be satisfied")
        model += P
    elif max_val <= 0:
        if not suppress_warnings:
            QUBOVertWarning.warn("Constraint is always satisfied")
    else:
        # don't mutate the P that we put in model._constraints
        P = P.copy()
        if min_val:
            for i in range(num_bits(-min_val, log_trick)):
                v = pow(2, i) if log_trick else 1
                P[(model._next_ancilla,)] += v
                max_val += v
        model += P
        return model

    return model


def add_sign_constraint(H, count, partial_poly, output_bool, lam, k_layer, j):
    aux_num_bits = int(math.log2(count + 1)-1)
    aux = qv.integer_var(f'aux_matrix_product_{k_layer}_{j}_', aux_num_bits)
    output_bool_coef = 2 ** aux_num_bits
    H.add_constraint_eq_zero(-1 * (aux + output_bool_coef * output_bool) + partial_poly, lam=lam)


def setup_optim_model(sample_input_spin, sample_input_target, model, args):
    LAMBDA = args.LAMBDA
    epsilon = args.epsilon
    qubo_vars = {}  # dict of variables
    H = qv.PCBO()
    sample_input_boolean = to_boolean(sample_input_spin)
    num_layers = len(model.fc_list)

    gt = sample_input_target

    sum_taus = 0
    for i in args.pixels_to_perturb:
        qubo_vars[f'tau_{i}'] = boolean_var(f'tau_{i}')
        sum_taus += qubo_vars[f'tau_{i}']

    if args.objective == 'sum_taus':
        H += LAMBDA['sum_taus'] * sum_taus

    if args.include_perturbation_bound_constraint:
        # log_H(sum_taus - epsilon, 'perturbation_bound_constraint')
        H.add_constraint_lt_zero(sum_taus - epsilon, LAMBDA['perturbation_bound_constraint'])

    # log_H(H, 'H_after_perturbation_bound_constraint')
    for k_layer in range(num_layers):
        if k_layer == 0:
            fc_in = sample_input_boolean

        # loop over output dimension of layer
        for j in range(getattr(model, 'fc_list')[k_layer].weight.shape[0]):
            sum_partials = 0.
            count_partials = 0
            # loop over input dimension of layer
            i_range = range(getattr(model, 'fc_list')[k_layer].weight.shape[1])
            for i in i_range:
                weight = to_boolean(binarize_to_spin(getattr(model, 'fc_list')[k_layer].weight)[j][i].item())
                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'] = boolean_var(
                    f'partial_matrix_product_{k_layer}_{i}_{j}')
                if k_layer == 0:
                    if i not in args.pixels_to_perturb:  # tau_i is 0
                        if fc_in[i] == 1:
                            if weight == 0:
                                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'] = 0
                            elif weight == 1:
                                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'] = 1
                        if fc_in[i] == 0:
                            if weight == 0:
                                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'] = 1
                            elif weight == 1:
                                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'] = 0
                    else:
                        # the first layer does not require xnor
                        if fc_in[i] == 1:  # TODO: if i not in range(3, 6) then assume tau_i is zero and subsequent vars
                            if weight == 0:
                                H.add_constraint_eq_BUFFER(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                            elif weight == 1:
                                H.add_constraint_eq_NOT(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                        elif fc_in[i] == 0:
                            if weight == 0:
                                H.add_constraint_eq_NOT(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                            elif weight == 1:
                                H.add_constraint_eq_BUFFER(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'],
                                    qubo_vars[f'tau_{i}'],
                                    lam=LAMBDA['hard_constraints']
                                )
                else:
                    if weight == 0:
                        H.add_constraint_eq_NOT(
                            qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'],
                            fc_in[i],
                            lam=LAMBDA['hard_constraints']
                        )
                    elif weight == 1:
                        H.add_constraint_eq_BUFFER(
                            qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'],
                            fc_in[i],
                            lam=LAMBDA['hard_constraints']
                        )
                sum_partials += qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}']
                count_partials += 1
            qubo_vars[f'matrix_product_{k_layer}_{j}'] = boolean_var(f'matrix_product_{k_layer}_{j}')
            output_bool = qubo_vars[f'matrix_product_{k_layer}_{j}']
            if args.objective == 'output':
                if k_layer == len(model.fc_list) - 1:
                    if gt == 0:
                        H += -1 * output_bool * LAMBDA['output'] + 1
                    else:
                        H += output_bool * LAMBDA['output']
            else:
                if k_layer == len(model.fc_list) - 1:
                    output_bool = int(not gt)

            # log_H(H, f'H_after_partial_matrix_prod_{k_layer}_{j}_{i}')
            add_sign_constraint(H, count_partials,
                                sum_partials,
                                output_bool=output_bool,
                                lam=LAMBDA['hard_constraints'],
                                k_layer=k_layer,
                                j=j,
                                )
            # log_H(H, f'H_after_matrix_prod_{k_layer}_{j}_{i}')
        fc_in = [qubo_vars[f'matrix_product_{k_layer}_{j}'] for j in
                 range(getattr(model, 'fc_list')[k_layer].weight.shape[0])]
    log_H(H, f'H_final')
    return H, qubo_vars
