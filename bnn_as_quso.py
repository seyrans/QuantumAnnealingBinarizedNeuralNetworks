# !pip install qubovert -q
import qubovert as qv
from qubovert import spin_var
from IPython.display import display, Math
from binarized_neural_net import binarize_to_spin
from utils import to_spin, to_boolean
import wandb


def log_H(H, name):
    # H_qubo = H.to_qubo()
    # wandb.log({f'num_variables_{name}': len(H.variables),
    #            f'qubo_num_terms_{name}': H_qubo.num_terms,
    #            f'qubo_num_binary_variables_{name}': H_qubo.num_binary_variables,
    #            f'qubo_num_quad_terms_{name}': H_qubo.num_terms - H_qubo.num_binary_variables
    # })
    pass


def latex(model):
    text = model.pretty_str().replace('x', r'').replace('(', '').replace(')', '').replace('__a', 'a_')
    display(Math(text))


def setup_optim_model(sample_input_spin, sample_input_target,  model, args):
    LAMBDA = args.LAMBDA
    epsilon = args.epsilon
    quso_vars = {}  # dict of variables
    H = qv.PCSO()

    num_layers = len(model.fc_list)

    gt = to_spin(int(sample_input_target))

    sum_taus = 0
    for i in range(len(sample_input_spin)):
        quso_vars[f'tau_{i}'] = spin_var(f'tau_{i}')
        # Count number of -1s
        sum_taus += to_boolean(- 1 * quso_vars[f'tau_{i}'])

    if args.optimize_taus:
        H += LAMBDA['sum_taus'] * sum_taus

    H.add_constraint_lt_zero(sum_taus - epsilon, LAMBDA['perturbation_bound_constraint'])

    for layer in range(num_layers):
        if layer == 0:
            fc_in = sample_input_spin

        # loop over output dimension of layer
        for j in range(getattr(model, 'fc_list')[layer].weight.shape[0]):
            partial = 0.
            # loop over input dimension of layer
            for i in range(getattr(model, 'fc_list')[layer].weight.shape[1]):
                weight = binarize_to_spin(getattr(model, 'fc_list')[layer].weight[j][i]).item()
                quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] = spin_var(
                    f'partial_matrix_product_{layer}_{i}_{j}')
                if layer == 0:
                    # the first layer is dependent on tau
                    if fc_in[i] == 1:
                        if weight == 1:
                            H.add_constraint_eq_zero(
                                quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] - quso_vars[f'tau_{i}'],
                                lam=LAMBDA['hard_constraints']
                            )
                        elif weight == -1:
                            H.add_constraint_eq_zero(
                                quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] + quso_vars[f'tau_{i}'],
                                lam=LAMBDA['hard_constraints']
                            )
                        else:
                            raise ValueError
                    if fc_in[i] == -1:
                        if weight == 1:
                            H.add_constraint_eq_zero(
                                quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] + quso_vars[f'tau_{i}'],
                                lam=LAMBDA['hard_constraints']
                            )
                        elif weight == -1:
                            H.add_constraint_eq_zero(
                                quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] - quso_vars[f'tau_{i}'],
                                lam=LAMBDA['hard_constraints']
                            )
                        else:
                            raise ValueError
                else:
                    if weight == -1:
                        H.add_constraint_eq_zero(
                            quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] + fc_in[i],
                            lam=LAMBDA['hard_constraints']
                        )
                    elif weight == 1:
                        H.add_constraint_eq_zero(
                            quso_vars[f'partial_matrix_product_{layer}_{i}_{j}'] - fc_in[i],
                            lam=LAMBDA['hard_constraints']
                        )
                    else:
                        raise ValueError
                partial += quso_vars[f'partial_matrix_product_{layer}_{i}_{j}']
            quso_vars[f'matrix_product_{layer}_{j}'] = spin_var(f'matrix_product_{layer}_{j}')

            # the activation function is a threshold
            if layer == len(model.fc_list) - 1:
                output_spin = -1 * gt
            else:
                output_spin = quso_vars[f'matrix_product_{layer}_{j}']
            H.add_constraint_gt_zero(output_spin * partial)

        fc_in = [quso_vars[f'matrix_product_{layer}_{j}'] for j in
                 range(getattr(model, 'fc_list')[layer].weight.shape[0])]
    return H
