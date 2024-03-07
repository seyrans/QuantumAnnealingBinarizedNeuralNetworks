# !pip install qubovert -q
import qubovert as qv
from dimod import Binary, BinaryQuadraticModel, Vartype
from IPython.display import display, Math
from binarized_neural_net import binarize_to_spin
from utils import to_boolean
import wandb


def log_H(H, name):
    H_qubo = H.to_qubo()[0]
    wandb.log({f'num_variables_{name}': len(H.variables),
               f'qubo_num_terms_{name}': H_qubo.num_terms,
               f'qubo_num_binary_variables_{name}': H_qubo.num_binary_variables,
               f'qubo_num_quad_terms_{name}': H_qubo.num_terms - H_qubo.num_binary_variables
    })


def latex(model):
    text = model.pretty_str().replace('x', r'').replace('(', '').replace(')', '').replace('__a', 'a_')
    return display(Math(text))


def add_constraint_greater_zero(H, count, partial_poly, num_layers, lam, k_layer, j, gt):
    # the activation function is a threshold
    if k_layer == num_layers - 1:
        output_bool = int(not gt)
    else:
        output_bool = f'matrix_product_{k_layer}_{j}'
    partial_poly = list(partial_poly.linear.items())
    aux_1 = f'aux_matrix_product_{k_layer}_{j}_1'
    aux_2 = f'aux_matrix_product_{k_layer}_{j}_2'
    aux_4 = f'aux_matrix_product_{k_layer}_{j}_4'
    aux_8 = f'aux_matrix_product_{k_layer}_{j}_8'
    aux_16 = f'aux_matrix_product_{k_layer}_{j}_16'
    aux_32 = f'aux_matrix_product_{k_layer}_{j}_32'
    if count == 3:
        aux_2 = output_bool
        if type(output_bool) != int:
            H.add_linear_equality_constraint([(aux_1, -1), (aux_2, -2)] + partial_poly, lam, 0)
        else:
            H.add_linear_equality_constraint([(aux_1, -1)] + partial_poly, lam, aux_2 * -2)
    elif count == 7:
        aux_4 = output_bool
        if type(output_bool) != int:
            H.add_linear_equality_constraint([(aux_1, -1), (aux_2, -2), (aux_4, -4)] + partial_poly, lam, 0)
        else:
            H.add_linear_equality_constraint([(aux_1, -1), (aux_2, -2)] + partial_poly, lam,  aux_4 * -4)
    elif count == 15:
        aux_8 = output_bool
        if type(output_bool) != int:
            H.add_linear_equality_constraint([(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8)] + partial_poly, lam, 0)
        else:
            H.add_linear_equality_constraint([(aux_1, -1), (aux_2, -2), (aux_4, -4)] + partial_poly, lam,
                                             aux_8 * -8)
    elif count == 31:
        aux_16 = output_bool
        if type(output_bool) != int:
            H.add_linear_equality_constraint(
                [(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8), (aux_16, -16)] + partial_poly, lam, 0)
        else:
            H.add_linear_equality_constraint(
                [(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8)] + partial_poly, lam, aux_16 * -16)
    elif count == 63:
        aux_32 = output_bool
        if type(output_bool) != int:
            H.add_linear_equality_constraint(
                [(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8), (aux_16, -16), (aux_32, -32)] + partial_poly, lam, 0)
        else:
            H.add_linear_equality_constraint(
                [(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8), (aux_16, -16)] + partial_poly, lam,
                aux_32 * -32)
    elif count == 127:
        aux_64 = output_bool
        if type(output_bool) != int:
            H.add_linear_equality_constraint(
                [(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8), (aux_16, -16), (aux_32, -32), (aux_64, -64)] + partial_poly,
                lam, 0)
        else:
            H.add_linear_equality_constraint(
                [(aux_1, -1), (aux_2, -2), (aux_4, -4), (aux_8, -8), (aux_16, -16), (aux_32, -32),
                 ] + partial_poly,
                lam, aux_64 * -64)
    else:
        raise NotImplementedError()


def setup_optim_model(sample_input_spin, sample_input_target, model, args):
    LAMBDA = args.LAMBDA
    epsilon = args.epsilon
    qubo_vars = {}  # dict of variables
    H = BinaryQuadraticModel(Vartype.BINARY)
    sample_input_boolean = to_boolean(sample_input_spin)

    num_layers = len(model.fc_list)

    gt = sample_input_target

    sum_taus = 0
    for i in range(len(sample_input_boolean)):
        qubo_vars[f'tau_{i}'] = Binary(f'tau_{i}')
        sum_taus += qubo_vars[f'tau_{i}']

    if args.optimize_taus:
        H += LAMBDA['sum_taus'] * sum_taus

    H.add_linear_inequality_constraint(list(sum_taus.linear.items()), LAMBDA['tau'], 'verf', -1*epsilon)

    for k_layer in range(num_layers):
        if k_layer == 0:
            fc_in = sample_input_boolean

        # loop over output dimension of layer
        for j in range(getattr(model, 'fc_list')[k_layer].weight.shape[0]):
            sum_partials = 0.
            # loop over input dimension of layer
            for i in range(getattr(model, 'fc_list')[k_layer].weight.shape[1]):
                weight = to_boolean(binarize_to_spin(getattr(model, 'fc_list')[k_layer].weight)[j][i].item())
                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}'] = Binary(
                    f'partial_matrix_product_{k_layer}_{i}_{j}')
                if k_layer == 0:
                    # the first layer does not require xnor
                    if fc_in[i] == 1:
                        if weight == 0:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', 1),
                                 (f'tau_{i}', 1)],
                                LAMBDA['hard_constraints'], 0
                            )
                        elif weight == 1:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', -1),
                                 (f'tau_{i}', 1)],
                                -1*LAMBDA['hard_constraints'], 0
                            )
                        else:
                            raise ValueError
                    if fc_in[i] == 0:
                        if weight == 0:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', -1),
                                 (f'tau_{i}', 1)],
                                -1 * LAMBDA['hard_constraints'], 0
                            )
                        elif weight == 1:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', 1),
                                 (f'tau_{i}', 1)],
                                LAMBDA['hard_constraints'], 0
                            )
                        else:
                            raise ValueError
                else:
                    if weight == 0:
                        if fc_in[i] == 0:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', -1)], 1, 1
                            )
                        else:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', 1)], 1, 0
                            )
                    elif weight == 1:
                        if fc_in[i] == 0:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', 1)], 1, 0
                            )
                        else:
                            H.add_linear_equality_constraint(
                                [(f'partial_matrix_product_{k_layer}_{i}_{j}', -1)], 1, 1
                            )
                    else:
                        raise ValueError
                sum_partials += qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}']
            qubo_vars[f'matrix_product_{k_layer}_{j}'] = Binary(f'matrix_product_{k_layer}_{j}')
            add_constraint_greater_zero(H, len(sum_partials),
                                        sum_partials,
                                        num_layers=num_layers,
                                        lam=LAMBDA['hard_constraints'],
                                        k_layer=k_layer,
                                        j=j,
                                        gt=gt
                                        )

        fc_in = [qubo_vars[f'matrix_product_{k_layer}_{j}'] for j in
                 range(getattr(model, 'fc_list')[k_layer].weight.shape[0])]
    return H
