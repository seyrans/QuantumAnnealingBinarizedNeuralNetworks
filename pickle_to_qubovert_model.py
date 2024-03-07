import qubovert as qv
from qubovert import boolean_var

import pickle
with open('pickles/50-322-4-7-sum_taus-2-3-113261-H.pickle', 'rb') as f:
    h = pickle.load(f)


h_new = qv.PCBO()
objective_lambda = 0.1


def build_term_from_constraint(constraint):
    term = qv.PUBO()
    for pair_bools, coefficient in constraint.items():
        if len(pair_bools) == 0:
            term += coefficient
        elif len(pair_bools) == 1:
            term += boolean_var(pair_bools[0]) * coefficient
        else:
            term += boolean_var(pair_bools[0]) * boolean_var(pair_bools[1]) * coefficient
    return term


# ----- Add less than constraints and objective -----
for constraint in h['constraints']['lt']:
    left_term = build_term_from_constraint(constraint)
    h_new.add_constraint_lt_zero(left_term)
    # ---- In this problem, the objective is the same as 0.1 * h['constraints']['lt']
    h_new += objective_lambda * left_term

# ----- Add equal zero constraints -----
for constraint in h['constraints']['eq']:
    left_term = build_term_from_constraint(constraint)
    h_new.add_constraint_eq_zero(left_term)

assert h_new.to_qubo().Q == h['qubo']
