import gurobipy as grb
import matplotlib.pyplot as plt
import numpy as np


def set_up_grb_model_qubo(target_qubo_0, const, args):
    const = const or 0
    grb_model = grb.Model()
    grb_model.setParam('OutputFlag', False)
    grb_model.setParam('Threads', args.gurobi_threads)

    # First add the input variables as Gurobi variables.
    gurobi_vars = []
    v = grb_model.addVar(vtype=grb.GRB.BINARY, name=f'x')
    grb_model.update()
    gurobi_vars.append(v)

    grb_loss = 0
    grb_model.update()

    for (i, j) in target_qubo_0:
        v = grb_model.getVarByName(f"{i}")
        if v is None:
            v = grb_model.addVar(vtype=grb.GRB.BINARY, name=f'{i}')
            gurobi_vars.append(v)
        grb_model.update()
        w = grb_model.getVarByName(f"{j}")
        if w is None:
            w = grb_model.addVar(vtype=grb.GRB.BINARY, name=f'{j}')
            gurobi_vars.append(w)
        grb_model.update()
        grb_loss += v * w * target_qubo_0[(i, j)]
        grb_model.update()
    grb_loss += const
    grb_model.update()
    return grb_model, gurobi_vars, grb_loss


grb_train = []
grb_time = float('inf')


def mycallback_time(model, where):
    if where == grb.GRB.Callback.MIP:
        _time = model.cbGet(grb.GRB.Callback.RUNTIME)
        best = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)

        grb_train.append((_time, best, cur_bd))
        if _time > grb_time and best < grb.GRB.INFINITY:
            model.terminate()
            print(_time, best, cur_bd)


def fix_zero(x):
    outx = []
    for i in x:
        if i == 0:
            outx.append(-1)
        else:
            outx.append(i)
    return outx


def mycallback_plot(model, where, ):
    if where == grb.GRB.Callback.MIP:
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        best = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time_ = int(time)
        global print_time_
        if print_time_ != time_:
            print(time, best, cur_bd)
            print_time_ = time_
        grb_train.append((time, best, cur_bd))
        if time > grb_time:
            model.terminate()

    if where == grb.GRB.Callback.MIPSOL:
        # MIP solution callback
        solution = model.cbGetSolution(model._Model__vars)
        plt.figure()
        plt.imshow(np.array([fix_zero(solution)]), cmap='gray')


print_time_ = -1