import random
from prepare_dataset import dataloaders
from training_logic import train_model
import bnn_as_qubo
import bnn_as_quso
import bnn_as_qubo_dimod
from annealer import run_solver, visualize_result
from get_args import args, iterable_args
import wandb
import pickle


def export_H(H, args):
    if args.export_H_as_pickle:
        filepath = ('./pickles/' +
                    str(len(H.variables)) + '-' +
                    str(H.to_qubo().num_terms) + '-' +
                    str(args.adaptive_avg_pool_shape) + '-' +
                    str(len(args.pixels_to_perturb)) + '-' +
                    str(args.objective) + '-' +
                    str(args.num_layers) + '-' +
                    str(args.hidden_dim) + '-' +
                    str(args.run_id)
                    )
        with open(filepath + '-H.pickle', 'wb') as f:
            pickle.dump({'constraints': {'lt': [dict(d) for d in H.constraints['lt']],
                                         'eq': [dict(d) for d in H.constraints['eq']]},
                         'qubo': H.to_qubo().Q}, f)
        with open(filepath + '-args.pickle', 'wb') as f:
            pickle.dump(args, f)


def main(args):
    config = vars(args)
    args.run_id = random.randint(0, 1000000)
    run = wandb.init(reinit=True,
                     save_code=True,
                     project=args.wandb_project_name,
                     config=config,
                     mode="online" if args.report_to_wandb else "disabled")
    train_dataloader, test_dataloader = dataloaders(args)

    if args.pixels_to_perturb is None:
        # Hacky way to select pixels to perturb
        class_0 = train_dataloader.dataset.tensors[0][~train_dataloader.dataset.tensors[1].to(bool)].mean(axis=0)
        class_1 = train_dataloader.dataset.tensors[0][train_dataloader.dataset.tensors[1].to(bool)].mean(axis=0)
        args.pixels_to_perturb = list((class_0 - class_1)
                                      .abs()
                                      .topk(min(args.pixels_to_perturb_len,
                                                train_dataloader.dataset.tensors[0].shape[1]),
                                            largest=True)
                                      .indices.numpy())
    print(args)

    model = train_model(train_dataloader, test_dataloader, args)

    if args.sample_output:
        samples = next(iter(train_dataloader))
        sample_pred = model(samples[0])
        sample_target = samples[1]
        print(sample_pred)
        print(sample_target)
        print((1-(sample_pred - sample_target).sum()/len(sample_pred)) * 100)

    if not args.only_train:
        pct_all_checks_passed = 0
        all_checks_passed = False
        image_index = 0
        if args.starting_image_index:
            image_index = args.starting_image_index
        num_images_solved = 0
        num_images = len(test_dataloader)
        if args.num_images_to_solve:
            num_images = args.num_images_to_solve
        while num_images_solved < num_images:
            print('image_index:', image_index)
            wandb.log({'image_index': image_index})
            sample_input_item = test_dataloader.dataset[image_index]
            sample_input_spin = sample_input_item[0]
            sample_input_target = sample_input_item[1]
            shape = sample_input_spin.shape[-2:]
            if model(sample_input_spin) != sample_input_target:
                # if the model already predicts the sample with an incorrect class, let's skip this sample.
                continue
            H_dimod = None
            qubo_vars = None
            if args.bnn_as_qubo_dimod_:
                H_dimod = bnn_as_qubo_dimod.setup_optim_model(sample_input_spin, sample_input_target,
                                                              model, args)
            if args.bnn_as_quso_:
                H = bnn_as_quso.setup_optim_model(sample_input_spin, sample_input_target,
                                                  model, args)
            else:
                H, qubo_vars = bnn_as_qubo.setup_optim_model(sample_input_spin, sample_input_target,
                                                             model, args)
            export_H(H, args)
            if not args.no_solver:
                H_solution, all_checks_passed = run_solver(model, sample_input_spin, sample_input_target, H, args, H_dimod, qubo_vars)
                if H_solution and args.visualize_result_ and all_checks_passed:
                    visualize_result(model, sample_input_spin, sample_input_target, H_solution, shape, image_index, args)
            num_images_solved += 1
            pct_all_checks_passed += int(all_checks_passed)
            image_index += 1
        pct_all_checks_passed /= num_images_solved
        wandb.log({'pct_all_checks_passed': pct_all_checks_passed,
                   'num_images_solved': num_images_solved})
    run.finish()


def run():
    for selected_targets in iterable_args.selected_targets:
        args.selected_targets = selected_targets
        for adaptive_avg_pool_shape in iterable_args.adaptive_avg_pool_shape:
            args.adaptive_avg_pool_shape = adaptive_avg_pool_shape
            for num_layers in iterable_args.num_layers:
                args.num_layers = num_layers
                for hidden_dim in iterable_args.hidden_dim:
                    args.hidden_dim = hidden_dim
                    for bnn_as_quso_ in iterable_args.bnn_as_quso_:
                        args.bnn_as_quso_ = bnn_as_quso_
                        for pixels_to_perturb_len in iterable_args.pixels_to_perturb_len:
                            args.pixels_to_perturb_len = pixels_to_perturb_len
                            for objective in iterable_args.objectives:
                                args.objective = objective
                                for include_perturbation_bound_constraint in iterable_args.include_perturbation_bound_constraint:
                                    args.include_perturbation_bound_constraint = include_perturbation_bound_constraint
                                    for epsilon in iterable_args.epsilon:
                                        args.epsilon = epsilon
                                        for LAMBDA in iterable_args.LAMBDA:
                                            args.LAMBDA = LAMBDA
                                            for solver in iterable_args.solver:
                                                args.solver = solver

                                                args.num_classes = len(iterable_args.selected_targets)
                                                if args.use_adaptive and args.downscale:
                                                    args.shape_2d = (args.adaptive_avg_pool_shape, args.adaptive_avg_pool_shape)
                                                main(args)

                                                # try:
                                                #     print(args)
                                                #     main(args)
                                                # except Exception as e:
                                                #     print('>>>>> Exception has occurred.')
                                                #     print(e)
                                                # break  # solver
                                            # break  # lambda


if __name__ == '__main__':
    run()
