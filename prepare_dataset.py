import collections
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import collections
from utils import to_boolean, to_spin
import wandb


def download_datasets(binarize=False):
    transform = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if binarize:
        transform += [lambda x: x > 0, lambda x: x.float()]

    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.Compose(transform))
    test_dataset = datasets.MNIST('./mnist', train=False, download=True, transform=transforms.Compose(transform))

    return train_dataset, test_dataset


def subset_datasets(dataset, selected_targets=(0, 1)):
    if len(selected_targets) != 2:
        raise ValueError("selected_targets should be two numbers")

    num_1, num_2 = selected_targets[0], selected_targets[1]
    target_indices = np.arange(len(dataset))
    selected_targets_tensors = (dataset.targets == num_1).logical_or(dataset.targets == num_2)
    selected_subset = torch.utils.data.Subset(dataset, target_indices[selected_targets_tensors])
    return selected_subset


def remove_contradicting_dataset(selected_dataset):
    selected_data_tensor = selected_dataset.tensors[0]
    selected_target_tensor = selected_dataset.tensors[1]
    selected_data_numpy, selected_target_numpy = remove_contradicting_numpy(
        selected_data_tensor.numpy(),
        selected_target_tensor.numpy()
    )
    selected_data_tensor = torch.Tensor(selected_data_numpy)
    selected_target_tensor = torch.Tensor(selected_target_numpy)
    selected_dataset = torch.utils.data.TensorDataset(selected_data_tensor, selected_target_tensor)
    return selected_dataset


def remove_contradicting_numpy(xs, ys):
    # Borrowed from TF-Quantum Tutorials
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label.
            pass

    num_uniq_1 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_2 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
    print("Initial number of images: ", len(xs))
    print("Number of unique images:", len(mapping.values()))
    print(f"Number of unique num_0s: ", num_uniq_1)
    print(f"Number of unique num_1s: ", num_uniq_2)
    wandb.log({"uniq_num_0s": num_uniq_1})
    wandb.log({"uniq_num_1s": num_uniq_2})
    print(f"Number of unique contradicting labels (both num_1 and num_2): ", num_uniq_both)
    print("Remaining non-contradicting unique images: ", len(new_x))
    print()
    return np.array(new_x), np.array(new_y)


def avg_pool_binarizer(x, args):
    """
    if use_adaptive is passed, this will use AdaptiveAvgPool2d.
    """
    if args.downscale:
        if args.use_adaptive:
            avg_pooler = nn.AdaptiveAvgPool2d(args.adaptive_avg_pool_shape)
        else:
            avg_pooler = nn.AvgPool2d(args.avg_pool_kernel_size, stride=args.avg_pool_stride)
        x = avg_pooler(x)
    x = x > 0.5
    if args.flatten_dataset:
        flatten_size = x.shape[-1] * x.shape[-2]
        x = x.view(-1, flatten_size)
        if args.pad_flattened_dataset:
            import math
            base2_log = math.log2(flatten_size + 1)
            if int(base2_log) != base2_log:
                pad_size = (2**(int(math.log2(flatten_size))+1)-1) - flatten_size  # convert to 2**n - 1 form
                x = nn.ConstantPad1d((0, pad_size), 0)(x)
    if args.dataset_spin_form:
        return to_spin(x)
    return x


def avg_pool_binarize_dataset(dataset, args, test=False):
    dataset_small = []
    targets_train = []
    for data, target in dataset:
        out = avg_pool_binarizer(data, args)
        dataset_small += list(out)
        target = (target == args.selected_targets[0])
        targets_train += list(target)
    if args.partial_training_dataset and not test:
        dataset_small = dataset_small[:args.partial_training_dataset]
        targets_train = targets_train[:args.partial_training_dataset]
    # FIXME: there are nicer ways to do stacking!
    targets_train = torch.stack((targets_train)).type(torch.FloatTensor)
    mnist_sm_bn_train = torch.stack((dataset_small)).type(torch.FloatTensor)
    return torch.utils.data.TensorDataset(mnist_sm_bn_train, targets_train)


def dataloaders(args):
    train_dataset_28, test_dataset_28 = download_datasets()
    selected_dataset_train = subset_datasets(train_dataset_28, args.selected_targets)
    selected_dataset_test = subset_datasets(test_dataset_28, args.selected_targets)
    selected_train_dataloader = torch.utils.data.DataLoader(selected_dataset_train,
                                                            batch_size=args.downscale_batch_size, shuffle=args.shuffle)
    selected_test_dataloader = torch.utils.data.DataLoader(selected_dataset_test,
                                                           batch_size=args.downscale_batch_size, shuffle=args.shuffle)
    if args.visualize_dataset:
        visualize_dataset(selected_train_dataloader, args, postfix='a')
    selected_dataset_train = avg_pool_binarize_dataset(selected_train_dataloader, args)
    selected_dataset_test = avg_pool_binarize_dataset(selected_test_dataloader, args, test=True)
    if args.remove_contradicting:
        print('Removing contradicting images from Training set')
        selected_dataset_train = remove_contradicting_dataset(selected_dataset_train)
        print('Removing contradicting images from Test set')
        selected_dataset_test = remove_contradicting_dataset(selected_dataset_test)

    print('Training set shape:', list(selected_dataset_train[0][0].shape))
    train_dataloader = torch.utils.data.DataLoader(selected_dataset_train, batch_size=args.batch_size, shuffle=args.shuffle)
    test_dataloader = torch.utils.data.DataLoader(selected_dataset_test, batch_size=args.batch_size, shuffle=args.shuffle)
    if args.visualize_dataset:
        visualize_dataset(train_dataloader, args, postfix='b')
    return train_dataloader, test_dataloader


def visualize_dataset(dataloader, args, postfix=''):
    for i in range(min(10, args.partial_training_dataset)):
        image = dataloader.dataset[i][0].squeeze()
        target = dataloader.dataset[i][1]
        if postfix == 'b':
            image = image[:args.shape_2d[0] * args.shape_2d[1]]
            image = image.reshape(args.shape_2d)
            target = not target
        plt.imshow(image, cmap='gray')
        from pathlib import Path
        Path("./imgs/").mkdir(parents=True, exist_ok=True)
        plt.savefig(f'./imgs/{i}-{postfix}-label_{int(target)}.png')
