import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from simple_cnn import CNN
from binarized_neural_net import BinaryNet
from binarized_neural_net2 import BinaryNet2
from binary_optimizer import MomentumWithThresholdBinaryOptimizer
import wandb


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.binary_cross_entropy(output, target)
        # loss = F.binary_cross_entropy_with_logits(output, target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx != 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.type(torch.FloatTensor).to(device)
            data = data.type(torch.FloatTensor).to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output
            if len(output.shape) != 1:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    return test_loss, test_acc


def train_model(train_dataloader, test_dataloader, args):
    # Training settings
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # initialize the NN
    s = train_dataloader.dataset[0][0].shape
    # input_size = s[-1] * s[-2]
    input_size = s[0]
    hidden_dim = args.hidden_dim  # num hidden layers
    args.num_classes = 2  # binary
    args.num_layers  # num layers
    if args.use_binary_net2:
        model = BinaryNet2(input_size, hidden_dim, args.num_layers, args.num_classes).to(device)
    else:
        model = BinaryNet(input_size, hidden_dim, args.num_layers, args.num_classes).to(device)
    # model = CNN().to(device)
    model_num_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    print('Total Number of parameters of Model:', model_num_params)
    wandb.log({'model_num_params': model_num_params})
    if args.optimizer == 'binop':
        optimizer = MomentumWithThresholdBinaryOptimizer(
            model.parameters(), {}, ar=1e-9, threshold=1e-9
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = optimizer
    if args.use_scheduler:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_dataloader)
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
        scheduler.step()
    wandb.log({'last_test_loss': test_loss, 'last_test_acc': test_acc})
    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")
    return model
