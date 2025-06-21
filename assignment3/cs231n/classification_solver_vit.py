import torch
from tqdm.auto import tqdm


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cpu'):
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.3f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num))

    return total_loss / total_num, total_correct_1 / total_num, total_correct_5 / total_num


class ClassificationSolverViT:
    def __init__(self, train_data, test_data, model, **kwargs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

        # Unpack keyword arguments
        self.learning_rate = kwargs.pop("learning_rate", 1.e-4)
        self.weight_decay = kwargs.pop("weight_decay", 0.0)
        self.batch_size = kwargs.pop("batch_size", 64)
        self.num_epochs = kwargs.pop("num_epochs", 2)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.loss_criterion = torch.nn.CrossEntropyLoss()

        self._reset()

    def _reset(self):

        self.epoch = 0
        self.results = {'train_loss': [], 'train_acc@1': [], 'test_loss': [], 'test_acc@1': []}

    def train(self, device='cpu'):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        self.model.to(device)
        self.loss_criterion.to(device)

        best_acc = 0.0
        for epoch in range(self.num_epochs):
            train_loss, train_acc_1, _ = train_val(self.model, train_loader, self.optimizer, epoch, self.num_epochs, device)
            self.results['train_loss'].append(train_loss)
            self.results['train_acc@1'].append(train_acc_1)

            test_loss, test_acc_1, _ = train_val(self.model, test_loader, None, epoch, self.num_epochs, device)
            self.results['test_loss'].append(test_loss)
            self.results['test_acc@1'].append(test_acc_1)
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
        
        self.results["best_test_acc"] = best_acc

