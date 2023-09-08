import os
import torch
from utils import progress_bar


class ModelRunner:
    def __init__(self, net, trainloader, testloader, device, optimizer, criterion):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.best_acc = 0

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = 100.*correct/total
        print(f'Train accuracy: {acc}')

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Save checkpoint.
        acc = 100.*correct/total
        print(f'Test accuracy: {acc}')
        if acc > self.best_acc:
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            print(f'SAVING! Previous best accuracy: {self.best_acc}. New best accuracy: {acc}')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            self.best_acc = acc
