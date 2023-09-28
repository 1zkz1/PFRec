import torch
import torch.nn as nn
import torch.optim as optim
import os


class ModelUtil:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, epochs: int, name='model',
                 path='models'):
        super(ModelUtil, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.name = name
        self.path = path

    # load torchscript
    @classmethod
    def load_ts(cls, name='model', path='models'):
        print('load model {}/{}.pt'.format(path, name))

        return torch.jit.load('{}/{}.pt'.format(path, name))

    def path_is_exist(self):
        folder = os.path.exists(self.path)
        if not folder:
            print('models directory is not exist! mkdir the directory.')
            os.makedirs(self.path)

        return

    def save_model(self):
        self.path_is_exist()

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epochs': self.epochs
        }, '{}/{}.pth'.format(self.path, self.name))
        print('save model:{} to {}/{}.pth'.format(self.name, self.path, self.name))

        return

    def load_model(self, name=None, path=None):
        name = name if name is not None else self.name
        path = path if path is not None else self.path
        print('load model {}/{}.pth'.format(path, name))
        checkpoint = torch.load('{}\{}.pth'.format(path, name))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs = checkpoint['epochs']

        return self.model, self.optimizer

    # convert model to TorchScript. you can use model directly(not need init model).
    def save_to_ts(self, example:torch.Tensor):
        self.path_is_exist()

        # net = torch.jit.script(self.model)

        torch.jit.trace(self.model, example).save('{}/{}.pt'.format(self.path, self.name))

        print('save model:{} to {}/{}.pt'.format(self.name, self.path, self.name))

        return

    def load_from_ts(self):
        self.path_is_exist()
        print('load model {}/{}.pt'.format(self.path, self.name))

        return torch.jit.load('{}/{}.pt'.format(self.path, self.name))




