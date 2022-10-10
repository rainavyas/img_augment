from .model_selector import model_sel
from ..training.trainer import Trainer

class Ensemble():
    def __init__(self, model_name, model_paths, device, num_classes=10):
        self.models = []
        for mpath in model_paths:
            model = model_sel(model_name, mpath, num_classes=num_classes)
            model.to(device)
            self.models.append(model)

    def eval(self, dl, criterion, device):
        '''
        Evaluate Ensemble predictions
        Returns list of accuracies
        '''
        accs = []
        for m in self.models:
            acc = Trainer.eval(dl, m, criterion, device)
            accs.append(acc)
        return accs