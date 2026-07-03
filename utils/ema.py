import torch


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                shadow_param = self.shadow[name].to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * shadow_param
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device).clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].to(param.device).clone()
        self.backup = {}

    def state_dict(self):
        return {name: param.clone() for name, param in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {}
        model_param_dict = dict(self.model.named_parameters())
        for name, param in model_param_dict.items():
            if param.requires_grad and name in state_dict:
                tensor = state_dict[name]
                if torch.is_tensor(tensor):
                    self.shadow[name] = tensor.detach().clone().to(param.device)
        for name, param in model_param_dict.items():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()

if __name__ == '__main__':
    model = None
    optimizer = None
    ema = EMA(model, 0.999)
    ema.register()


    def train():
        optimizer.step()
        ema.update()


    def evaluate():
        ema.apply_shadow()
        ema.restore()
