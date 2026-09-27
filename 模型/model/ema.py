class EMA:
    def __init__(self, model, decay):
        self.decay = decay  # 衰减率，用于更新shadow参数的更新速度
        self.shadow = {}    # 为每个可训练参数创建一个shadow参数，shadow参数的初始值与模型参数相同
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    # 每次调用时，根据当前模型参数和shadow参数，更新shadow参数
    def __call__(self, model, num_updates=99999):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]