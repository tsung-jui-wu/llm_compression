import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class TokenMapper(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim, bias=False)
        
        self.model.to(args.device)

    def forward(self, one_hot_token):
        return self.model(one_hot_token)