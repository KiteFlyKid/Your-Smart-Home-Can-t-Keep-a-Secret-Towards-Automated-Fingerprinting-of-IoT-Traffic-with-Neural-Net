import torch
import torch.nn as nn
from torch.autograd import Variable


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding=nn.Embedding(9604,30)
        self.lstm = nn.LSTM(input_size-1+30,
                            hidden_size,
                            num_layers,
                            batch_first=True,dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):


        embed=self.embedding(features[:,:,1].to(torch.long))
        #embed: 64,20,30
        input=torch.cat([embed,features[:,:,1:]],dim=2)
        #input: 64,20,40
        h0 = torch.zeros(self.num_layers, input.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])  #which equals to _[0][-1]
        return out