import torch
from torch import nn
from main import*
import torch.utils.data as Data

N = 5

def getTrainData():
    train_X, train_Y = read_TrainData('train.csv', N=1)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_X),torch.FloatTensor(train_Y))
    loader = Data.DataLoader(
    dataset=torch_dataset,      
    batch_size= N,      
    shuffle=False,               
    num_workers=8,)
    return loader


def getTestData():
    test_X, test_Y = read_TestData('test.csv', N=1)
    return torch.FloatTensor(test_X),torch.FloatTensor(test_X)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(         
            input_size=19,
            hidden_size=19,         
            num_layers=1,
        )
        self.out = nn.Linear(19, 1)

    def forward(self, x):
        r_out, h_c = self.rnn(x, None)
        out = self.out(r_out)
        return out





loss_func = torch.nn.MSELoss()
model = RNN()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5)

epoch = 500
loader = getTrainData()
test_X, test_Y = getTestData()
for epoch_ in range(epoch):
    for step, (b_x, b_y) in enumerate(loader):       
        b_x = torch.unsqueeze(b_x,dim = 1)
        output = model(b_x)
        output = torch.squeeze(output,dim=1)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()                           
        loss.backward()                                 
        optimizer.step()


        if step % 50 == 0:
            test_output = model(torch.unsqueeze(test_X,dim=1))   
            test_output = torch.squeeze(test_output,dim=1) 
                 # (time_step(N), 1 , input_size)
            test_loss = np.mean(((test_output.detach().numpy()-torch.unsqueeze(test_Y[:,9],dim=1).detach().numpy())**2))
            print('Epoch: ', epoch_, '| train loss: %.4f' % loss.data.numpy(), '| test loss: %.2f' % test_loss)



    
    
    