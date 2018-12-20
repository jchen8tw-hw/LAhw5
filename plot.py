from main import*
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotting(train_set_loss,test_set_loss,file='./test.png'):
    assert len(train_set_loss) == len(test_set_loss)
    length = len(train_set_loss)
    plt.figure(figsize = (12,8))
    plt.gcf().clear()
    plt.xlabel('N')
    plt.ylabel('MSE loss')
    plt.plot(list(range(1,length+1)),train_set_loss,'b',label = 'train loss')
    plt.plot(list(range(1,length+1)),test_set_loss,'r',label = 'test loss')
    plt.xticks(list(range(1,length+1)))
    plt.legend()
    
    plt.savefig(file)



if __name__ == '__main__' :
    loss_trains = []
    loss_tests = []
    for N in range(1,49):
        train_X, train_Y = read_TrainData('train.csv', N=N)
        model = Linear_Regression()
        model.train(train_X, train_Y)
        test_X, test_Y = read_TestData('test.csv', N=N)
        Predict_train = model.predict(train_X)
        Predict_test = model.predict(test_X)
        loss_train = MSE(Predict_train, train_Y)
        loss_test = MSE(Predict_test,test_Y)
        loss_trains.append(loss_train)
        loss_tests.append(loss_test)
    plotting(loss_trains,loss_tests)


