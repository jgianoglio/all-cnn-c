"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import hw2.all_cnn
from hw2 import preprocessing as P
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))
            
class imgDataset(torch.utils.data.Dataset):
    """Utternace dataset."""

    def __init__(self, dataType):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        xTrain = np.load("dataset/train_feats.npy")
        xTest = np.load("dataset/test_feats.npy")
        train_labels = np.load("dataset/train_labels.npy")
        #test_labels = np.zeros(10000)
        
        # Preprocess training and test data to normalize
        xTrain, xTest = P.cifar_10_preprocess(xTrain, xTest, image_size=32)
                
        xTest = xTrain[:1000]
        xTrain = xTrain[1000:11000]
        yTest = train_labels[:1000]
        yTrain = train_labels[1000:11000]
        
        if dataType == 1:
            self.trainX = xTrain
            self.trainY = yTrain
        if dataType == 2:
            self.trainX = xTest
            self.trainY = yTest

    def __len__(self):
        return len(self.trainX)

    def __getitem__(self, idx):
        sample = self.trainX[idx], self.trainY[idx]

        return sample
    
def init_xavier(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        
def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)
            
def main(batch_size=100, learning_rate=0.0001, epochs=2, log_interval=10): 
    
    # USE THE FOLLOWING FOR TESTING
    batch_size=64
    learning_rate=0.0001
    epochs=2
    log_interval=10
    
    
    # create dataset
    dataset = imgDataset(1) # Train data -- 40K of original 50K train data
    datasetTest = imgDataset(2) # Test Data -- 10K of orginal train data
    
    # Create dataLoader -- for GPU, include pim_memory and num_workers
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
        )
    
    test_loader = torch.utils.data.DataLoader(
        datasetTest,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
        )
    
    model = hw2.all_cnn.all_cnn_module()
    model.apply(init_xavier)
    
        # create a stochastic gradient descent optimizer
    optimizer = optim.Adam(model.parameters(),
                          lr=learning_rate,
                          weight_decay=0.001)
    # create a loss function
    criterion = torch.nn.CrossEntropyLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float()
            target = target.long()
            data, target = to_variable(data), to_variable(target)
            optimizer.zero_grad()
            model_out = model(data)
            loss = criterion(model_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(model.state_dict(), './model1.pt')

    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.float()
        target = target.long()
        data, target = Variable(data, volatile=True), Variable(target)
        model_out = model(data)
        # sum up batch loss
        test_loss += criterion(model_out, target).data[0]
        pred = model_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
