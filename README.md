# EVA-Assignment-6-Regularization-and-BN

Dataset = MNIST

Number of epochs = 25

Three CNN Models were built and tested using a Single code, used a for loop to ierate through the following 3 Models
1) L2 Regularization with BN (Batch Normalization)
2) L1 Regularization with BN 
3) L1 with L2 and with BN

L2 with BN - Optimizer and Losses used
--------------------------------------
    (Weight Decay parameter used for L2) optimizer = optim.SGD(model.parameters(), lr=0.9, momentum=0.9, dampening = 0, weight_decay = 0 , nesterov = False)

    F.nll_loss(y_pred, target)

    Accuracy with L2 with BN
    Last 4 Epoch ranges from 99.0% to 99.2%

L1 with BN - Optimizer and Losses used
---------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=0.9, momentum=0.9)

    F.nll_loss(y_pred, target)
    PLUS
    lambda_l1 = 0.00001 # Lambda_l1 set here
     for q in model.parameters():
         l1 = l1 + q.abs().sum()
     loss = loss + l1 * lambda_l1
     
Accuracy with L1 with BN
    Last 2 epoch = 99.2%

Note: with lambda_l1 = 0.1 and 0.01 , accuracies were fluctuating across epoch, so used lambda_l1 = 0.00001

L1 with L2 with BN - Optimizer and Losses used
-----------------------------------------------
    (Weight Decay parameter used for L2) optimizer = optim.SGD(model.parameters(), lr=0.9, momentum=0.9, dampening = 0, weight_decay = 0 , nesterov = False)    
    
    F.nll_loss(y_pred, target)
    PLUS
    lambda_l1 = 0.00001 # Lambda_l1 set here
    for q in model.parameters():
      l1 = l1 + q.abs().sum()
    loss = loss + l1 * lambda_l1
    
Accuracy with L1 and L2 with BN
    Last 4 epochs, test accuracy is around 99.2%
    
Note: with lambda_l1 = 0.1 and 0.01 , accuracies were fluctuating across epoch, so used lambda_l1 = 0.00001
