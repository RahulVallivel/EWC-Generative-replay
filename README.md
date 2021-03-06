# EWC-Generative-replay
Catastrophic forgetting is an important problem plaguing neural networks. It affects the performance of the neural network in continual learning task. Previous attempts have tried to mitigate catastrophic forgetting using using Elastic weight consolidation (EWC) or by using a memory buffer to replay previous tasks. This project aims to combine EWC and a Generative memory block made up of a Generative Adversarial Networks and study its effects.

# Implementation
The objective ois to make the classifier learn sequential Tasks and not forget them i.e, maintain their performance from the previous task while being trained on a new task.

We trained our classifier on three tasks A,B and C. Each task is a classification problem on a permuted MNIST dataset.

<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.32.54%20PM.png" width="348">

### EWC
For elastic weight consolidation after training the classifier with each task, the validation data was used to calculate the fisher information matrix. The diagonal elements in the Fisher information matrix contains the information about which weights are important for performance in task A. Now we start training the classifier on the data for task B. But now with the loss function we add a extra regularization term which will not allow the weights important for Task A to deviate much. This regularization term is made by multiplying the fisher information with the difference of the optimal weights for task A and the current weight while training task B.

<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.24.35%20PM.png" width="248">

### Generative Replay
This Generative memory block is made up of two Generator networks and one Discriminator network as shown in Fig2. One Generative network is called as Scholar and the other is an old scholar that is the version of the Generator for previous task. First The GAN with the scholar is trained on task A then the weights of the scholar GAN are copied to the old scholar. Now the data for task A is not available and task B data is available . Now during training of task B, a batch of task B data and a batch of task A data from the old scholar is mixed and given as training data to scholar GAN. At the end of training of scholar GAN for task B, it can generate both task A data and task B data. Now the classifier(solver) is to be trained on the data for task B. first the data from the scholar Generator are passed through the classifier to generate labels then the labels are paired with the scholar generated images to make a batch of task A images generated from the Generative memory this batch is mixed with the task B data and given as input for the classifier, so now the classifier has access to data from both the tasks hence it will not suffer from catastrophic forgetting.

<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.23.14%20PM.png" width="348">
<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.23.28%20PM.png" width="548">
<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.23.41%20PM.png" width="548">

# Results

#### Graph showing the testing accuracy for two tasks with and without EWC + Generative replay

<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.22.36%20PM.png" width="548">

#### Graph showing the testing accuracy for three tasks with and without EWC + Generative replay


<img src="https://github.com/RahulVallivel/EWC-Generative-replay/blob/master/img/Screen%20Shot%202018-12-16%20at%205.22.07%20PM.png" width="548">


