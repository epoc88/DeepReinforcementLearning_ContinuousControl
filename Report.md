

# Project 2: Continuous Control task

### DDPG

In project 2, the goal is to solve the continuous control task by DDPG, which is a model-free, off-policy actor-critic algorithm based on the deterministic policy gradient 
[DPG](http://proceedings.mlr.press/v32/silver14.pdf). It uses deep neural network as function approximator to learn policies in the high-dimensional continuous action spaces. Moreover, it is able to learn the large, non-linear actor-critic in a stable and robust way, as it combines the insights from the Deep Q Network into the algorithm: 1. train the network off-policy with samples from a replay buffer to minimize correlations between samples; 2. use double Q network with separated target and local Q network to increase the stability.

The algorithm: Firstly, it creates an agent with both critic _Q(s,a|&theta;)_ and 
actor _&pi;(s|&theta;)_ network initialized. Since this work is an episodic task, the network is trained iteratively with many episodes. Within each episode, the action _a<sub>t</sub>=&pi;(s|&theta;)+Noise<sub>t</sub>_ is generated
based on the current policy network &pi;,  the noise term is added to allow for exploration. 
By executing the action, the reward and the new state are updated, and these experiences are added to the replay buffer.
For learning the model weight, the local critic network parameters are 
updated by minimizing the error between estimated Q target and the actual Q value at current timestep. Then the local actor network parameters can be learned by maximizing the expected Q(s,a') values where a' is from the actor network prediction &pi;(s). 
After the local critic and local actor network weights are updated, a soft update on the target critic and actor network is done.
<img src="https://github.com/epoc88/DeepReinforcementLearning_ContinuousControl/blob/master/images/DDPG.png" width="60%" align="top-left" alt="" title="DDPG algorithm" />

For learning, data collection process are performed in a parallel, i.e., the 20 identical agent environment is used to collect the experience data (state, action, reward, next_state, done) which are then added to replay buffer. 
 
### Implementation details
#### actor network: 
The state observation is input and the action is output. 

Fully connected layer 1 - 400 nodes  
Fully connected layer 2 - 300 nodes  


#### critic network: 
Uses similar architecture as actor network, which takes both state observation and action as input and the scalar (_Q_ value) as the output.

Fully connected layer 1 - 400 nodes  
Fully connected layer 2 - 300 nodes  

#### Hyperparameters  

Hyperparameters | value
---|---
Batch size | 128
Gamma | 0.99
Tau | 1e-3
Actor learning rate | 1e-3 
Actor learning rate minimum | 1e-4 
Critic learning rate | 1e-3 
Critic learning rate minimum | 1e-4 
Steps / each Learn   | 20
Minibatches per learning step| 8 
OU sigma |0.2
OU theta | 0.15
Epsilon decay for noise process | 1e-6


### Score plot
The environment is solved at **98** episodes. 
<img src="https://github.com/epoc88/DeepReinforcementLearning_ContinuousControl/tree/master/images/DDPG.png" width="60%" align="top-left" alt="" title="DDPG algorithm" />

![score plot](https://github.com/score_plot.png)  

### Future work
##### Improving performance
Training time takes too long in each episode. Even tough GPU has been turned on for training at ddpgn_agent.py -> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"). However, overall training time still took too long, about three and half hours for 100 episodes.

##### Hyper parameter tuning
Current hyperparameters are the initial guess, and have not been optimized. A selection of hyperparameter mechanism or visualization could be done in the future.
 
