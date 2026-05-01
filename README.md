# Overview
## Premise
We see intelligence as the raw ability to solve problems. In contrast, we see wisdom as the ability to create and leverage high-order information in order to calibrate problem solving.

This repository constitutes our attempt to imbue AI systems with more of what we consider wisdom.

At the present stage (2026/05), this repository is geared towards previously unextracted human behaviour characteristics out of human behaviour datasets, through an exotic interplay of computational methods.
## Strategy
### Introduction & motivation
Our belief is that, in practice, several conditions need to all be met in order for this extraction to start showing visible signs of success: imitation of human behaviour in its purest possible form, proper calibration of random search conditioned on gradient-based representations, etc (extensive details in later sections).

In order to meet all of these conditions, we believe that a multitude of methods need to be incorporated into a working solution. Several of these methods have, in isolation and in older forms, been scientifically peer-reviewed. However, many of them have not.

In order to account for time constraints, we will thus be taking the (somewhat) measured “leap of faith” of betting on years of conceptual design and experimentation directed towards this very purpose.
### Overview
We provide, later in this document, 1) a specification sheet that contains an appropriately detailed description of all necessary components for the implementation of a) the envisioned approach & b) the baselines to compare against; 2) a proposed rough path for experimentation.

These components are meant to provide a solid framework for a sufficiently capable AI agent to both 1) generate the full desired apparatus and 2) run extensive incremental experimentation in order to build any of the missing understanding that very well could be required in order to successfully combine all of these methods together.
## Central hypotheses
### Modern AI systems & their relation with high-order information
While modern AI systems exhibit increasingly more advanced and efficient problem-solving in a wide range of domains, they appear, to us, limited in their ability to create and understand high-order information such as matters of the human condition, societal dynamics, etc.

While the argument can be made that these systems are vastly different from us, it is also undeniable that they have also been exposed to an amount of information pertaining to these high-order concepts at a scale that no human has ever been, by far.

It thus appears to us that such wisdom is not going to naturally emerge from the current paradigm.
### The need for higher fidelity human behaviour imitation
In the pursuit of designing systems that exhibit wisdom, it appears that one has to make a judgement call with respect to a gradient that spans from pure imitation of existing behaviour all the way to independent discovery.

In our perspective, wisdom as it appears in humanity is an extremely advanced product of evolution. As a result, we consider it naive to practically hope for success from anything that distances itself from pure imitation of existing behaviour.

We thus believe that our best bet at embedding wisdom into computational systems is through the betterment of human behaviour imitation modeling.

In our opinion, many organic life forms and their societies do show signs of wisdom, but their behaviour is practically harder to collect digitally.
### Random search as a missing component
#### Current paradigm
We observe that, over the past ~15 years, AI research has largely focused on exploring a wide range of priors within a gradient-based optimization dominated framework.

Gradient-based optimization methods are tightly coupled to the data distribution, funneling information from the data directly into the model’s representation space.
#### Proposed augmentation
Our hypothesis is that, given our non-omniscience, meaningful value remains unextracted when search and optimization are constrained to prior and data space.

We thus argue for the need to augment the optimization process through the exploration of random space.

The most popular random space exploration method is perhaps random search. It is an optimization method that is employed when we 1) do not have the necessary priors to successfully solve a task and 2) do not have a dataset to draw information from.

When we do have a dataset to draw information from however, it becomes natural to turn towards evolutionary search. In evolutionary search, the representation space is also perturbed through random search, but data now plays a regularizing role in constraining this exploration.
#### Drawbacks & remediations
From our humble perspective, random and evolutionary search permeate several conditions that, in practice, make their successful use quite exotic relative to modern gradient-based methods.

We introduce what we consider to be the two major conditions

##### The indirect influence of data

Most popular perhaps is the practical inconvenience that emerges from the now indirect influence of data onto representation-space. This results in a slower, less efficient and noisier incorporation of valuable data information than is the case with gradient-based optimization.

We propose to mitigate this drawback by having gradient-based methods extract large amounts of information from data a always extracting as much data information as possible using gradient-based methods before operating evolutionary search in the realm of this extracted information.

##### 

A second key element is the complexity of navigating random space. With respect to our objectives, much of random space is noise.
# Specification sheet
## Data
We make use of the following data setups, that are incrementally better fits to the goal of imitating human behaviour:

Environments:

Control tasks with <10 input values (e.g. position, velocity), <10 output values (e.g. actuator pressure)
Retro video games with ~2000 input values (RAM indices), ~10 output values (e.g. up, down, jump)
Retro video games with millions of input values (image stream), ~10 output values (e.g. up, down, jump)

For each of these environments we have human subject behaviour data. In all settings, this data is not stationary since the subjects’ behaviour evolves with experience. For the retro video games, we also have fMRI neuroimaging data of the subjects that corresponds to the collected behaviour data.


In each environment, we aim to create models that are able to accurately model the complete behaviour of our human subjects, meaning not only their behaviour at a particular skill level but also the way their behaviour evolves with experience. However, for the sake of troubleshooting, we ought to experiment our way up increasingly fitting behaviour, namely:


Fixed naive behaviour policy
Repeats a single action
Alternates between actions following a given pattern
Pre-trained capable RL agent policy
Subject “stable” behaviour (meaning we would need to find a particular time slice where behaviour statistics are “stable”)
Full subject behaviour
## Methods
### Standard deep learning
dw
### Neuroevolution
#### Overview
The proposed evolutionary algorithm is an island-ba genetic algorithm that maintains a population of a fixed number (N) of agents.

Each agent has three distinct roles: generator, discriminator and mutator.


#### Generational inheritance


Generational inheritance is key in making evolutionary search more efficient.

In that framework, agents that make it through the selection stage transfer not only their genotype, as is common practice, but also their final environment state, final memory state, and cumulative lineage fitness to their offspring.

Then, during its own evaluation stage, the offspring resumes from its parent’s stored nonterminal environment state, final memory state, and starts with inherited fitness equal to the parent’s cumulative lineage fitness.

The child is evaluated for E actions, accumulating local reward. If termination occurs before E is exhausted under fixed E=k, reset the environment and use the remaining actions; otherwise descendants of terminal agents start from a fresh environment. After evaluation, store the child’s final env state and memory state, and set fitness = inherited fitness + local reward. Selection is based on this cumulative fitness.

#### Adversarial imitation
We now explain the generator and discriminator roles.

Every iteration, 

##### Application to generational inheritance

Generational inheritance is straight-forward to implement for the generator role.

However it is not so straightforward for the discriminator role, given that an agent’s chile will most likely not encounter the same generator 
#### Neural networks evolution
Each agent evolves its own neural network.
The neural network is made up of nodes and connections.

##### Architecture overview

The networks consist of three types of nodes:

Input nodes: Non-parametric nodes that receive external input signals.
   One input node per observation dimension.

2. **Hidden nodes**: Parametric nodes with learnable connections. Each hidden
   node has at most MAX_INCOMING_CONNECTIONS (3) incoming connections with
   associated weights. Outputs are computed as: standardize(weights · inputs).

3. **Output nodes**: Same as hidden nodes, but their outputs are the network's
   final predictions. One output node per action dimension.

Input nodes simply output the input values.
Output nodes output values back out of the network.
Hidden nodes.



The algorithm runs an infinite amount of iterations. The population at a given iteration is called a generation. Within an iteration, three stages are run sequentially: perturbation, evaluation and selection.



### Perturbation


 called generations (G). Within a single 
 composed of three stages:
- perturbation
- evaluation
- selection

During the perturbation stage, 

 with 50% truncation selection, meaning that the top 50% of agents with the highest fitness scores are selected and duplicated over
# Agent roles

While we only evolve a single population, agents in the population each get to play 3 roles: generator, discriminator and mutator.

Each agent has a single network model. Their network has an output space including the output space (meaning that when generating, an agent has access to it discrimination output and vice versa)

X Mutator role

In addition to their generator and discriminator roles, agents also ought to mutate themselves.
This operation is independent of the random mutations that already occur.


Agents are now also able to mutate themselves. In order to do so, they input from network


X Neural networks of dynamic complexity



X All representational capacity in the architecture

Network connections do not have weights. Instead, 
they simply sum the outputs of each neuron that they input, feed it through a per-neuron running standardization and output that transformed signal

Each neuron computes a sum of all 

### Generational inheritance




In the context of evolutionary adversarial generation, generator memory is straightforward to 

X Evolved network on top of gradient-based network

Modern AI policy network
Then have a network evolve akin to a parasite.
Starts by mapping from the output space back into the output space
Can start to go fetch from earlier layers after a certain point


X Objective: show that evolutionary algorithms can extract untapped value on top of gradient-based methods

	X Attempt showing in the context of static policy behaviour imitation, however it is likely
	that there is too little value left to extract.

	We have a dataset of human 

## Experimentation path
