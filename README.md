This is a single-page, carefully sectioned markdown document that details our research at various levels of abstraction.

It is optimized for both human and AI agent readability.

The tipping point motivation for creating this document was to fully decouple research methodology and experimentation away from a consistent codebase, which has only recently become legitimate.

This document has two main sections: `# Overview` and `# Specifications`.

We describe, in `# Overview`, our high-level understanding and research reasoning so as to contextualize `# Specifications`.

# Overview

We begin this section by sharing, in `## Why artificial sapience?` our current understanding of both sapience and its relation with artificial intelligence.

We then proceed, in `## Central hypotheses`, to introduce and motivate several hypotheses that shape the nature of our research direction.

Finally, in `## Research strategy`, we share insights into how we operationalize our research endeavour.

## Why artificial sapience?

We motivate, in this sub-section, the value of artificial sapience in contrast to artificial intelligence.


### Sapience

Intelligence, wisdom and understanding are three advanced concepts that concern themselves with problem solving.

Just like most high-order concepts, all these concepts have neither universally agreed definitions nor clear boundaries with each other.

To the best of our knowledge however, sapience is the term that best englobes the three terms in how they manifest in human beings.

### Intelligence

Our perspective is that intelligence is proportional to the likelihood of solving a given problem.

There is per-problem intelligence: an entity can be very good at solving very specific problems but very bad at solving everything else. In that case, it is narrowly very intelligent.

More broadly valuable however is general intelligence. Human beings and modern artificial intelligence systems for instance are able to solve a wide range of problems.

### Sacrificing sapience for the sake of greater intelligence

In the pursuit of increasing the likelihood of solving a given set of problems, greater intelligence is willing to forgo certain endeavours, such as 1) considering information pertaining to the greater context that the set of problems is in (wisdom) & 2) building comprehension that is not strictly required to “solve” the tasks (understanding).


### The artificial intelligence imbalance

We believe that there is an imbalance in the world of artificial systems wherein intelligence is prioritized at the expense of the other concepts that make up sapience.

This repository represents our attempt at a contribution in the pursuit of restoring this balance.

## Central hypotheses

We share, in this sub-section, the three core hypotheses that motivate our research endeavours.


### Modern AI systems, wisdom & understanding

While modern AI systems exhibit increasingly more advanced and efficient problem-solving in a wide range of domains, they appear, to us, limited in their ability to create and understand certain categories of high-order information such as matters of the human condition, societal dynamics, etc.

While the argument can be made that these systems are vastly different from us, it is also undeniable that they have also been exposed to an amount of information pertaining to these high-order concepts at a scale that no human has ever been, by far.

It thus appears to us that the wisdom and understanding required to make sense of these high-order information categories are not going to emerge meaningfully from the present paradigm.

### The need for higher fidelity human behaviour imitation

In the pursuit of designing systems that exhibit sapience, it appears to us that one has to position itself with respect to a gradient that spans from pure imitation of existing behaviour all the way to independent discovery.

From our perspective, sapience as it appears in humanity is an advanced product of evolution that sits at the top of an incredibly sophisticated dependency graph. As a result, we believe the practical chances of success in building sapient systems to drop very sharply as we ever so slightly distance ourselves from pure imitation of existing behaviour.

We thus believe that our best bet at embedding sapience into computational systems is through the betterment of human behaviour imitation modeling.

### Random search to enhance human behaviour imitation

#### Current AI paradigm

We observe that, over the past ~15 years, AI research has largely focused on exploring a wide range of priors within a gradient-based optimization dominated framework.

Gradient-based optimization methods are tightly coupled to the data distribution, funneling information from the data directly into the model’s representation space.

#### Random search as an extension

From our perspective there are three categories of spaces that problem-solving computational methods can live in: prior, data and random space.

—


The classical random space exploration method is random search. It is an optimization method that is employed when we 1) do not have the complete priors to successfully solve a task and 2) do not have a dataset to draw information from.

The classical random space exploration method, when we do have a dataset that we wish to draw information from, is evolutionary search. In evolutionary search, the representation space is also perturbed through random search, but data now plays a regularizing role, constraining random space exploration.

Inverse reinforcement learning is a class of methods that get to live in all three prior, data and random spaces. In practice however, the exploration of random space is minimal and focused on early-stage optimization.

—

Our hypothesis is that, given our non-omniscience, meaningful value remains unextracted when search and optimization are constrained to prior and data space.

#### Main constraints & remediations

From our perspective, evolutionary search permeates several conditions that, in practice, make their successful calibration quite exotic relative to modern gradient-based methods.

We introduce what we consider to be two of these conditions, and how we think best to approach them.

##### Data indirectness

Given that, relative to gradient-based methods, data is retrograded from being a direct to an indirect optimization signal, valuable data information now gets assimilated into representation space at a much slower and less efficient pace.

We propose to mitigate this phenomenon by largely building on top of the modern AI paradigm. In practice, this means first extracting information and training policies with gradient-based methods, and only then, within that generated realm, operate evolutionary search.

##### Navigating random space

Perturbing representation space using random space is a delicate endeavour. With respect to our objectives, much of random space is destructive noise.

We thus need to calibrate our search to satisfy a delicate balance between representation construction and noise application.

In the pursuit of that interest, we implement mechanisms throughout our methods in order to 1) calibrate noise application and 2) shield existing representations.

## Research strategy

We share, in this sub-section, our strategy for operationalizing our research endeavour.


### Motivation

Our belief is that, in practice, several conditions need to all be met in order to reach visible signs of sapience: imitation of human behaviour in its most native form, proper calibration of random search conditioned on gradient-based representations, etc.

In order to meet all of these conditions, we believe that a multitude of novel methods need to be incorporated into a working solution.

In this attempt of ours, several of the methods that we later detail have, in isolation and in older forms, been scientifically peer-reviewed. However, many of them have not.

In order to account for time constraints, we are taking the (somewhat) measured “leap of faith” of building towards our objective while betting instead on 1) our observed empirical results and 2) years of conceptual experimentation.

### Outline

We provide, in the upcoming `Specifications` section of this document 1) an appropriately detailed description of all the components envisioned for the implementation of both a) the envisioned approach & b) baselines to compare against; and 2) a proposed rough path for experimentation.

These components are meant to provide a solid framework for a sufficiently capable AI agent to both 1) generate the research apparatus and 2) run extensive incremental experimentation in order to build any of the missing understanding that could be required in order to successfully combine all of these methods together.

### Current objective

At the present stage (2026/05), our research is geared towards attempting to first extract previously unextracted human behaviour characteristics out of human behaviour datasets. We begin with the ability to model human behaviour as it progresses through time.

# Specifications

This section is composed of three sub-sections: `## Modeling methods`, `## Data setup` & `## Experimentation path`.

We specify in each of these sections our complete methodologies and the reasoning behind each of our design choices.

## Modeling methods

We detail, in this sub-section, all of our `### Evolutionary search` methods and how we make them concord with modern methods in `### Leveraging the gradient-based paradigm`.


### Evolutionary search

We begin by introducing our methods that pertain to evolutionary search.

In `#### Core design`, we detail design decisions that pertain to the core evolutionary search algorithm itself.

In the later sub-sections we detail more specific mechanisms with the following purposes:
- `#### Adversarial imitation`: open-ended human behaviour imitation
- `#### Generational inheritance`: increased evaluation-efficiency
- `#### Neural networks`: flexible representation construction
- `#### Mutator role`: evolving self-adaptability


#### Core design

There are many types of evolutionary search algorithms that offer a wide-range of trade-offs that suit different contexts.

In our context of evolving neural networks (details in `#### Neural networks`), there are roughly two practically proven classes of methods: genetic algorithms and evolution strategies.

We opt for genetic algorithms.

##### Why not evolution strategies?

Evolution strategies methods build on the idea of population-scale recombination of agents, placing higher recombination weight to better performing agents.

We are of the opinion that recombination of neural networks, in practice, imposes two conditions that we believe too restrictive for our purposes.

###### Network design requirements


Firstly, recombination requires a network format suitable for recombination. While methods like NEAT have shown that dynamic-connectivity networks can be recombined, these methods generally imply a wide range of assumptions and accept practical inefficiencies. In practice, recombination thus requires the use of fixed-connectivity networks, so as to operate in real value parameter (weight and bias) space. 

While general intelligence has emerged from human-made fixed connectivity networks, it has thus far only emerged in a pure gradient-based optimization context.

It is thus still unclear, given the many disconnects between gradient-based optimization and evolutionary search, how much value can be leveraged from fixed-size networks.

We further explain our reasoning for the use of dynamic-connectivity networks in `#### Neural networks`.

###### Representation incompatibility


Secondly, we believe recombination to be too opinionated of a design decision.

Recombination assumes that the gains from combining partial solutions outweigh the disruption caused by merging incompatible representations. While this tradeoff often works empirically, the degree to which useful information is destroyed remains an active research concern.

##### Genetic algorithm

Following our line of reasoning about recombinations, we opt to make our genetic algorithm crossover-free. However, we detail in `#### Mutator role` how we create a channel for agents to observe and integrate other agents’ valuable representations, thus not completely forgoing recombination.

—

Our population maintains a single population of a fixed number `pop_size` of agents.

Each agent has three distinct roles: generator, discriminator and mutator (more details in `#### Adversarial imitation` and `#### Mutator role`).


The algorithm loops every iteration through three standard stages: mutation, evaluation and selection.

The generation of agents at iteration `<iter>` is `gen <iter>`.

##### Islands

We also choose to make the genetic algorithm island-based. We explain our reasoning in `#### Adversarial imitation`.

There are `num_islands` islands, each composed of the same `island_size` (`pop_size / num_islands`) number of agents.

##### Mutation stage

During the mutation stage, all but a few agents (see `###### Elitism`) are randomly perturbed independently.

Finally, as per their mutator role, all agents mutate themselves (we detail this operation in the `Mutator role` section).

##### Evaluation stage

During the evaluation stage, all agents behave in a virtual environment and are assigned a quantitative personal fitness score.

The mechanics that decide the value of that personal fitness score are detailed in `#### Adversarial imitation`.

* In order to better understand the notion of “personal fitness score”, we quickly foreshadow here the concept of “lineage fitness score”, with an agent fitness score being the sum of both its personal and lineage fitness score (details in `#### Generational inheritance`).

##### Selection stage

We now introduce our selection stage heuristics.


###### Truncation

Selection for both the islands and the population is achieved using a classical truncation selection with a rate of 50%.


This corresponds to ranking agents by fitness score, discarding the bottom 50% of agents and duplicating the top 50% of agents (to put it in illustrated language terms: selected agents become parents, produce two offspring identical to themselves, and then erase themselves).

###### Why 50%?

50% is the largest truncation selection percentage wherein we do not need heuristics to decide how many offspring a parent produces (for instance, 60% needs a heuristic to decide which parents produce how many of the remaining 40% of agents).

Smaller heuristic-free truncation selection percentages like 25%, 12.5% choose to increasingly diminish random space exploration breadth in order to increasingly invest resources in the best current solutions.

We choose 50% because 1) we believe the optimization dynamics of our `#### Adversarial imitation` context to be too intricate to confidently make the type of trade-off described above & 2) because we do not see value in further optimizing the number of offspring per parent: our reasoning is that if top parents came across relatively more valuable representations, their children are more likely to themselves become parents relative to other children with lower performing parents.

###### Interaction between islands and the global population

Per-island 50% truncation selection occurs every iteration except every `island_merge_freq` iterations.

Every `island_merge_freq` iterations, the global population is ranked and a global 50% truncation selection occurs instead.

Islands are then reformed by dispatching the global population’s new offsprings randomly across the `num_islands` islands.


###### Elitism

Following the evaluation stage, every island has an elite: the agent with the highest fitness score on the island.

The elite is marked to have one of its two offspring not be mutated during its next iteration mutation stage.

This mark remains applied for either per-island selection and population-wide selection (though population-wide selection could see a given island’s elite agent discarded due to not being ranked in the top 50% fitness scores of the global population).

###### Why elitism?

First of all we choose to implement some form of elitism, over no form of elitism, in order to provide some shielding against XXX.

However we decide not to implement a heavier form of elitism so as to not considerably slow-down the search: the representations that we are looking for are very complex and are likely very deep in the search space.

#### Adversarial imitation

We optimize for imitation using an adversarial imitation algorithm.

##### Why adversarial imitation?

We choose to operate adversarial imitation rather than other forms of supervised/unsupervised imitation.

Our reasoning is multi-fold:

1) Adversarial imitation is the most open-ended imitation technique in that it maintains at any given time multiple optimization paths: up to the theoretical perfect discriminator / generator, representation formation is only  are only constrained tothe path to new representations is open-ended.

This is in contrast with static optimization paths where representation creation is much more constrained to a given path.

2) The operationalization of adversarial imitation in an evolutionary search context is quite a bit different than it is with gradient-based optimization.

Adversarial imitation in gradient-based optimization appears to have two drawbacks that presently make them second-class citizens to diffusion models: 1) training instability and 2) one-shot generation.

Point 1 is only relevant in the context of gradient-based optimization and point 2 is only relevant in the context of having a fixed inference budget, which we do not (see the `Neural Networks` section).



As mentioned previously, each agent is both a generator and a discriminator.

Every iteration, each agent is evaluated for both its generation and discrimination performance.

Generation is quite straightforward: the agent simply behaves in the virtual environment.

Discrimination is a bit more involved: the agent observes the behaviour of all other agents on its island.

—


We now introduce the method.




##### 
#### 

Every iteration, 

#### Generational inheritance

Generational inheritance is meant to make evolutionary search more efficient with respect to evaluation.
It does so by having agents run on portions of a task instead of full tasks.


Agents that are selected transfer to their two offspring not only their genotype, but also their final environment state, final memory state, and cumulative lineage fitness scores.

During its own evaluation stage, the offspring thus loads and resumes from its parent’s memory state and environment state. Its fitness score becomes the sum of its lineage fitness score plus its own. The selection stage thus accounts for that sum rather than the individual’s fitness.

Agents are evaluated for `num_states` states. If the virtual environment terminates before `num_states` take place, we reset the environment and memory (but not the fitness score) and run for the remaining states.

—

This mechanism has an important impact on the selection process, as described in the previous sub-section. Children whose lineage was relatively more competent have more room to explore mutations that do not yield immediate returns.

We see value in this property for the same reasons that occur in the natural world.



##### Application to generational inheritance

Generational inheritance is straight-forward to implement for the generator role.

However it is not so straightforward for the discriminator role, given that an agent’s chile will most likely not encounter the same generator 

#### Neural network

$$$$$ Evolution

Each agent maintains one neural network.

The network is a directed graph. Structurally, the network is made up of nodes and connections.

£££££ Computation 

While network evolution is performed per-network, computation is batched so it can run in parallel on the GPU.

The ‘3 x num_nodes’ 



















































#### Neural networks

Each agent maintains one neural network that is a directed graph of nodes.

##### Nodes

The graph has three types of nodes: `input`, `hidden` and `output` nodes.

Nodes can communicate with other nodes through node-to-node directed connections.

Each node (except `input` nodes) maintains a list of `in` nodes: nodes that connect to that given node.
Each node has up to 3 `in` nodes. We explain the rationale in `##### Execution`.

Each node also maintains a list of `out` nodes`: nodes that the given node connects to.

###### No weights nor biases

Nodes do not assign weights to the values they receive, nor do they apply a bias.

All nodes (including `input` nodes) run an online Welford’s algorithm standardization of the sum of their input values using Welford's online algorithm, and output the result of that standardization.

##### Network architecture

There are `n` `input` nodes, corresponding to the `n` real values that the network inputs. Input nodes only input their assigned real value input.



There are `m` `output` nodes, corresponding to the `m` real values that the network outputs.


##### Parameters

###### Overview

Each network maintains its own set of these 4 mutable parameters:
- `avg_num_grow_mutations`, float, x >= 0
- `avg_num_prune_mutations`, float, x >= 0
- `num_network_passes_per_input`, int, x >= 1
- `local_connectivity_probability`, float, x >= 0, x <= 1

We decide to make these parameters mutable because we do not believe to have good general priors to set them properly ourselves.


###### `self.avg_num_grow_mutations` & `self.avg_num_prune_mutations`

The `avg_num_grow_mutations` and `avg_num_prune_mutations` values control how much growing and pruning randomly occurs.

We create these parameters in order to give the evolutionary search the flexibility to control the average occurrence of both mutations.

We believe this to be valuable because different stages of the evolution process likely benefit from different growth/prune regimes.

—

We randomly set the initial value, independently for both `avg_num_grow_mutations` `avg_num_prune_mutations`, to a random value between 0.1 and 1.0 (uniform sampling).

This wide range reflects the fact that we do not have good priors for setting these values.
Because we setup the perturbation of these parameters to be multiplicative (see ###### `self.perturb_parameters()`). We set the lower bound to 0.1 so as to not be trapped by being too close to 0, and the upper bound to 1.0 so as to not risk too much representation perturbation in early evolutionary search.


###### `self.num_network_passes_per_input`

The `num_network_passes_per_input` value controls how many node passes the network operates for every new series of input values. When it is greater than one, the given series of input values is simply fed repeatedly. Only the final output values are fed out of the network.

We create this parameter in order to give networks the flexibility to control how much processing time a given series of input values is given before moving on to a new one.

We believe this to be valuable for two main reasons:

1) If the network grows large, given that our networks do not use layers, an increasing number of node passes will be required in order for a given node to communicate to another given node.

2) We do not want to get in the way of the possibility that certain network configurations could derive value from having more network structure involved for any series of input values.

—

We randomly set the initial value of this parameter to a random value according to a shifted geometric distribution where p(1) = 50%, p(2) = 25% etc.

We choose this format because the evolutionary search begins with minimal networks and large `num_network_passes_per_input` are likely to be a waste of compute. However we do not have good priors for this design decision and thus wish to leave the option for large values open.

###### `self.local_connectivity_probability`


The `local_connectivity_probability` value controls the probability of accepting the current set of nodes considered (see `###### self.grow_connection()`). When `local_connectivity_probability` is high, nearby nodes are more likely to be made connected to each other.

We create this parameter in order to give the evolutionary search the flexibility to have some control over whether to push for local or global connectivity at different stages of the search. We choose to make this a global per-network parameter rather than a per-node parameter in order not to explode the search dimensionality.

—

We randomly set the initial value of this parameter to a value between 0 and 1 (uniform sampling).
This reflects, once again, that we do not know what a good initial value is.

##### Mutation stage

All networks are modified during the mutation stage.

The modifications made during the mutation stage are both random or agentic (see `#### Mutator role`).

There are three sub-stages in the random mutation stage: perturbing parameters, pruning node(s) and growing node(s). Parameter perturbation occurs first so that node growing and pruning reflect that change. Node pruning occurs after node growing so as to not prune untested new representational capacity. 

###### `self.perturb_parameters()`

####### `self.avg_num_grow_mutations` & `self.avg_num_prune_mutations`

rand_val = 1.0 + 0.01 * random gaussian sample (independent for both parameters)
parameter *= rand_val

—

This random mutation is multiplicative and makes going from 10 to 11 as likely as going from 1.0 to 1.1.
We choose multiplicativity over additivity in order to account for X.

We handpick the 0.01 sigma value somewhat randomly.

####### `self.num_network_passes_per_input`

rand_val = uniform sampling of 1, 0 or -1
parameter += rand_val (while making sure parameter stays in its desired range)

—

We believe that larger and deeper networks are likely to benefit from running multiple node passes. We thus purposely make this parameter able to vary quite a bit over time.

We do not want a network to overfit to its current `self.num_network_passes_per_input` value.

####### `local_connectivity_probability`

rand_val = 0.01 * random gaussian sample
parameter += rand_val (while making sure parameter stays in its desired range)

—

X

###### Grow node(s)

`self.grow_node()` is the only method that creates more representational capacity in a given network.

It does so by sampling three existing nodes and creating a `hidden` node that inputs from the first two sampled nodes and outputs to the third sampled node.

####### Sampling nearby nodes

####### Grow connection

###### Prune node(s)

####### Prune connection





#### Mutator role

We now detail how agents are given the opportunity to alter their own connectivity.




###### Indirect recombination
D

### Leveraging the gradient-based paradigm

We now detail how we plan to bring evolutionary search and the modern paradigm together.

Grafting network evolution on top of fully functional gradient-based models




## Data setup

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

## Experimentation path

We now detail a sequential order of experimentation and their purposes.

We propose to first attempt our most complex experiment using our full list of methods and most advanced dataset, and only if it fails do we start to remove some of the complexity. 

Our most complex experiment is that of 
