This is a single-page, carefully sectioned markdown document that details our research at various levels of abstraction.

It is optimized for both human and AI agent readability.

The tipping point motivation for creating this document was to fully decouple research methodology and experimentation away from a consistent codebase, which has only recently become legitimate.

This document has two main sections: `Overview` and `Specifications`.

We describe, in the `Overview` section, our high-level understanding and research reasoning so as to contextualize our `Specifications` section.

# Overview

We begin this section by sharing, in `Why artificial sapience?` our current understanding of both sapience and its relation with artificial intelligence.

We then proceed, in `Central hypotheses`, to introducing and motivating several hypotheses that shape the nature of our research direction.

Finally, in `Research strategy`, we share insights into how we operationalize our research endeavour.

## Why artificial sapience?

### Sapience

Intelligence, wisdom and understanding are three advanced concepts that concern themselves with problem solving.

Just like most high-order concepts, all these concepts have neither universally agreed definitions nor clear boundaries with each other.

To the best of our knowledge, sapience is the term that best englobes the three terms as they manifest in human beings.

### Intelligence

Our perspective is that intelligence is proportional to the likelihood of solving a given problem.

There is per-problem intelligence: an entity can be very good at solving very specific problems but very bad at solving everything else. In that case, it is narrowly very intelligent.

More broadly valuable however is general intelligence. Human beings for instance are able to solve a wide range of problems.

### Sacrificing sapience for the sake of greater intelligence

In the pursuit of increasing the likelihood of solving a given set of problems, greater intelligence is willing to forgo certain endeavours, such as 1) considering information pertaining to the greater context that the set of problems is in (wisdom) & 2) building comprehension that is not strictly required to “solve” the tasks (understanding).


### The artificial intelligence imbalance

We believe that there is an imbalance in the world of artificial systems wherein intelligence is prioritized at the expense of the other concepts that make up sapience.

This repository represents our attempt at a contribution to restore this balance.

## Central hypotheses

### Modern AI systems, wisdom & understanding

While modern AI systems exhibit increasingly more advanced and efficient problem-solving in a wide range of domains, they appear, to us, limited in their ability to create and understand certain categories of high-order information such as matters of the human condition, societal dynamics, etc.

While the argument can be made that these systems are vastly different from us, it is also undeniable that they have also been exposed to an amount of information pertaining to these high-order concepts at a scale that no human has ever been, by far.

It thus appears to us that the wisdom and understanding required to make sense of these categories of information are not going to emerge meaningfully from the present paradigm.

### The need for higher fidelity human behaviour imitation

In the pursuit of designing systems that exhibit sapience, it appears to us that one has to position itself with respect to a gradient that spans from pure imitation of existing behaviour all the way to independent discovery.

From our perspective, sapience as it appears in humanity is an advanced product of evolution that sits at the top of an incredibly sophisticated dependency graph. As a result, we believe the chance of success in building sapient systems to drop very sharply for anything that distances itself from pure imitation of existing behaviour.

We thus believe that our best bet at embedding sapience into computational systems is through the betterment of human behaviour imitation modeling.

### Random search to better human behaviour imitation

#### Current AI paradigm

We observe that, over the past ~15 years, AI research has largely focused on exploring a wide range of priors within a gradient-based optimization dominated framework.

Gradient-based optimization methods are tightly coupled to the data distribution, funneling information from the data directly into the model’s representation space.

#### Random search as an extension

From our perspective there are three categories of spaces that problem-solving computational methods can live in: prior, data and random space.

—


The classical random space exploration method is random search. It is an optimization method that is employed when we 1) do not have the complete priors to successfully solve a task and 2) do not have a dataset to draw information from.

The classical random space exploration method when we do have a dataset that we wish to draw information from is evolutionary search. In evolutionary search, the representation space is also perturbed through random search, but data now plays a regularizing role, constraining random space exploration.

Inverse reinforcement learning is a class of methods that can be made to explore all three prior, data and random space. In practice however, the exploration of random space is minimal and focused on early-stage optimization.

—

Our hypothesis is that, given our non-omniscience, meaningful value remains unextracted when search and optimization are constrained to prior and data space.

We thus propose to explore the interplay of prior, data and random space.

#### Main constraints & remediations

From our perspective, random and evolutionary search permeate several conditions that, in practice, make their successful calibration quite exotic relative to modern gradient-based methods.

We introduce what we consider to be two of these conditions, and how we think best to approach them.

##### Data indirectness

Given that, relative to gradient-based methods, data is retrograded from being a direct to an indirect optimization signal, valuable data information now gets assimilated into representation space at a much slower and less efficient pace.

We propose to mitigate this phenomenon by largely building on top of the modern paradigm. In practice, this means extracting information using gradient-based methods and operating evolutionary search in the realm of this extracted information.

##### Navigating random space

Perturbing representation space using random space is a delicate endeavour. With respect to our objectives, much of random space is destructive noise.

We thus need to calibrate our search to satisfy a delicate balance between representation construction and noise application.

In the pursuit of that interest, we propose to explore methods that can flexibly calibrate themselves with respect to that balance.

## Research strategy

### Motivation

Our belief is that, in practice, several conditions need to all be met in order to reach visible signs of deeper understanding and wisdom: imitation of human behaviour in its complete form, proper calibration of random search conditioned on gradient-based representations, etc.

In order to meet all of these conditions, we believe that a multitude of methods need to be incorporated into a working solution. Several of the methods that we later detail have, in isolation and in older forms, been scientifically peer-reviewed. However, many of them have not.

In order to account for time constraints, we are taking the (somewhat) measured “leap of faith” of building towards our objective while betting instead on our observed empirical results in addition to years of conceptual experimentation.

### Outline

We provide, in the upcoming `Specifications` section of this document 1) an appropriately detailed description of all the components envisioned for the implementation of both a) the envisioned approach & b) baselines to compare against; and 2) a proposed rough path for experimentation.

These components are meant to provide a solid framework for a sufficiently capable AI agent to both 1) generate the apparatus and 2) run extensive incremental experimentation in order to build any of the missing understanding that could be required in order to successfully combine all of these methods together.

### Current objective

At the present stage (2026/05), our research is geared towards attempting to first extract previously unextracted human behaviour characteristics out of human behaviour datasets.

# Specifications

## Methods

### Evolutionary algorithm

#### Core design

##### Overview

###### Genetic algorithm

There are many types of evolutionary search algorithms.

We choose to operate a genetic algorithm in order to explore random space deeper than is possible with evolutionary strategies, the current most popular evolutionary search method that runs per-iteration population-wide recombination.


It maintains a single population of a fixed number `pop_size` of agents.

Each agent has three distinct roles: generator, discriminator and mutator (more details below).


The algorithm loops every iteration through three classical stages: mutation, evaluation and selection.
The generation of agents at iteration `<iter>` is `gen <iter>`.

###### Islands

We also choose to make the genetic algorithm island-based. We explain our reasoning in the `Adversarial imitation` section.

There are `num_islands` islands, each composed of the same `island_size` (`pop_size / num_islands`) number of agents.

###### Mutation stage

During the mutation stage, all agents are randomly perturbed.

We do not implement crossovers. Our perspective is that while there are many benefits to crossovers in isolated practical situations, they are very hard to set up properly in the more complex situation that we operate in.

Finally, as per their mutator role, all agents mutate themselves (we detail this operation in the `Mutator role` section).


###### Evaluation stage

During the evaluation stage, all agents behave in a virtual environment and are quantitatively graded with a fitness score.

###### Selection stage

Selection for both the islands and the population is achieved using 50% truncation selection.

We choose 

This corresponds to ranking agents by fitness score, discarding the bottom 50% of agents and duplicating the top 50% of agents (to put it in illustrated language terms: selected agents become parents, produce two offspring, and erase themselves).

Per-island selection occurs every iteration. Every `island_merge_freq` iterations, the global population is ranked and 50% truncation selection occurs. New islands are then formed using the selected population at random.

#### Generational inheritance

Generational inheritance is meant to make evolutionary search more efficient with respect to evaluation.
It does so by having agents run on portions of a task instead of full tasks.


Agents that are selected transfer to their two offspring not only their genotype, but also their final environment state, final memory state, and cumulative lineage fitness scores.

During its own evaluation stage, the offspring thus loads and resumes from its parent’s memory state and environment state. Its fitness score becomes the sum of its lineage fitness score plus its own. The selection stage thus accounts for that sum rather than the individual’s fitness.

Agents are evaluated for `num_states` states. If the virtual environment terminates before `num_states` take place, we reset the environment and memory (but not the fitness score) and run the remaining states.

#### Adversarial imitation

We now explain the generator and discriminator roles introduced earlier.

##### 
#### 

Every iteration, 

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

Each agent maintains one neural network.

Each network is a directed graph of nodes.



Each network is represented as a directed graph of nodes. The population is flattened into batched tensors for evaluation, while mutation remains graph-based and sequential.

##### Graph representation

The graph has three node roles:

- Input nodes receive external observations. There is one input node per observation dimension.
- Hidden nodes are mutable computational nodes. They can be added or removed by evolution. Each hidden node also has a fixed polarity, sampled uniformly at birth from `{-1, +1}`.
- Output nodes return values from the network. There is one output node per action/output dimension.

Input and output nodes are created during initialization and are permanent. Hidden nodes are grown and pruned over evolutionary time. Every node has a mutable identifier, used for tensor indexing, and an immutable identifier, used to clone and reconstruct the graph after pruning has changed mutable positions.

Connections are incoming-node and outgoing-node references. Hidden and output nodes may have at most `MAX_INCOMING_CONNECTIONS = 3` incoming connections. Connections have no weights, signs, or other parameters: they only specify which node outputs are summed by the receiving node.

The graph maintains synchronized views of nodes by role and connection status:

- `all`: every node in mutable-index order.
- `input`, `hidden`, `output`: nodes grouped by role.
- `receiving`: nodes with incoming connections, repeated once per incoming connection.
- `emitting`: nodes with outgoing connections, repeated once per outgoing connection.
- `being_pruned`: a temporary recursion guard during cascading pruning.

##### Computation and normalization

Input nodes receive observation values. Hidden and output nodes compute an unweighted sum of up to three incoming standardized node values:

```text
x = sum(input_z_i)
z = standardize_per_node(x)
```

The network does not use connection weights, node biases, or activation functions. Instead, every node has persistent running standardization statistics:

- `n`: number of observed raw values.
- `mean`: running mean.
- `m2`: running sum of squared deviations.
- `x`: previous raw value.
- `z`: previous standardized output.

Standardization uses Welford's online algorithm. New non-zero values update the running statistics and return a z-score. Zeros and previous z-scores pass through without updating. A node returns zero until it has at least two samples, giving new structures a short warmup period before they emit meaningful standardized signals.

Hidden-node polarity is applied after standardization:

```text
hidden_output = polarity * z
```

Output nodes do not have polarity. They return their standardized value directly, keeping each output dimension stable while inhibitory or contrastive structure emerges inside the hidden graph.

These normalization statistics are learned network state, not episode-local hidden state. Resetting a network for a new episode therefore does not clear them.

##### Growth

Initialization creates all input and output nodes. After initialization, growth creates hidden nodes.

When a hidden node is grown, three connections are created:

- `in_node_1 -> new_node`
- `in_node_2 -> new_node`
- `new_node -> out_node_1`

Sampling first connects unused parts of the graph: the first incoming connection prioritizes input nodes that do not yet emit, and the outgoing connection prioritizes output nodes that do not yet receive. This helps early networks connect all inputs and outputs before growing denser internal structure.

When there are multiple candidates, sampling is graph-local: the sampler expands outward from a reference node and accepts candidates at each distance with probability `local_connectivity_probability`. Higher values encourage local modular wiring; lower values allow more global wiring.

If several grow mutations happen in one mutation call, they are chained: each new hidden node becomes the starting input node for the next growth operation.

##### Pruning

Pruning removes hidden nodes. Input and output nodes are never pruned.

When a hidden node is pruned, its incoming and outgoing connections, standardization state, and fixed polarity are removed. Later mutable identifiers are decremented so tensor indices stay contiguous. Pruning can cascade: if another hidden node loses all incoming or outgoing connections, it is pruned as well. A temporary `being_pruned` list prevents recursive pruning loops.

##### Mutation

Mutation first perturbs evolvable scalar parameters:

- `avg_num_grow_mutations` is multiplied by `1 + 0.01 * randn`.
- `avg_num_prune_mutations` is multiplied by `1 + 0.01 * randn`.
- `local_connectivity_probability` receives additive `0.01 * randn` noise and is clamped to `[0, 1]`.
- `num_network_passes_per_input` decreases by one with probability 1/100 when above one, and increases by one with probability 1/100.

Then the architecture is perturbed. Average grow/prune mutation values are converted into integer counts stochastically: the integer part is always used, and the fractional part is the probability of adding one more mutation. Pruning is applied first, growth second.

After mutation, the network regenerates computation tensors:

- `in_nodes_indices`: for each output/hidden node, the mutable identifiers of its incoming nodes, padded with `-1`.
- `hidden_polarities`: the fixed `-1` or `+1` polarity for each hidden node.

Mutable nodes are ordered as output nodes first, then hidden nodes. The incoming-node index tensor and polarity tensor both use this order.

##### Batched population computation

Although networks have heterogeneous topologies, the population is evaluated through one flattened tensor layout. The population-level forward call accepts `[num_nets, num_inputs]` and returns `[num_nets, num_outputs]`.

Before computation, the population is converted into batched lookup tensors:

- input-node start indices for each network in the flattened node tensor;
- flattened positions of all input nodes;
- flattened positions of all output nodes;
- flattened positions of all non-input nodes;
- a flattened incoming-node lookup table for every mutable node;
- all hidden-node polarities;
- a pass mask describing which networks should still update at each recurrent-style pass.

Index `0` in the flattened tensors is a dummy node that always outputs zero. Empty connection slots are clamped to this index so padded inputs can participate safely in vectorized gathers.

The forward pass proceeds as follows:

```text
1. Start from the previous standardized node outputs.
2. Insert the current observations into the flattened input-node positions.
3. Standardize the input nodes.
4. For each network pass:
   a. gather the three inputs for every mutable node;
   b. compute unweighted sums;
   c. standardize the updated node values;
   d. apply fixed polarity to hidden nodes;
   e. update only the mutable nodes whose network is still active for this pass.
5. Gather and reshape output nodes into [num_nets, num_outputs].
```

The number of passes is the maximum `num_network_passes_per_input` across the population. Networks with fewer passes are masked out after their own pass count, giving networks evolvable recurrent-like depth while preserving one batched computation path.

##### Resampling and cloning

Selection resamples the population by replacing each network with a clone of a selected parent. Cloning uses explicit graph serialization instead of generic deep copying because incoming/outgoing node references are circular and large evolved graphs can exceed Python's recursion limit.

Serialization stores each node's role, mutable identifier, immutable identifier, hidden-node polarity where applicable, incoming immutable identifiers, running standardization state, self-adapting mutation parameters, total nodes grown, and device placement. Reconstruction first creates all nodes, then reconnects incoming edges by immutable identifier.

Forward passes update the population's batched running-standardization tensor, so individual graph objects are stale during evaluation. Before mutation after a forward phase, the batched statistics are synchronized back into each network so offspring inherit the current normalization state.

##### Performance notes

Mutation is performed sequentially across networks, then tensor data is cached for fast batched preparation. Multiprocessing was rejected because individual mutations were too cheap relative to inter-process communication and serialization overhead. The current approach keeps graph mutation simple and uses tensor operations for forward computation.



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

## Experimentation path



