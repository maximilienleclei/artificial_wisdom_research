# Introduction
## Premise
This research codebase is an attempt to surface novel results arising from the interplay of hypotheses described below.
## Strategy
Ideally, each of these hypotheses would be examined carefully, both independently and in incremental relation to the others, before undertaking such a venture.

However, we are constrained by time and resources. We are therefore taking a measured “leap of faith”: placing our trust in a combination of intermediary peer-reviewed findings and many years of conceptual experimentation.

We provide a complete specification sheet of the envisioned approach meant to be fed to an AI agent. The AI agent is tasked to implement the approach and run extensive experimentation in order to build the sufficient understanding required to surface the aforementioned envisioned novel results.
# Hypothesis Core
## Current Paradigm
Modern AI systems draw from only a small subset of the broader landscape of computational methods.

For instance, although the space of computational search and optimization methods is vast, today’s dominant AI systems rely heavily on gradient-based optimization.

This suggests that, so far, value has been most readily attainable by focusing on this relatively narrow subset of techniques.

However, the research community remains uncertain about the ultimate range of value that current popular research trajectories may yield.
## Hypothesized Limitation
Gradient-based optimization is tightly coupled to the data distribution.

Within this paradigm, computational information from the data distribution is funneled directly into a model’s representation space.

Our core hypothesis is that there is untapped value in allowing the search/optimization process to explore beyond the confines of data space.

This hypothesis is rooted in the real-world observation that creativity often emerges from unpopular, unconventional, or otherwise underexplored trajectories.
## Proposed Remediation
### Evolutionary Algorithms
Our best bet for executing this vision is evolutionary algorithms.

In this paradigm, data is relegated to a regularizing role, while the representation space is perturbed through random search.
### Downside & Remediation
A key downside of evolutionary algorithms is that data influences representation-space formation only indirectly. As a result, information originating from the data distribution is incorporated into the representation space more slowly, less efficiently, and with greater noise than in gradient-based optimization.

Practically, this suggests that we should first extract as much signal as possible using gradient-based methods, reserving evolutionary search for the kinds of exploration that gradients are poorly suited to perform.
# Specification Sheet
## Challenge Overview
The challenge is to imitate human behaviour, using both gradient-based methods and an evolutionary algorithm, at a higher level of fidelity than gradient-based methods alone are able to.
## Formulation
For each task,
## Overview
The proposed evolutionary algorithm is a genetic algorithm that maintains a population of a fixed number (N) of agents.

Each agent performs three distinct roles: generator, discriminator and mutator.
## Neural Networks
Each agent evolves its own neural network.



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

X Generational inheritance

Generational inheritance transfers genotype, final environment state, final memory state, and cumulative lineage fitness from selected parents to offspring.

A child is initialized from the selected parent’s mutated genotype, resumes from the parent’s stored nonterminal environment state, resumes from the parent’s final memory state, and starts with inherited fitness equal to the parent’s cumulative lineage fitness.

The child is evaluated for E actions, accumulating local reward. If termination occurs before E is exhausted under fixed E=k, reset the environment and use the remaining actions; otherwise descendants of terminal agents start from a fresh environment. After evaluation, store the child’s final env state and memory state, and set fitness = inherited fitness + local reward. Selection is based on this cumulative fitness.


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

