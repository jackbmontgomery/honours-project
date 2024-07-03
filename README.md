# Honours Project

## Holiday Work Plan: 1 July - 22 July

**The goal for the holiday is that I would like to have implemented a multi-agent environment with the encoding of some collaboration within the generative model of an agent**

### Week 1

#### Practice

- [x] [active inference from scratch](https://github.com/infer-actively/pymdp/blob/master/docs/notebooks/active_inference_from_scratch.ipynb)
- [x] [T-Maze Demo](https://github.com/infer-actively/pymdp/blob/master/docs/notebooks/tmaze_demo.ipynb)
- [x] [Cue Chaining](https://github.com/infer-actively/pymdp/blob/master/docs/notebooks/cue_chaining_demo.ipynb) Notebooks

#### Practical

- [ ] Implement learning of the A element in a gridworld with one clue location and an unknown reward location

#### Thoughts on environment

I am finding the foraging to be an interestesting environment. The ideas that are still unclear here and what the dyanmics of the collaboration are going to be.

## Getting carried away below

### Current idea on environemnt

I want to place objets and agents into an evironment and have the objective of each object needs to be moved back to the landing

#### Step by step plan to implement - Phase 1

**ALl of this can be implemented with no learning for now, I just want to define environment and model dynamics**

1. One agent in an environment that is aware that they need to visit a specific block
2. One agent aware that that they want to pick up an object and return it to the landing block
3. Multiple objects that all need to be moved back into the block

### Notes

When learning A (or another model element) the the difference between A_gm, qA and pA is not very clear at the momemnt. A -
