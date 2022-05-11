# Deep Coordination Graphs
This is a simple implementation of Deep Coordination Graphs (DCG) (Boehmer et al. 2020). The core algorithm is implemented, but it does not include

* Training with privileged information (DCG-S)
* Recurrent observation history embedding

# To do
- [ ] Add a script for training
- [ ] Add gifs of learned behaviors
- [x] Factored edges use a single network
- [x] Implement parameter sharing more explicitly using batch inference
- [x] Implement action selection by message passing
- [x] Implement factored edges
