# Deep Coordination Graphs
This is a simple implementation of Deep Coordination Graphs (DCG) (Boehmer et al. 2020). The core algorithm is implemented, but it does not include

* Training with privileged information (DCG-S)
* Recurrent observation history embedding

# To do
- [ ] Implement parameter sharing more explicitly using batch inference
- [ ] Implement action selection by message passing
- [ ] Add a script for training
- [ ] Add gifs of learned behaviors
- [ ] Factored edges use a single network
- [x] Implement factored edges
