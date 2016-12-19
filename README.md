# clockwork RNNs

[Clockwork RNNs](https://arxiv.org/pdf/1402.3511v1.pdf?) use multiple RNN blocks at different timescales to deal with the vanishing gradient problem. They were shown to work well for speech recognition and sequence generation, but I hypothesize that they are ill-suited for many other sequence modeling tasks. In this repository, I will not only implement traditional clockwork RNNs, but also experiment with using them for sequence modeling tasks like text prediction in which the timesteps are decorrelated and irregularly spaced.
