# clockwork RNNs

[Clockwork RNNs](https://arxiv.org/pdf/1402.3511v1.pdf?) use multiple RNN blocks at different timescales to deal with the vanishing gradient problem. They were shown to work well for speech recognition and sequence generation, but I hypothesize that they are ill-suited for many other sequence modeling tasks. In this repository, I will not only implement traditional clockwork RNNs, but also experiment with related architectures.

# Traditional CWRNNs

In a traditional CWRNN, each sub-block in the RNN has a different period `p`. At timesteps `t` where `t%p = 0`, we update the sub-block's state `s` using the traditional RNN update: `s = tanh(W*[inS; in] + b)`. Here, `inS` is a concatenation of the old state `s` with the states of all the "slower" blocks. The `in` vector is the input at time `t`.

The instant I saw the CWRNN update formula, I perceived a flaw in its design: a block with a high period does not receive information about every input. For instance, if I trained a CWRNN on a text corpus and gave it the string `hello`, a sub-block with period 2 would only see `hlo`. Since slower blocks only receive information from even slower blocks, using exponential periods (2, 4, 8, ...) makes it impossible for a block with period `p` to receive information about inputs for times `t` where `t%p != 0`. For tasks like text modeling, I hypothesize that this limitation would prevent slower blocks from capturing any long-term contextual information.

The tasks in the original CWRNN paper are almost "special" in that they do not suffer from the CWRNN's input-visibility limitation. The first task is sequence prediction *with no inputs*. Obviously, since the RNN has no inputs, the input-visibility limitation would not be manifest. The second task, speech recognition, lends itself naturally to CWRNNs despite their limitations. The reason is that speech (or any sound) is a sum of periodic functions. Thus, if you only sample every `p` timesteps from a speech signal, you still get some information about the signal. Essentially, the slower blocks in such an environment are seeing downsampled audio data.

# Fully-Connected CWRNNs

To mitigate the input-visibility limitation of CWRNNs, I propose a minor change. In my new architecture, fully-connected CWRNN, sub-blocks receive information from *all* sub-blocks, whether faster or slower. This way, a sub-block can learn about skipped timesteps by looking at the state of a faster sub-block which did not miss those inputs. For text modeling, it could be seen how this architecture might make sense: a rapid sub-block could process words, a slower sub-block could process sentences based on those words, etc.
