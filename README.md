# XOR_demo_tensorflow
This is an example to help teachers and students.

Implements the XOR network in Goodfellow et al (2016), chapter 6.
Provides support for TensorBoard.
The network does not train well with good practice initial values, probably
because its learning capacity is just barely able to represent the problem.

The network was first build using the layer() function which initialized weights
and biases to "good practice" values, but the network didn't train successfully.

The functions layer1() and layer2(), were added to allow one to initialize the weights
and biases to specifically chosen values. When the weights are initialized near the known
solution given by Goodfellow et al, the network converges.

TensorBoard output was added following instructions in the youtube video by
Dendelion Mane titled "Hands-on TensorBoard (TensorFlow Dev Summit 2017)"
at https://www.youtube.com/watch?v=eBbEDRsCmv4.
