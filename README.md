# Tools for coding and decoding for spiking neural networks

Implementation of encoding and decoding methods for spiking neural networks.

The implementations are based on these papers:

[Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems](https://doi.org/10.1145/3546790.3546792)

[A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks](https://doi.org/10.1007/s11063-021-10562-2)

[Pulsewidth Modulation-Based Algorithm for Spike Phase Encoding and Decoding of Time-Dependent Analog Data](https://doi.org/10.1109/TNNLS.2019.2947380)

For the rate enconding, a snnTorch function is utilized, but I added a seed in order to control the
reproducibility of the results and the decoding of the spikes.

[snnTorch](https://snntorch.readthedocs.io/en/latest/index.html)

## Notes

The data must be already between [0 , 1].


I am still working on the implementations.




