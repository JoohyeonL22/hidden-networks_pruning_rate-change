
--------------------------------------------------------------------------------------------
# Changing pruning rate by comparing in all layers.
This project is based on 'What's hidden in a randomly weighted neural network?'(Ramanujan et al.)

Paper: https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11113866&googleIPSandBox=false&mark=0&ipRange=false&accessgl=Y&language=ko_KR&hasTopBanner=true

baseline arxiv link ('What's hidden in a randomly weighted neural network?'): https://arxiv.org/abs/1911.13299

# A problem from original study on network pruning using edge-popup algorithm
In 'What's hidden in a randomly weighted neural network?', There is pruning on conv, FC layer.

In original project, comparing score of edges to pruning works on only same layers.

Hence, I changed several codes to comparing edges between  other layers.

I changed code in 'main.py', 'utils/conv_type.py', and 'models/frankle.py'.
- Default setting is k=0.5. If you change pruning rate, you should change k1, k2, k3, k4, k5 to your own rate and input the same pruning rate as argument.
- There will be remained top-k% score edge and be pruned other edges.
- It works on only Conv2 model.

# Result
<img width="100%" src="https://github.com/JoohyeonL22/Light-the-Spire/assets/106375416/f46a2a45-292f-4142-a378-7ff05febfb93"/>

If k value set near 0 or 1, the accuracy is better than the original method.

- SGD means a not pruned network.
- ukn means use ukn as weight initialization  and usc means use usc as weight initialization.
- _ ori means original method and _ k means my own improved method.
