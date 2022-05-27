# What's hidden in a randomly weighted neural network?

by Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari

arxiv link: https://arxiv.org/abs/1911.13299

--------------------------------------------------------------------------------------------
This project is based on What's hidden in a randomly weighted neural network?

In What's hidden in a randomly weighted neural network?, There is pruning on conv, FC layer
There will be remained top-k% score edge and pruned other edges.
But, comparing score of edges to pruning on only same layers.
So, I changed that compares with edges in other layers.
It works on only Conv2 model.

I changed code in main.py, utils/conve_type.py, models/frankle.py
default setting is k=0.5. if you change pruning rate, you should change k1, k2, k3, k4, k5 to your own rate and input same pruning rate as argument.

