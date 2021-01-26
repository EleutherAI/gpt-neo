---
title: 'GPTNeo: Distributed Training for Large Scale Language Models'
tags:
  - Python
  - Tensorflow
  - Mesh Tensorflow
  - Model parallelism
  - Language modelling
authors:
  - name: Sid Black
    affiliation: 1
  - name: Leo Gao
    affiliation: 1
  - name: Phil Wang
    affiliation: 1
affiliations:
 - name: EleutherAI
   index: 1
date: 26 January 2021
bibliography: paper.bib

---

# Summary

Training very large language models across many devices requires techniques 
like model parallelism to be able to fit the model parameters within each device's
memory. 

`GPTNeo` is a Mesh-Tensorflow [@Shazeer:2018] based Python package for training 
language models similar to GPT2 [@Radford:2019] and GPT3 [@Brown:2020] of up to 
tens of billions of parameters on both GPUs and Google Cloud TPUs. `GPTNeo` supports 
a wide range of model architecture features like local attention, efficient 
attention [@Shen:2018], Mixture of Experts [@Shazeer:2017], Axial positional 
embedding [@Ho:2019], and Masked language modeling. `GPTNeo` also features integration 
with the Sacred experiment management tool [@Greff:2017].

`GPTNeo` was built for the EleutherAI GPT-3 replication project, and to help enable 
large scale language modeling experiments on TPUs. It has already been used in 
scientific publications like [@Gao:2020], and will be used in future EleutherAI
research. 

# Acknowledgements

The authors would like to thank TensorFlow Research Cloud for providing the 
computational resources necessary for testing GPTNeo during development. 

# References
