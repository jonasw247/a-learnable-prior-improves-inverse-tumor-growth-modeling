# A Learnable Prior Improves Inverse Tumor Growth Modeling

Software of the publication: https://arxiv.org/pdf/2403.04500


![image](https://github.com/jonasw247/a-learnable-prior-improves-inverse-tumor-growth-modeling/assets/13008145/09c62318-2918-41ec-bcb3-2459b83a7ea3)
First, a DL network predicts a prior of tumor model parameters (orange) based
on imaging data. Second, this prior is used as an initialization for the subsequent sampling-based optimization
strategy (blue dots) to predict the final tumor model parameters, leading to the full modeling of tumor concentration.

# Prior Network Architecture
![image](https://github.com/jonasw247/a-learnable-prior-improves-inverse-tumor-growth-modeling/assets/13008145/e2ab9663-3c67-45cd-97f6-c847caaf6049)

A design of the inverse model architecture close to a ResNet. It takes as an input the binary brain tumor segmentations and outputs tumor growth parameters. The structure was the same as by Ezhov et al. [^1]

# Please Cite:
@article{weidner2024learnable,
  title={A Learnable Prior Improves Inverse Tumor Growth Modeling},
  author={Weidner, Jonas and Ezhov, Ivan and Balcerak, Michal and Metz, Marie-Christin and Litvinov, Sergey and Kaltenbach, Sebastian and Feiner, Leonhard and Lux, Laurin and Kofler, Florian and Lipkova, Jana and others},
  journal={arXiv preprint arXiv:2403.04500},
  year={2024}
}

# References
[^1]: Ezhov, Ivan, et al. "Learn-Morph-Infer: a new way of solving the inverse problem for brain tumor modeling." Medical Image Analysis 83 (2023): 102672.
https://github.com/IvanEz/learn-morph-infer
