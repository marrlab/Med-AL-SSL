# Med-AL-SSL [![CodeFactor](https://www.codefactor.io/repository/github/ahmadqasim/med-al-ssl/badge)](https://www.codefactor.io/repository/github/ahmadqasim/med-al-ssl) [![Known Vulnerabilities](https://snyk.io/test/github/AhmadQasim/Med-AL-SSL/badge.svg?targetFile=requirements.txt)](https://snyk.io/test/github/AhmadQasim/Med-AL-SSL?targetFile=requirements.txt)
Repository for implementation of active learning and semi-supervised learning algorithms and applying it to medical imaging datasets

## Active Learning algorithms
* Least Confidence Sampling [1]
* Margin Sampling [1]
* Ratio Sampling [1]
* Maximum Entropy Sampling [1]
* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [2]
* Learning Loss for Active Learning [3]
* BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [4]

## Semi-Supervised Learning algorithms
* Pseudo Labeling [5]
* Autoencoder [5]
* A Simple Framework for Contrastive Learning of Visual Representations [6]
* FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence [7]

## Requirements
```
numpy>=1.18.5
torch>=1.4.0
torchvision>=0.5.0
sklearn>=0.0
scikit-learn>=0.23.1
pandas>=1.0.4
torchlars>=0.1.2
Pillow>=7.1.2
matplotlib>=3.2.1
```

## Usage
### Arguments
```
(In progress)
```

## References
[1] Settles, B. (2009). Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences.

[2] Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning (pp. 1050-1059).

[3] Yoo, D., & Kweon, I. S. (2019). Learning loss for active learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 93-102).

[4] Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning. In Advances in Neural Information Processing Systems (pp. 7026-7037).

[5] Van Engelen, J. E., & Hoos, H. H. (2020). A survey on semi-supervised learning. Machine Learning, 109(2), 373-440.

[6] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.

[7] Sohn, K., Berthelot, D., Li, C. L., Zhang, Z., Carlini, N., Cubuk, E. D., ... & Raffel, C. (2020). Fixmatch: Simplifying semi-supervised learning with consistency and confidence. arXiv preprint arXiv:2001.07685.
