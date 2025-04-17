# BayesFlow

BayesFlow is a Python library for simulation-based **Amortized Bayesian Inference** with neural networks.
It provides users and researchers with:

- A user-friendly API for rapid Bayesian workflows
- A rich collection of neural network architectures
- Multi-backend support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)

BayesFlow (version 2+) is designed to be a flexible and efficient tool that enables rapid statistical inference
fueled by continuous progress in generative AI and Bayesian inference.

## Conceptual Overview

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/bayesflow_landing_dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="_static/bayesflow_landing_light.jpg">
        <img alt="Overview graphic on using BayesFlow. It is split in three columns: 1. Choose your backend: BayesFlow is based on Keras, so you can choose PyTorch, TensorFlow or JAX. 2. Define your simulator: You specify your simulator in Python, and use it to generate simulated data. 3. Choose your algorithm: You define a generative neural network that you can use for estimation after training." src="_static/bayesflow_landing_dark.jpg">
</picture>
</div>

A cornerstone idea of amortized Bayesian inference is to employ generative
neural networks for parameter estimation, model comparison, and model validation
when working with intractable simulators whose behavior as a whole is too
complex to be described analytically.

## Getting Started

Using the high-level interface is easy, as demonstrated by the minimal working example below:

```python
import bayesflow as bf

workflow = bf.BasicWorkflow(
    inference_network=bf.networks.FlowMatching(),
    summary_network=bf.networks.TimeSeriesTransformer(),
    inference_variables=["parameters"],
    summary_variables=["observables"],
    simulator=bf.simulators.SIR()
)

history = workflow.fit_online(epochs=50, batch_size=32, num_batches_per_epoch=500)

diagnostics = workflow.plot_default_diagnostics(test_data=300)
```

For an in-depth exposition, check out our walkthrough notebooks below.

1. [Linear regression starter example](_examples/Linear_Regression_Starter.ipynb)
2. [From ABC to BayesFlow](_examples/From_ABC_to_BayesFlow.ipynb)
3. [Two moons starter example](_examples/Two_Moons_Starter.ipynb)
4. [Rapid iteration with point estimators](_examples/Lotka_Volterra_point_estimation_and_expert_stats.ipynb)
5. [SIR model with custom summary network](_examples/SIR_Posterior_Estimation.ipynb)
6. [Bayesian experimental design](_examples/Bayesian_Experimental_Design.ipynb)
7. [Simple model comparison example](_examples/One_Sample_TTest.ipynb)

More tutorials are always welcome! Please consider making a pull request if you have a cool application that you want to contribute.

## Install

```{eval-rst}
.. tab-set::

    .. tab-item:: pip

        The v2 version is not available on PyPI yet, please install from source.

    .. tab-item:: source

        .. code-block:: bash

            pip install git+https://github.com/bayesflow-org/bayesflow.git
```

### Backend

To use BayesFlow, you will also need to install one of the following machine learning backends.
Note that BayesFlow **will not run** without a backend.

- [Install JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [Install PyTorch](https://pytorch.org/get-started/locally/)
- [Install TensorFlow](https://www.tensorflow.org/install)

If you don't know which backend to use, we recommend JAX as it is currently the fastest backend.

Once installed, [set the backend environment variable as required by keras](https://keras.io/getting_started/#configuring-your-backend) to one of the following:

- `KERAS_BACKEND=jax`
- `KERAS_BACKEND=torch`
- `KERAS_BACKEND=tensorflow`

For example, to set the backend to `jax`, you can use one of the following:

```{eval-rst}
.. tab-set::

    .. tab-item:: Python

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"
            import bayesflow

    .. tab-item:: Conda

        .. code-block:: bash

            conda env config vars set KERAS_BACKEND=jax

    .. tab-item:: Shell

        .. code-block:: bash

            export KERAS_BACKEND=jax
```

**Caution:** Some development environments (e.g., VSCode or PyCharm) can silently overwrite environment variables. If you have set your backend as an environment variable and you still get keras-related import errors when loading BayesFlow, these IDE shenanigans might be the culprit. Try setting the keras backend in your Python script via `import os; os.environ["KERAS_BACKEND"] = "<YOUR-BACKEND>"`.

## Contributing

To contribute to BayesFlow, please check out the [git repository](https://github.com/bayesflow-org/bayesflow)

### Reporting Issues

If you encounter any issues, please don't hesitate to open an issue here on [Github](https://github.com/bayesflow-org/bayesflow/issues) or ask questions on our [Discourse Forums](https://discuss.bayesflow.org/).

## Getting Help

Please use the [BayesFlow Forums](https://discuss.bayesflow.org/) for any BayesFlow-related questions and discussions, and [GitHub Issues](https://github.com/bayesflow-org/bayesflow/issues) for bug reports and feature requests.

## Citing BayesFlow

You can cite BayesFlow along the lines of:

- We approximated the posterior using neural posterior estimation (NPE) with learned summary statistics (Radev et al., 2020), as implemented in the BayesFlow framework for amortized Bayesian inference (Radev et al., 2023a).
- We approximated the likelihood using neural likelihood estimation (NLE) without hand-crafted summary statistics (Papamakarios et al., 2019), leveraging its implementation in BayesFlow for efficient and flexible inference.

1. Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023a). BayesFlow: Amortized Bayesian workflows with neural networks. *The Journal of Open Source Software, 8(89)*, 5702.([arXiv](https://arxiv.org/abs/2306.16015))([JOSS](https://joss.theoj.org/papers/10.21105/joss.05702))
2. Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. *IEEE Transactions on Neural Networks and Learning Systems, 33(4)*, 1452-1466. ([arXiv](https://arxiv.org/abs/2003.06281))([IEEE TNNLS](https://ieeexplore.ieee.org/document/9298920))
3. Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., & Bürkner, P.-C. (2023b). JANA: Jointly amortized neural approximation of complex Bayesian models. *Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, 216*, 1695-1706. ([arXiv](https://arxiv.org/abs/2302.09125))([PMLR](https://proceedings.mlr.press/v216/radev23a.html))

**BibTeX:**

```
@article{bayesflow_2023_software,
  title = {{BayesFlow}: Amortized {B}ayesian workflows with neural networks},
  author = {Radev, Stefan T. and Schmitt, Marvin and Schumacher, Lukas and Elsemüller, Lasse and Pratz, Valentin and Schälte, Yannik and Köthe, Ullrich and Bürkner, Paul-Christian},
  journal = {Journal of Open Source Software},
  volume = {8},
  number = {89},
  pages = {5702},
  year = {2023}
}

@article{bayesflow_2020_original,
  title = {{BayesFlow}: Learning complex stochastic models with invertible neural networks},
  author = {Radev, Stefan T. and Mertens, Ulf K. and Voss, Andreas and Ardizzone, Lynton and K{\"o}the, Ullrich},
  journal = {IEEE transactions on neural networks and learning systems},
  volume = {33},
  number = {4},
  pages = {1452--1466},
  year = {2020}
}

@inproceedings{bayesflow_2023_jana,
  title = {{JANA}: Jointly amortized neural approximation of complex {B}ayesian models},
  author = {Radev, Stefan T. and Schmitt, Marvin and Pratz, Valentin and Picchini, Umberto and K\"othe, Ullrich and B\"urkner, Paul-Christian},
  booktitle = {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = {1695--1706},
  year = {2023},
  volume = {216},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

## FAQ

-------------

**Question:**
I am starting with Bayesflow, which backend should I use?

**Answer:**
We recommend JAX as it is currently the fastest backend.

-------------

**Question:**
I am getting `ModuleNotFoundError: No module named 'tensorflow'` when I try to import BayesFlow.

**Answer:**
One of these applies:
- You want to use tensorflow as your backend, but you have not installed it.
See [here](https://www.tensorflow.org/install).


- You want to use a backend other than tensorflow, but have not set the environment variable correctly.
See [here](https://keras.io/getting_started/#configuring-your-backend).


- You have set the environment variable, but it is not being picked up by Python.
This can happen silently in some development environments (e.g., VSCode or PyCharm).
Try setting the backend as shown [here](https://keras.io/getting_started/#configuring-your-backend)
in your Python script via `os.environ`.

-------------

**Question:**
What is the difference between Bayesflow 2.0+ and previous versions?

**Answer:**
BayesFlow 2.0+ is a complete rewrite of the library. It shares the same
overall goals with previous versions, but has much better modularity
and extensibility. What is more, the new BayesFlow has multi-backend support via Keras3,
while the old version was based on TensorFlow.

-------------

**Question:**
I still need the old BayesFlow for some of my projects. How can I install it?

**Answer:**
You can find and install the old Bayesflow version via the `stable-legacy` branch on GitHub.

-------------

## Awesome Amortized Inference

If you are interested in a curated list of resources, including reviews, software, papers, and other resources related to amortized inference, feel free to explore our [community-driven list](https://github.com/bayesflow-org/awesome-amortized-inference).

## Acknowledgments

This project is currently managed by researchers from Rensselaer Polytechnic Institute, TU Dortmund University, and Heidelberg University. It is partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation, Project 528702768). The project is further supported by Germany's Excellence Strategy -- EXC-2075 - 390740016 (Stuttgart Cluster of Excellence SimTech) and EXC-2181 - 390900948 (Heidelberg Cluster of Excellence STRUCTURES), as well as the Informatics for Life initiative funded by the Klaus Tschira Foundation.

## License \& Source Code

BayesFlow is released under {mainbranch}`MIT License <LICENSE>`.
The source code is hosted on the public [GitHub repository](https://github.com/bayesflow-org/bayesflow).

Indices
-------

* {ref}`genindex`
* {ref}`modindex`


```{toctree}
:maxdepth: 0
:titlesonly:
:hidden:

examples
api/bayesflow
about
Contributing <contributing>
Developer docs <development/index>
```
