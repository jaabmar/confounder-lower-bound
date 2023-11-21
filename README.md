# Testing for Hidden Confounding in Observational Studies

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2202.01315)
[![Python 3.11.5](https://img.shields.io/badge/python-3.11.5-blue.svg)](https://python.org/downloads/release/python-3115/)
[![Pytorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the Python implementation of [Hidden yet quantifiable: A lower bound for confounding strength using randomized trials](https://arxiv.org/abs/2202.01315).

* [Overview](#overview)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Citation](#citation)

## Overview

<p align="center">
<img align="middle" src="motivating_example.png"  alt="An illustrative example of the drug regulatory process: our lower bound allows taking proactive measures to address the unobserved confounding problem."/>
</p>

Provide a more in-depth paragraph about your project. What does it do? What problem does it solve? Why is this project useful? (TODO)

## Getting Started

### Dependencies

List with the necessary dependencies for your project:

- Python 3.11.5
- Numpy 1.24.3
- Scipy 1.10.1
- MLinsights 0.4.664
- Scikit-learn 1.3.0
- Statsmodels 0.13.5
- Pandas 1.5.3
- XGBoost 1.7.3
- Scikit-uplift 0.5.1
- Quantile-forest 1.2.0
- Torch 2.0.1
- CVXPY 1.3.1

### Installing

To set up the environment and install the package, follow these steps:

```bash
conda create -n myenv python=3.11.5
conda activate myenv
pip install --upgrade pip
pip install -e .         
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/jaabmar/confounder-lower-bound.git
```

## Usage

Example of using the package:

```bash
import numpy as np
from scipy.stats import bootstrap
from sklearn.ensemble import RandomForestClassifier

from test_confounding.ATE.ate_bounds import BootstrapSensitivityAnalysis
from test_confounding.CATE.cate_bounds import MultipleCATEBoundEstimators
from test_confounding.CATE.utils_cate_test import compute_bootstrap_variance
from test_confounding.datasets import synthetic
from test_confounding.test import run_multiple_cate_hypothesis_test
from test_confounding.utils_evaluate import e_x_func

data = synthetic.Synthetic(
        num_examples_obs =  20000,
        num_examples_rct = 5000,
        gamma_star = 5.0,
        effective_conf = 1.0,
        sigma_y = 0.01,
        seed=42,    
)

x_rct, t_rct, y_rct = (
  data.rct.x,
  data.rct.t,
  data.rct.y,
)

x_obs, t_obs, y_obs = (
    data.x,
    data.t,
    data.y,
)

ate = y_rct[t_rct == 1].mean() - y_rct[t_rct == 0].mean()
ate_variance = compute_bootstrap_variance(Y=y_rct, T=t_rct, n_bootstraps=n_bootstrap, arm=None)

bounds_estimator = MultipleCATEBoundEstimators(
    gammas=user_conf, n_bootstrap=n_bootstrap
)

bounds_estimator.fit(x_obs=x_obs, t_obs=t_obs, y_obs=y_obs)

results_dict_cate = run_multiple_cate_hypothesis_test(bounds_estimator = bounds_estimator, ate = ate, ate_variance = ate_variance, alpha = 5.0, x_rct = x_rct, user_conf = user_conf, verbose = False)
```

For detailed tutorials with synthetic and semi-synthetic data, refer to [Tutorial (Part 1)](src/synthetic.ipynb) and [Tutorial (Part 2)](src/semi_synthetic.ipynb).

## Contributing

We welcome contributions to improve this project. Here's how you can contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries, please reach out:

- Javier Abad Martinez - [javier.abadmartinez@ai.ethz.ch](mailto:javier.abadmartinez@ai.ethz.ch)
- Piersilvio de Bartolomeis - [pdebartol@ethz.ch](mailto:pdebartol@ethz.ch)

## Citation

If you find this code useful, please consider citing our paper:
 ```
@inproceedings{debartolomeis2023hidden,
  title={Hidden yet quantifiable: A lower bound for confounding strength using randomized trials},
  author={de Bartolomeis, Piersilvio^*, and Abad Martinez, Javier^* and Donhauser, Konstantin and Yang, Fanny},
  booktitle={Under Review},
  year={2023}
}
```