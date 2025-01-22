# mlrose-ky: Machine Learning, Randomized Optimization, and SEarch

[![PyPI Version](https://img.shields.io/pypi/v/mlrose-ky?style=flat&label=PyPi&color=177BBB)](https://pypi.org/project/mlrose-ky/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mlrose-ky?style=flat&label=Downloads&color=177BBB)](https://pypi.org/project/mlrose-ky/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/dcb6d0168803406484009504a41ed466)](https://app.codacy.com/gh/knakamura13/mlrose-ky?utm_source=github.com&utm_medium=referral&utm_content=knakamura13/mlrose-ky&utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/994149f095134e27a9685e6e419961a9)](https://app.codacy.com/gh/knakamura13/mlrose-ky/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![wakatime](https://wakatime.com/badge/user/3e093fe5-ba7b-427a-b7f6-22e93ae34ce8/project/7953f1b0-1775-4be1-92c7-13ee3a8dcc81.svg)](https://wakatime.com/badge/user/3e093fe5-ba7b-427a-b7f6-22e93ae34ce8/project/7953f1b0-1775-4be1-92c7-13ee3a8dcc81)

`mlrose-ky` is a Python package for applying some of the most common randomized optimization and search algorithms to a range of different
optimization problems, over both discrete- and continuous-valued parameter spaces.

## Project Background

`mlrose-ky` is a fork of the [`mlrose-hiive`](https://github.com/hiive/mlrose) repository, which itself was a fork of the
original [`mlrose`](https://github.com/gkhayes/mlrose) repository.

The original `mlrose` was developed to support students of Georgia Tech's OMSCS/OMSA offering of CS 7641: Machine Learning.

Later, `mlrose-hiive` introduced a number of improvements (for example, the `Runners` submodule) and bug fixes on top of `mlrose`, though it
lacked documentation, contained some mysterious bugs and inefficiencies, and was unmaintained as of around 2022.

Today, `mlrose-ky` introduces additional improvements and bug fixes on top of `mlrose-hiive`. Some of these improvements include:

- Added documentation to every class, method, and function (i.e., descriptive docstrings, strong type-hints, and comments)
- New documentation available here: https://nkapila6.github.io/mlrose-ky/
- Increased test coverage from ~5% to 100%!
- Actively being maintained
- Fully backwards compatible with `mlrose-hiive`
- Optimized Python code with NumPy vectorization

## Main Features

This repository includes implementations of all randomized optimization algorithms taught in the course, as well as functionality to apply
these algorithms to integer-string optimization problems, such as N-Queens and the Knapsack problem; continuous-valued optimization
problems, such as the neural network weight problem; and tour optimization problems, such as the Travelling Salesperson problem. It also has
the flexibility to solve user-defined optimization problems.

### *Randomized Optimization Algorithms*

- Implementations of: hill climbing, randomized hill climbing, simulated annealing, genetic algorithm, and (discrete) MIMIC;
- Solve both maximization and minimization problems;
- Define the algorithm's initial state or start from a random state;
- Define your own simulated annealing decay schedule or use one of three pre-defined, customizable decay schedules: geometric decay,
  arithmetic decay, or exponential decay.

### *Problem Types*

- Solve discrete-value (bit-string and integer-string), continuous-value, and tour optimization (travelling salesperson) problems;
- Define your own fitness function for optimization or use a pre-defined function.
- Pre-defined fitness functions exist for solving the: One Max, Flip Flop, Four Peaks, Six Peaks, Continuous Peaks, Knapsack, Travelling
  Salesperson, N-Queens, and Max-K Color optimization problems.

### *Machine Learning Weight Optimization*

- Optimize the weights of neural networks, linear regression models, and logistic regression models using randomized hill climbing,
  simulated annealing, the genetic algorithm, or gradient descent;
- Supports classification and regression neural networks.

## Project Improvements and Updates

The `mlrose-ky` project is undergoing significant improvements to enhance code quality, documentation, and testing. Below is a list of tasks
that have been completed or are in progress:

1. **Fix Python Warnings and Errors**: All Python warnings and errors have been addressed, except for a few unavoidable ones like "duplicate
   code." ✅

2. **Add Python 3.10 Type Hints**: Type hints are being added to all function and method definitions, as well as method properties (
   e.g., `self.foo: str = 'bar'`), to improve code clarity and maintainability. ✅

3. **Enhance Documentation**: NumPy-style docstrings are being added to all functions and methods, with at least a one-line docstring at the
   top of every file summarizing its contents. This will make the codebase more understandable and easier to use for others. ✅

4. **Increase Test Coverage**: Tests are being added using Pytest, with a goal of achieving 100% code coverage to ensure the robustness of
   the codebase.  ✅

5. **Resolve TODO/FIXME Comments**: A thorough search is being conducted for any TODO, FIXME, or similar comments, and their respective
   issues are being resolved.  ✅

6. **Optimize Code**: Vanilla Python loops are being optimized where possible by vectorizing them with NumPy to enhance performance.

7. **Improve Code Quality**: Any other sub-optimal code, bugs, or code quality issues are being addressed to ensure a high standard of
   coding practices.  ✅

8. **Clean Up Codebase**: All commented-out code is being removed to keep the codebase clean and maintainable. ✅

## Installation

`mlrose-ky` was written in Python 3 and requires NumPy, SciPy, and Scikit-Learn (sklearn).

The latest version can be installed using `pip`:

```bash
pip install mlrose-ky --upgrade
```

Once it is installed, simply import it like so:

```python
import mlrose_ky as mlrose
```

## Documentation

The official `mlrose-ky` documentation can be found [here](https://nkapila6.github.io/mlrose-ky/).

A Jupyter notebook containing the examples used in the documentation is also
available [here](https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb).

## Licensing, Authors, Acknowledgements

`mlrose-ky` was forked from the `mlrose-hiive` repository, which was a fork of the original `mlrose` repository.

The original `mlrose` was written by Genevieve Hayes and is distributed under
the [3-Clause BSD license](https://github.com/gkhayes/mlrose/blob/master/LICENSE).

You can cite `mlrose-ky` in research publications and reports as follows:

- Nakamura, K. (2024).
  ***mlrose-ky: Machine Learning, Randomized Optimization, and SEarch package for Python***. https://github.com/knakamura13/mlrose-ky/.
  Accessed: *day month year*.

Please also keep the original authors' citations:

- Rollings, A. (2020).
  ***mlrose: Machine Learning, Randomized Optimization and SEarch package for Python, hiive extended remix***. https://github.com/hiive/mlrose. Accessed: *day month year*.
- Hayes, G. (2019).
  ***mlrose: Machine Learning, Randomized Optimization and SEarch package for Python***. https://github.com/gkhayes/mlrose. Accessed: *day
  month year*.

Thanks to David S. Park for the MIMIC enhancements (from https://github.com/parkds/mlrose).

BibTeX entries:

```bibtex
@misc{Nakamura24,
    author = {Nakamura, K.},
    title = {{mlrose-ky: Machine Learning, Randomized Optimization and SEarch package for Python}},
    year = 2024,
    howpublished = {\url{https://github.com/knakamura13/mlrose-ky/}},
    note = {Accessed: day month year}
}

@misc{Rollings20,
    author = {Rollings, A.},
    title = {{mlrose: Machine Learning, Randomized Optimization and SEarch package for Python, hiive extended remix}},
    year = 2020,
    howpublished = {\url{https://github.com/hiive/mlrose/}},
    note = {Accessed: day month year}
}

@misc{Hayes19,
    author = {Hayes, G.},
    title = {{mlrose: Machine Learning, Randomized Optimization and SEarch package for Python}},
    year = 2019,
    howpublished = {\url{https://github.com/gkhayes/mlrose/}},
    note = {Accessed: day month year}
}
```

## Collaborators

<!-- readme: collaborators -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/nkapila6">
                    <img src="https://avatars.githubusercontent.com/u/12816113?v=4" width="100;" alt="nkapila6"/>
                    <br />
                    <sub><b>Nikhil Kapila</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/knakamura13">
                    <img src="https://avatars.githubusercontent.com/u/20162718?v=4" width="100;" alt="knakamura13"/>
                    <br />
                    <sub><b>Kyle Nakamura</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/gitongah">
                    <img src="https://avatars.githubusercontent.com/u/39062444?v=4" width="100;" alt="gitongah"/>
                    <br />
                    <sub><b>Edwin Mbaabu</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: collaborators -end -->

## Contributors

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/knakamura13">
                    <img src="https://avatars.githubusercontent.com/u/20162718?v=4" width="100;" alt="knakamura13"/>
                    <br />
                    <sub><b>Kyle Nakamura</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/hiive">
                    <img src="https://avatars.githubusercontent.com/u/24660532?v=4" width="100;" alt="hiive"/>
                    <br />
                    <sub><b>hiive</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/gkhayes">
                    <img src="https://avatars.githubusercontent.com/u/24857299?v=4" width="100;" alt="gkhayes"/>
                    <br />
                    <sub><b>Dr Genevieve Hayes</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ChristopherBilg">
                    <img src="https://avatars.githubusercontent.com/u/3654150?v=4" width="100;" alt="ChristopherBilg"/>
                    <br />
                    <sub><b>Chris Bilger</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/nkapila6">
                    <img src="https://avatars.githubusercontent.com/u/12816113?v=4" width="100;" alt="nkapila6"/>
                    <br />
                    <sub><b>Nikhil Kapila</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/gitongah">
                    <img src="https://avatars.githubusercontent.com/u/39062444?v=4" width="100;" alt="gitongah"/>
                    <br />
                    <sub><b>Edwin Mbaabu</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/Agrover112">
                    <img src="https://avatars.githubusercontent.com/u/42321810?v=4" width="100;" alt="Agrover112"/>
                    <br />
                    <sub><b>Agrover112</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/domfrecent">
                    <img src="https://avatars.githubusercontent.com/u/12631209?v=4" width="100;" alt="domfrecent"/>
                    <br />
                    <sub><b>Dominic Frecentese</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/harrisonfloam">
                    <img src="https://avatars.githubusercontent.com/u/130672912?v=4" width="100;" alt="harrisonfloam"/>
                    <br />
                    <sub><b>harrisonfloam</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/AlexWendland">
                    <img src="https://avatars.githubusercontent.com/u/3949212?v=4" width="100;" alt="AlexWendland"/>
                    <br />
                    <sub><b>Alex Wendland</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/cooknl">
                    <img src="https://avatars.githubusercontent.com/u/5116899?v=4" width="100;" alt="cooknl"/>
                    <br />
                    <sub><b>CAPN</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/KevinJBoyer">
                    <img src="https://avatars.githubusercontent.com/u/31424131?v=4" width="100;" alt="KevinJBoyer"/>
                    <br />
                    <sub><b>Kevin Boyer</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/parkds">
                    <img src="https://avatars.githubusercontent.com/u/10566599?v=4" width="100;" alt="parkds"/>
                    <br />
                    <sub><b>David</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/davideasaf">
                    <img src="https://avatars.githubusercontent.com/u/11589969?v=4" width="100;" alt="davideasaf"/>
                    <br />
                    <sub><b>David Asaf</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jfs42">
                    <img src="https://avatars.githubusercontent.com/u/43157283?v=4" width="100;" alt="jfs42"/>
                    <br />
                    <sub><b>Jason Seeley</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/sareini">
                    <img src="https://avatars.githubusercontent.com/u/26151060?v=4" width="100;" alt="sareini"/>
                    <br />
                    <sub><b>Muhammad Sareini</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/nibelungvalesti">
                    <img src="https://avatars.githubusercontent.com/u/9278042?v=4" width="100;" alt="nibelungvalesti"/>
                    <br />
                    <sub><b>nibelungvalesti</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/tadmorgan">
                    <img src="https://avatars.githubusercontent.com/u/4197132?v=4" width="100;" alt="tadmorgan"/>
                    <br />
                    <sub><b>W. Tad Morgan</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/mjschock">
                    <img src="https://avatars.githubusercontent.com/u/1357197?v=4" width="100;" alt="mjschock"/>
                    <br />
                    <sub><b>Michael Schock</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jlm429">
                    <img src="https://avatars.githubusercontent.com/u/10093986?v=4" width="100;" alt="jlm429"/>
                    <br />
                    <sub><b>John Mansfield</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/dstrube1">
                    <img src="https://avatars.githubusercontent.com/u/7396679?v=4" width="100;" alt="dstrube1"/>
                    <br />
                    <sub><b>David Strube</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/austin-bowen">
                    <img src="https://avatars.githubusercontent.com/u/4653828?v=4" width="100;" alt="austin-bowen"/>
                    <br />
                    <sub><b>Austin Bowen</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/bspivey">
                    <img src="https://avatars.githubusercontent.com/u/6569966?v=4" width="100;" alt="bspivey"/>
                    <br />
                    <sub><b>Ben Spivey</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/dreadn0ught">
                    <img src="https://avatars.githubusercontent.com/u/31293924?v=4" width="100;" alt="dreadn0ught"/>
                    <br />
                    <sub><b>David</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/brokensandals">
                    <img src="https://avatars.githubusercontent.com/u/328868?v=4" width="100;" alt="brokensandals"/>
                    <br />
                    <sub><b>Jacob Williams</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/vpipkt">
                    <img src="https://avatars.githubusercontent.com/u/7798319?v=4" width="100;" alt="vpipkt"/>
                    <br />
                    <sub><b>Jason T Brown</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/zerolinux5">
                    <img src="https://avatars.githubusercontent.com/u/1896803?v=4" width="100;" alt="zerolinux5"/>
                    <br />
                    <sub><b> Jesus Magana</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ksbeattie">
                    <img src="https://avatars.githubusercontent.com/u/1534843?v=4" width="100;" alt="ksbeattie"/>
                    <br />
                    <sub><b>Keith Beattie</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/MatthewCaseres">
                    <img src="https://avatars.githubusercontent.com/u/43053796?v=4" width="100;" alt="MatthewCaseres"/>
                    <br />
                    <sub><b>Matthew Caseres</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/cbhyphen">
                    <img src="https://avatars.githubusercontent.com/u/12734117?v=4" width="100;" alt="cbhyphen"/>
                    <br />
                    <sub><b>cbhyphen</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/dyodx">
                    <img src="https://avatars.githubusercontent.com/u/45729071?v=4" width="100;" alt="dyodx"/>
                    <br />
                    <sub><b>dyodx</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wyang36">
                    <img src="https://avatars.githubusercontent.com/u/5606561?v=4" width="100;" alt="wyang36"/>
                    <br />
                    <sub><b>Kira Yang</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->
