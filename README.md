# Source code for "How large language models judge and influence human cooperation"

Preprint available here: https://arxiv.org/abs/2507.00088

This folder contains all the source code to reproduce the results of the paper.

## Requirements

The code base is partially done in Julia 1.10 (cooperation model) and partially done in Python 3 (LLM norm extraction and processing).

The required Julia packages can be added via the command

```
] add CairoMakie Colors LaTeXStrings NonlinearSolve Memoization ArnoldiMethod Distributions JSON StatsBase
```

The project also makes use of numpy and torch, and of the following standard python libraries: re, json, itertools and statistics.

## Code Structure and utilization

Evaluating an LLM consists of four steps: 1) dataset generation, 2) LLM prompting, 3) norm aggregation, and 4) norm evaluation via the cooperation model. We next detail each of these steps.

### Dataset generation

The code related to dataset generation is available in the prompt_dataset folder. In particular:
* **prompt_dataset/prompt_templates.py** - contain all the prompt templates, interventions and format instructions that are used to form the dataset, and can be modified as necessary;
* **prompt_dataset/prompt_elements_elements.json** - contains all the elements that are used to fill the dataset, and can be modified as necessary;
* **prompt_dataset/generate_prompt_dataset.ipynb** - contains the code necessary to create the prompt dataset. It first generates a prompt_elements.json file containing all possible combinations of prompt elements. It then iterates through all prompt templates using all combinations of prompt elements to generate prompt_dataset.json, which is then used to prompt LLMs.

As such, to generate the entire dataset from scratch run generate_prompt_dataset.ipynb in its entirety.

Importantly, prompt interventions require their own dataset, which is also generated under generate_prompt_dataset.ipynb, and must be individually prompted to the LLM following the process below.

## LLM prompting

The code related to prompting the LLMs is available in the LLMs, scripts and prompt_dataset folder. In order to run this code, it is necessary to add your own API keys for each model or, when using local LLMs, alter the directory for the model. Models besides the ones tested must be manually added similarly to the current implemented ones.

* **LLMs/** - contain the code necessary to implement each LLM, as well as init.py, which is used to initialize a given model.
* **scripts/evaluate.py** - contains the code necessary to prompt an LLM, generating a json file with the answer to each prompt
* **prompt_dataset/create_dataloader.py** - contains the code necessary to load the prompt dataset, used by evaluate.py

The responses of each model must then be put in results_llm for the norm aggregation step (see below). For reproducibility sake, results_llm.json already contains all the responses from the LLMs used in the paper.

## Norm aggregation

To aggregate the norms of each LLM, the prompt answers json generated from scripts/evaluate.py must be placed in the results_llm folder. Every model's answer placed there will be processed to extract their social norms from the prompt answers. The code related to this is placed in the prompt_dataset folder under:

* **prompt_dataset/norm_processor.ipynb** - contains the code to aggregate each LLM's answer to each prompt dataset into a norm, seperated also by subdataset, under all_llm_norms.json. It also generates the error response rate of each LLM, under all_llm_errors.json.

Running norm_processor.ipynb in its entirety will generate all_llm_norms.json, which is then loaded in julia and used in the cooperation model.

## Cooperation model

Most of the cooperation model code does the mathematical operations described in the paper, consisting of calculating the cooperation and reputation dynamics for a given set of parameters. The code is contained entirely in the cooperation_model folder, and its structure of the code is as follows:

* **cooperation_model/stats.jl** - The primary file used to run experiments. It contains all the parameters that the experiment uses, calls the relevant functions to obtain the metrics mentioned above, and calls the plotting functions;
* **cooperation_model/reputation.jl** - Contains all the code to calculate the reputation equilibrium, via ODEs, at a given strategy state;
* **cooperation_model/strategy.jl** - Contains all the code to calculate the full strategy Markov chain, using reputation.jl at each state to calculate transition probabilities. Also calculates all metrics at each state and packs them for stats.jl;
* **cooperation_model/plotter.jl** - Contains all code relative to plotting for each of the type of experiment in stats.jl;
* **cooperation_model/tests.jl** - Contains tests to verify the code base;
* **cooperation_model/utils.jl** - Contains utility code used throughout the code base, pertaining to data processing and some common mathematical functions such as obtaining the stationary distribution of the strategy Markov chain.
* **cooperation_model/norm_influence.jl** - Can be used to define additional norm influence functions, to study scenarios besides the one described in the paper where the LLM's social norm dominates the population.

To run the cooperation model, go to stats.jl and select the model parameters and LLMs to study, as well as the type of plot to do in the bottom of the code.

Running the Julia code produces a folder in the "results_cooperation/<foldername>" directory, where the data is then stored. This data is then read to make the plots. This data can then be re-read to remake plots without rerunning experiments, by setting the justGenPlots flag to true.