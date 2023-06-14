<!-- markdownlint-disable MD013 -->
# Reinforcement Learning from Human Feedback w/ TRL

[![formatter | docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![style | google](https://img.shields.io/badge/%20style-google-3666d6.svg)](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)

> :warning: **The pipeline is under construction. You might need to change a few things as explained below.**

Reinforcement learning from Human Feedback (RLHF) has been widely adopted in
recent times by the Artificial Intelligence industry.

Here, we build a simple RLHF pipeline using the :hugs:[`trl`] library.

## Usage

`rlhf-trl` supports Python 3.10+ with Poetry 1.5+.

Install necessary dependencies:

```sh
# Using pip
pip install -r requirements.txt

# Using poetry
poetry install
```

Or install development dependencies:

```sh
# Using pip
pip install -r requirements-dev.txt

# Using poetry
poetry install --with dev
```

Check available arguments by running the follwoing:

```sh
python scripts/train.py --help
```

## Customize

In order to customize the pipeline to your own use case, you'll have to change
a few things including:

- Loading and preparing your data [`data.py`]
  - Modify the `load_data` function as needed.
  - Modify the `collator` function to suite your data.

- Evaluating results [`evaluate.py`]
  - Moidfy the `evaluate` function to fit your data & model requirements.

[`data.py`]: ./src/rlhf_trl/data.py
[`evaluate.py`]: ./scripts/evaluate.py

## Training

To train the PPO model given a pre-trained supervised fine-tuned model and
pre-trained reward model, run the following command:

```sh
accelerate launch --config_file configs/accelerate.yml scripts/train.py \
    --sft_model_name <path_or_hf_name_of_sft_model> \
    --tokenizer_name <path_or_hf_name_of_tokenizer> \
    --reward_model_name <path_or_hf_name_of_reward_model> \
    --dataset_path <path/to/dataset>
```

**NOTE:** Depending your dataset source, your `dataset_path` argument can either
be from huggingface hub or local path.

## Evaluation

To evaluate the performance of your SFT model over the PPO model, run the following
command:

```sh
python -m scripts.evaluate \
    --sft_model_name <path_or_hf_name_of_sft_model> \
    --tokenizer_name <path_or_hf_name_of_tokenizer> \
    --reward_model_name <path_or_hf_name_of_reward_model> \
    --dataset_path <path/to/dataset>
    --ppo_model_name <path_or_hf_name_of_ppo_model> \
    --eval_name ppo_v2_eval
```

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (MIT)

This project is opened under the [MIT][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[`trl`]: https://github.com/lvwerra/trl
[original repository]: https://github.com/victor-iyi/rlhf-trl
[issues]: https://github.com/victor-iyi/rlhf-trl/issues
[license]: ./LICENSE
