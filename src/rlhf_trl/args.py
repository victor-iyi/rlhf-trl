from dataclasses import dataclass
from dataclasses import field

from transformers import HfArgumentParser


@dataclass
class ScriptArgs:
    """The name of the Causal LM wwe wish to fine-tune with PPO."""

    model_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the Causal LM wwe wish to fine-tune with PPO.',
        },
    )

    tokenizer_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the tokenizer to use.',
        },
    )

    reward_model_name: str | None = field(
        default='',
        metadata={
            'help': 'The name of the reward model to use.',
        },
    )

    dataset_path: str | None = field(
        default='',
        metadata={
            'help': 'The path to the dataset.',
        },
    )

    log_with: str | None = field(
        default='wandb',
        metadata={
            'help': 'The logger to use.',
        },
    )

    learning_rate: float | None = field(
        default=1e-5,
        metadata={
            'help': 'The learning rate to use.',
        },
    )

    output_max_length: int | None = field(
        default=1024,
        metadata={
            'help': 'The maximum length of the output sequence for generation.',
        },
    )

    output_min_length: int | None = field(
        default=32,
        metadata={
            'help': 'The minimum length of the output sequence for generation.',
        },
    )

    mini_batch_size: int | None = field(
        default=1,
        metadata={
            'help': 'The mini-batch size to use for PPO.',
        },
    )

    batch_size: int | None = field(
        default=32,
        metadata={
            'help': 'The batch size to use for PPO.',
        },
    )

    ppo_epochs: int | None = field(
        default=4,
        metadata={
            'help': 'The number of PPO epochs.',
        },
    )

    gradient_accumulation_steps: int | None = field(
        default=4,
        metadata={
            'help': 'The number of gradient accumulation steps.',
        },
    )

    adafactor: bool | None = field(
        default=False,
        metadata={
            'help': 'Whether to use AdaFactor instead of AdamW.',
        },
    )

    early_stopping: bool | None = field(
        default=False,
        metadata={
            'help': 'Whether to use early stopping.',
        },
    )

    target_kl: float | None = field(
        default=0.1,
        metadata={
            'help': 'The target KL divergence for early stopping.',
        },
    )

    reward_baseline: float | None = field(
        default=0.0,
        metadata={
            'help': 'The reward baseline is a value subtracted from the reward.',
        },
    )

    batched_gen: bool | None = field(
        default=False,
        metadata={
            'help': 'Whether to use batched generation.',
        },
    )

    save_freq: int | None = field(
        default=None,
        metadata={
            'help': 'The frequency with which to save the model.',
        },
    )

    output_dir: str | None = field(
        default='runs/',
        metadata={
            'help': 'The output directory.',
        },
    )

    seed: int | None = field(
        default=42,
        metadata={
            'help': 'The seed.',
        },
    )

    steps: int | None = field(
        default=20_000,
        metadata={
            'help': 'The number of training steps.',
        },
    )

    init_kl_coef: float | None = field(
        default=0.2,
        metadata={
            'help': 'The initial KL coefficient (used for adaptive & linear control.',
        },
    )

    adap_kl_ctrl: bool | None = field(
        default=True,
        metadata={
            'help': 'Whether to adaptively control the KL coefficient, linear otherwise.',
        },
    )


def parse_args() -> ScriptArgs:
    """Parse the command line arguments.

    Returns:
        ScriptArgs: The parsed command line arguments.

    """
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]
    return args
