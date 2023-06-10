from dataclasses import dataclass, field


@dataclass
class ScriptArgs:
    """The name of the Causal LM wwe wish to fine-tune with PPO."""

    model_name: str = field(
        default='',
        metadata={
            'help': 'The name of the Causal LM wwe wish to fine-tune with PPO.',
        }
    )

    tokenizer_name: str = field(
        default='',
        metadata={
            'help': 'The name of the tokenizer to use.',
        }
    )

    reward_model_name: str = field(
        default='',
        metadata={
            'help': 'The name of the reward model to use.',
        }
    )

    log_with: str = field(
        default='wandb',
        metadata={
            'help': 'The logger to use.',
        }
    )

    learning_rate: float = field(
        default=1e-5,
        metadata={
            'help': 'The learning rate to use.',
        }
    )

    output_max_length: int = field(
        default=1024,
        metadata={
            'help': 'The maximum length of the output sequence for generation.',
        }
    )

    mini_batch_size: int = field(
        default=1,
        metadata={
            'help': 'The mini-batch size to use for PPO.',
        }
    )

    batch_size: int = field(
        default=32,
        metadata={
            'help': 'The batch size to use for PPO.',
        }
    )

    ppo_epochs: int = field(
        default=4,
        metadata={
            'help': 'The number of PPO epochs.',
        }
    )

    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            'help': 'The number of gradient accumulation steps.',
        }
    )
