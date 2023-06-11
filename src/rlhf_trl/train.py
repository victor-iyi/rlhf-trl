from rlhf_trl.args import parse_args
from rlhf_trl.data import get_tokenizer
from rlhf_trl.data import load_data
from rlhf_trl.trainer import build_trainer
from tqdm import tqdm
from trl.core import LengthSampler


def train() -> None:
    """Train the model."""

    # Parse arguments.
    args = parse_args()

    # Tokenizer & dataset.
    tokenizer = get_tokenizer(args.tokenizer_name)
    dataset = load_data(args.dataset_path, tokenizer, split='train')

    # PPO Trainer.
    config, ppo_trainer = build_trainer(args, tokenizer, dataset)

    gen_kwargs = {
        'top_k': 0.0,
        'top_p': 0.9,
        'do_sample': True,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    output_length_sampler = LengthSampler(
        args.output_min_length,
        args.output_max_length,
    )

    # device = ppo_trainer.accelerator.device
    # if ppo_trainer.accelerator.num_processes == 1:
    #     device = (
    #         'cuda' if torch.cuda.is_available()
    #         else 'mps'
    #         if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    #         else 'cpu',
    #     )

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= config.total_ppo_epochs:
            break

        prompt_tensors = batch['input_ids']
        response_tensors = ppo_trainer.generate(
            prompt_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **gen_kwargs,
        )
        batch['response'] = tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True,
        )

        # TODO: Compute reward score.

        # TODO: Run the PPO step.

        # Save the model.
        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(f'{args.output_dir}-{epoch}')
