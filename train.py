import os
import time

import torch
from rlhf_trl.args import parse_args
from rlhf_trl.data import collator
from rlhf_trl.data import get_tokenizer
from rlhf_trl.data import load_data
from rlhf_trl.reward import reward_fn
from rlhf_trl.trainer import build_trainer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from trl.core import LengthSampler


def main() -> None:
    """Train the model."""

    # Parse arguments.
    args = parse_args()

    # Tokenizer & dataset.
    tokenizer = get_tokenizer(args.tokenizer_name)
    dataset = load_data(args.dataset_path, tokenizer, split='train')

    # PPO Trainer.
    config, ppo_trainer = build_trainer(
        args=args,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

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

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else 'cpu',
        )

    # Reward model.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
    )
    reward_model = reward_model.to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name,
    )

    print(f'Using device: {device}')
    start_time = time.time()
    # Training loop.
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), desc='Training PPO'):
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

        # Compute reward score.
        scores = reward_fn(
            model=reward_model,
            tokenizer=reward_tokenizer,
            prompt_text=batch['query'],
            response_text=batch['response'],
            device=device,
        )
        rewards = [
            torch.tensor(score[0] - args.reward_baseline)
            for score in scores
        ]

        # Run the PPO step.
        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # Save the model.
        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(os.path.join(args.output_dir, f'ppo-{epoch}', 'model'))

    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    hours, mins = divmod(mins, 60)
    print(f'Training took {hours:.0f}h, {mins:.0f}m {secs:.0f}s.')

    print('\nSaving model!')
    ppo_trainer.save_pretrained(os.path.join(args.output_dir, 'model'))


if __name__ == '__main__':
    main()
