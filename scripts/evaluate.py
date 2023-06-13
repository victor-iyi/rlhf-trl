import os

import jsonlines
import torch
from datasets import Dataset
from datasets import Split
from peft import PeftConfig
from peft import PeftModel
from rlhf_trl.args import parse_args
from rlhf_trl.args import ScriptArgs
from rlhf_trl.data import collator
from rlhf_trl.data import get_tokenizer
from rlhf_trl.reward import reward_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def load_data_v2(path: str, tokenizer: AutoTokenizer, split: str = 'test') -> Dataset:
    """Load the OpenAssistant dataset.

    Args:
        path (str): Path to the dataset.
        split (str): Split to load.

    Returns:
        Dataset: The dataset.

    """
    assert split in ['train', 'test', 'all'], 'split must be either train, test or all.'
    path = os.path.join(path, f'{split}.jsonl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist.')

    with jsonlines.open(path) as reader:
        data = [obj for obj in reader]

    query, input_ids, oa_ans, cgpt_ans = [], [], [], []

    qa_prompt: str = '<|prompter|>{}<|endoftext|><|assistant|>'

    for obj in tqdm(
        data,
        total=len(data),
        desc='Loading data',
    ):
        prompt = qa_prompt.format(obj['prompt'])
        input_id = tokenizer(prompt, padding='max_length', max_length=1024, truncation=True, return_tensors='pt').input_ids
        input_ids.append(input_id[0])
        query.append(qa_prompt.format(obj['prompt']))
        oa_ans.append(obj['openassistant-answer'])
        cgpt_ans.append(obj['chatgpt-answer'])

    split = 'train' if split == 'all' else split
    ds = Dataset.from_dict(
        {
            'query': query,
            'input_ids': input_ids,
            'openassistant-answer': oa_ans,
            'chatgpt-answer': cgpt_ans,
        },
        split=Split.TRAIN if split == 'train' else Split.TEST,
    )
    ds.set_format(type='torch')  # , columns=['query', 'input_ids', 'openassistant-answer', 'chatgpt-answer'])

    return ds


def evaluate(args: ScriptArgs) -> None:
    """Evaluate the model.

    Args:
        args (ScriptArgs): The script arguments.

    """
    tokenizer = get_tokenizer(args.tokenizer_name, padding_side='left')
    # 1. Load the test data.
    ds = load_data_v2(
        path=args.dataset_path,
        tokenizer=tokenizer,
        split='test',
        # return_answers=True,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collator)

    # Set the device.
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else 'cpu',
    )
    print(f'Using device: {device}')

    # 2. Load the (PPO & SFT) model.
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        # device_map='balanced',
        load_in_8bit=True,
    )
    peft_config = PeftConfig.from_pretrained(args.ppo_model_name)
    ppo_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True,
    )
    ppo_model = PeftModel.from_pretrained(ppo_model, args.ppo_model_name)

    # 3. Reward model.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
    )
    reward_model = reward_model.to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name,
    )

    gen_kwargs = {
        'top_k': 0.0,
        'top_p': 0.9,
        'max_new_tokens': 256,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    # data = {
    #     'prompt': [],
    #     'sft_output': [],
    #     'sft_scores': [],
    #     'ppo_output': [],
    #     'ppo_scores': [],
    #     'oa_ans': [],
    #     'oa_score': [],
    #     'chatgpt_ans': [],
    #     'chatgpt_score': [],
    # }
    data = []

    # 4. Make prediction with each model on the test data.
    for batch in tqdm(loader, desc='Evalutaing'):
        input_ids = torch.stack(batch['input_ids'], dim=0).to(device)

        # 4.1. Make prediction with sft_model.
        sft_encode = sft_model.generate(
            input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            **gen_kwargs,
        )
        sft_seq_len = len(sft_encode['scores'])
        sft_tokens = sft_encode['sequences'][:, -sft_seq_len:]

        sft_output = tokenizer.batch_decode(sft_tokens, skip_special_tokens=True)

        # 4.2. Make prediction with ppo_model.
        ppo_encode = ppo_model.generate(
            input_ids=input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            **gen_kwargs,
        )
        ppo_seq_len = len(ppo_encode['scores'])
        ppo_tokens = ppo_encode['sequences'][:, -ppo_seq_len:]
        ppo_output = tokenizer.batch_decode(ppo_tokens, skip_special_tokens=True)

        # 4.3. Calculate the reward score for each.
        sft_scores = reward_fn(
            model=reward_model,
            tokenizer=reward_tokenizer,
            prompt_text=batch['query'],
            response_text=sft_output,
            device=device,
        )

        ppo_scores = reward_fn(
            model=reward_model,
            tokenizer=reward_tokenizer,
            prompt_text=batch['query'],
            response_text=ppo_output,
            device=device,
        )

        # 4.4. Calculate the reward score for chatgpt-answers and openassistant-answers.
        oa_scores = reward_fn(
            model=reward_model,
            tokenizer=reward_tokenizer,
            prompt_text=batch['query'],
            response_text=batch['openassistant-answer'],
            device=device,
        )

        chatgpt_scores = reward_fn(
            model=reward_model,
            tokenizer=reward_tokenizer,
            prompt_text=batch['query'],
            response_text=batch['chatgpt-answer'],
            device=device,
        )

        # 4.5. Compile the results.
        for i in range(len(batch['query'])):
            data.append({
                'prompt': batch['query'][i],
                'sft_output': sft_output[i],
                'sft_scores': sft_scores[i],
                'ppo_output': ppo_output[i],
                'ppo_scores': ppo_scores[i],
                'oa_ans': batch['openassistant-answer'][i],
                'oa_score': oa_scores[i],
                'chatgpt_ans': batch['chatgpt-answer'][i],
                'chatgpt_score': chatgpt_scores[i],
            })

    # 5. Save the results.
    save_path = os.path.join(args.eval_save_path, f'{args.eval_name}.jsonl')
    os.makedirs(args.eval_save_path, exist_ok=True)

    print(f'Saving evaluation results to {save_path}...')
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(data)


def main() -> None:
    """Start the evaluation."""
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
