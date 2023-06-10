import os
import jsonlines
import tqdm
from datasets import Dataset, Split
from transformers import AutoTokenizer


def load_data(
    path: str,
    tokenizer: AutoTokenizer,
    split: str = 'train',
    max_size: int | None = None,
    max_token: int = 1024,
) -> Dataset:
    """Load the OpenAssistant dataset.

    Args:
        path: Path to the dataset.
        split: Split to load.
        max_size: Maximum number of examples to load.
            Defaults to None.

    Returns:
        Dataset: The dataset.

    """
    assert split in ['train', 'test'], 'split must be either train or test.'

    path = os.path.join(path, f'tech-crunch-qa-{split}.jsonl')
    if os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist.')

    with jsonlines.open(path) as reader:
        data = [obj for obj in reader]

    prompts, input_ids = [], []
    qa_prompt: str = '<|propmpter|>{}<|endoftext|><|assistant|>'

    for obj in tqdm.tqdm(
        data,
        total=len(data) if max_size is None else max_size,
        desc='Loading data',
    ):
        prompt = qa_prompt.format(obj['prompt'])
        prompts.append(prompt)

        tokenized_prompt = tokenizer(prompt, truncation=True)
        input_ids.append(tokenized_prompt['input_ids'])

        if max_size is not None and len(prompts) >= max_size:
            break

    ds = Dataset.from_dict({
        'prompts': prompts,
        'input_ids': input_ids,
    }, split=Split.TRAIN if split == 'train' else Split.TEST)

    ds = ds.filter(lambda x: len(x['input_ids']) <= max_token, batched=False)
    ds.set_format(type='torch', columns=['input_ids'])

    return ds