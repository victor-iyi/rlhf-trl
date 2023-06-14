import torch
from spacy.language import Language
from transformers import AutoModel
from transformers import AutoTokenizer


def reward_shaping_fn(
    nlp: Language,
    prompt_text: list[str],
    response_text: list[str],
) -> list[float]:
    """Compute the reward for a given response to a prompt.

    Args:
        nlp (Language): Spacy language model.
        prompt_text (list[str]): List of strings representing the prompt.
        response_text (list[str]): List of strings representing the response.

    Returns:
        list[float]: A list of floats representing the reward.

    """
    prompt_entities = []

    for prompt in prompt_text:
        prompt_entities.append(
            {item.text.lower() for item in nlp(prompt).ents},
        )

    model_entities = []

    for response in response_text:
        model_entities.append(
            {item.text.lower() for item in nlp(response).ents},
        )

    rewards = []

    for prompt, prompt_entity, model_entity in zip(
        prompt_text,
        prompt_entities,
        model_entities,
    ):
        score, flag = 0, 0

        if len(model_entity) == 0:
            score = 0
            flag = 1

        for entity in model_entity:
            if entity not in prompt_entity and entity not in prompt.lower():
                score += 1
                flag = 1
        if flag == 0:
            score = 1

        rewards.append(float(score))

    return rewards


def reward_fn(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompt_text: list[str],
    response_text: list[str],
    device: str,
) -> list[torch.FloatTensor]:
    """Compute the reward for a given response to a prompt.

    Args:
        model (AutoModel): Huggingface model.
        tokenizer (AutoTokenizer): Huggingface tokenizer.
        prompt_text (list[str]): List of strings representing the prompt.
        response_text (list[str]): List of strings representing the response.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.

    Returns:
        list[float]: A list of floats representing the reward.

    """
    with torch.no_grad():
        encoding = tokenizer(
            prompt_text,
            response_text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
        )
        encoding = encoding.to(device)

        logits = model(**encoding).logits
        scores = logits.cpu().numpy().flatten().tolist()

        return scores


if __name__ == '__main__':
    prompt_text = ['Victor lives in Nigeria.', 'Victor works at Google.']
    response_text = ['Victor lives in Nigeria.', 'Victor works at Google.']
    device = 'cpu'

    # Using reward model.
    # from transformers import AutoModelForSequenceClassification
    # model_name = 'OpenAssistant/reward-model-deberta-v3-base'
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # scores = reward_fn(model, tokenizer, prompt_text, response_text, device)
    # print(scores)

    # Using reward shaping.
    from core import get_ner
    nlp = get_ner(device=device, resolve_text=True, size='small')
    scores = reward_shaping_fn(nlp, prompt_text, response_text)
    print(scores)
