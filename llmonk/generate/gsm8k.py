import torch
from datasets import load_dataset
from tqdm import tqdm
import pydra
import multiprocessing
import random
import requests
from functools import partial

from llmonk.utils import save_yaml, GenerateScriptConfig
from llmonk.generate.vllm_utils import vllm_manager


def get_few_shot_prompt(item):
    few_shot_items = item["few_shot_items"]

    few_shot_pieces = []
    for f in few_shot_items:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/gsm8k/gsm8k.yaml
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)

    few_shot_prompt = "".join(few_shot_pieces)

    return few_shot_prompt


def run_inference(item, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    few_shot_prompt = (
        "Question: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. "
        "She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, "
        "$38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt "
        "for them. She has $16 left from her budget. How much did Alexis pay for the shoes?\n"
        "Answer:\n"
        "## Step 1: Let S be the amount Alexis paid for the shoes.\n"
        "## Step 2: She spent S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143.\n"
        "## Step 3: She has $16 left from her budget, so S + 143 = 200 - 16 = 184.\n"
        "## Step 4: Thus, Alexis paid S = 184 - 143 = $<<184-143=41>>41 for the shoes.\n"
        "#### 41\n\n"
        "Question: Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, "
        "and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. "
        "Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. "
        "What was the final weight of the box of goodies, in pounds?\n"
        "Answer:\n"
        "## Step 1: To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, "
        "bringing the weight to 2*3=<<2*3=6>>6 pounds.\n"
        "## Step 2: Next, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds.\n"
        "## Step 3: Finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds.\n"
        "#### 16\n\n"
    )
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:"

    url = f"http://localhost:{config.vllm_port}/generate"

    num_samples = config.num_samples
    batch_size = config.batch_size

    assert num_samples % batch_size == 0

    samples = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):

        body = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "n": batch_size,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stop": config.stop_strings,
            "logprobs": 1,
        }

        response = requests.post(url, json=body)
        respj = response.json()
        samples.extend(respj["text"])

    out = {
        "prompt": prompt,
        "question": item["question"],
        "samples": samples,
        "gt_answer": item["answer"],
    }

    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(
    config: GenerateScriptConfig,
):

    test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    test_dataset = test_dataset[:10]
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))
    train_dataset = train_dataset[:10]

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)

    for i, data in enumerate(train_dataset):
        data["id"] = i

    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        data["id"] = i
        data["few_shot_items"] = few_shot_items

    random.shuffle(test_dataset)

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(test_dataset)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    test_dataset = test_dataset[offset:limit:stride]

    print(f"Total number of items to process: {len(test_dataset)}")

    with vllm_manager(config) as vllm_port:
        config.vllm_port = vllm_port

        go_func = partial(run_inference, config=config)

        if config.num_workers not in [0, None]:
            with multiprocessing.Pool(config.num_workers) as pool:
                predictions = list(
                    tqdm(
                        pool.imap_unordered(go_func, test_dataset),
                        total=len(test_dataset),
                    )
                )
        else:
            predictions = []
            for item in tqdm(test_dataset):
                predictions.append(go_func(item))


if __name__ == "__main__":
    main()
