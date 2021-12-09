import transformers
import random
import tensorflow as tf
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# def load_pipeline(task, model, device=-1):
#     print(f"Loading the pipeline '{model}'...")
#     return transformers.pipeline(task, model, device)


def clean_text(txt):
    return ' '.join(txt.strip().split())


def generate_responses(prompt, pipeline, seed=None):
    if seed is not None:
        set_seed(seed)

    outputs = pipeline(prompt)
    # responses = list(map(lambda x: clean_text(
    #     x['generated_txt'][len(prompt):]), outputs))
    return outputs


# generation_pipeline = load_pipeline(
#     'text-generation', model="microsoft/DialoGPT-medium", device=-1)

# turn = {
#     'user_messages': [],
#     'bot_messages': []
# }
# turn['user_messages'].append('hey, how are you?')
# prompt = ""
# for user_message in turn['user_message']:
#     prompt += clean_text(user_message) + \
#         generation_pipeline.tokenizer.eos_token


# # print(bot_message)
pipeline = transformers.pipeline('text-generation')

bot_message = generate_responses(
    'hey, how are you?',
    pipeline,
    seed=None,
)

print(bot_message)
