import transformers


def load_pipeline(task, model, device):
    print(f"Loading the pipeline '{model}'...")
    return transformers.pipeline(task, model, device=device)


def strip_text(txt):
    return ' '.join(txt.strip().split())


def generate_responses(prompt, pipeline, **kwargs):
    outputs = pipeline(prompt, **kwargs)
    responses = list(map(lambda x: strip_text(
        x['generated_text'][len(prompt):]), outputs))
    return responses
