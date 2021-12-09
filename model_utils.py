import transformers


def load_pipeline(task, model, device):
    print(f"Loading the pipeline '{model}'...")
    return transformers.pipeline(task, model, device=device)


def clean_text(txt):
    """Remove unnecessary spaces."""
    return ' '.join(txt.strip().split())


def generate_responses(prompt, pipeline, seed=None, **kwargs):
    outputs = pipeline(prompt, **kwargs)
    responses = list(map(lambda x: clean_text(
        x['generated_text'][len(prompt):]), outputs))
    return responses
