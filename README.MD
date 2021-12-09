# Rybae

<img src="./assets/example.jpg" width="300" height="350">

<br>

Rybae is powered by [DialoGPT](https://github.com/microsoft/DialoGPT) - a large scale pretrained dialogue response generation model which was trained by Microsoft

Currently, this model runs on Tensorflow 2.0 and uses the `huggingface` transformer

**Note**: The bot is not always switched on to save compute cost

<br>

## Get started

1. Install project dependencies

```
pip install -r requirements.txt
```

2. Change the `TELGRAM_TOKEN` to your own.

```py
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
```

3. Run it from root folder

```py
python3 main.py
```

## References

- [Polakawo/gpt2bot on Pytorch](https://github.com/polakowo/gpt2bot)
- [Microsoft DialoGPT - Main](https://github.com/microsoft/DialoGPT)
- [Huggingface Transformers 🤗](https://huggingface.co/docs/transformers/index)
