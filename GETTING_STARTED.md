# MecAgent

## Install packages

```
uv venv

source .venv/bin/activate

uv sync
```

## Launch training

Open `mecagents/config.py` and check if the current configuration match what you expect

Current confiraguration is for LoRA finetuning of MLP layers only


Configuration can be made also in `train_modular.py`

```
nohup uv run train_modular.py
```
