from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from .base import ModelBase, gguf, logger
from .qwen import Qwen3_5TextModel
from .qwen3vl import Qwen3VLVisionModel


def _is_baselabs_checkpoint(dir_model: Path) -> bool:
    # qev/readout.py of OneJev (serve.patch of the model repo) writes readout.json and the "modeling_overrides" of config.json
    if not ((dir_model / "readout.json").is_file() and (dir_model / "config.json").is_file()):
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return "modeling_overrides" in json.load(f)


# Any: the decoded config.json, as the hparams loaders of ModelBase return it
@ModelBase.register_hparams_loader(_is_baselabs_checkpoint)
def _load_baselabs_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected Base Labs decision checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["BaseLabsDecisionModel"]
    return hparams


@ModelBase.register("BaseLabsDecisionModel")
@ModelBase.example("baselabs/elephant-1", "baselabs/mammoth-1")
class BaseLabsDecisionModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    no_mtp = True  # the checkpoint has no MTP head

    # prompt and labels follow qev/prompt.py ("qev-labels-v2") of OneJev @81ce62f + serve.patch, the runtime that setup.sh of the model repo pins
    _SYSTEM_PROMPT = (
        "Apply the question to the state. Choose exactly one of the listed options. "
        "Respond with only its uppercase letter, with no explanation or reasoning."
    )
    _LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                      "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
    _N_SLOTS = 256

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        overrides = self.hparams["modeling_overrides"]
        attention = overrides.get("attention", {})
        if attention.get("full_attention") != "block_bidirectional" or attention.get("linear_attention") != "causal":
            raise ValueError(f"unsupported attention {attention!r}, the baselabs-decision type expects block_bidirectional full attention and causal linear attention")
        # a "head" readout reads a learned head (readout_head.pt) instead of the LM head
        if overrides.get("readout") != "slot_lm":
            raise ValueError(f"unsupported readout {overrides.get('readout')!r}, the baselabs-decision type reads the LM head (slot_lm)")
        # the server uses a temperature of 1, as qev does without a calibration.json
        if (self.dir_model / "calibration.json").is_file():
            raise ValueError("calibration.json is not supported")

    def _slot_labels(self) -> list[str]:
        pairs = [a + b for a in self._LETTERS for b in self._LETTERS if a + b not in self._NOT_ONE_TOKEN]
        return list(self._LETTERS) + pairs[: self._N_SLOTS - len(self._LETTERS)]

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    def _systemone_template(self) -> str:
        # the server gives the template the texts rendered as qev renders them (render_state, render_entry), it only lays them out
        # an empty instructions text takes the default of the question type, a description is none if it is not given
        # the server splits the prompt at sep: the state block and the question block are the two blocks of the attention mask
        nl = "{{ '\\n' }}"
        name = "{% if type == 'noul' %}{{ 'yes' if o.key == 'true' else 'no' }}{% elif type == 'score' %}level {{ o.key }}{% else %}{{ o.key }}{% endif %}"
        text = (
            "{% if o.description is none and type == 'noul' %}"
            "{% set text = 'the statement is true / the answer is yes' if o.key == 'true' else 'the statement is false / the answer is no' %}"
            "{% elif o.description is none %}{% set text = '' %}{% else %}{% set text = o.description %}{% endif %}"
        )
        instructions = (
            "{% if instructions %}{{ instructions }}"
            "{% elif type == 'noul' %}Is the statement true, or is the answer to the question yes?"
            "{% elif type == 'choice' %}Which option applies to the state?"
            "{% else %}Which level describes the state?{% endif %}"
        )
        options = (
            "{% for o in options %}" + text + "{{ o.label }}. " + name + "{% if text %}: {{ text }}{% endif %}"
            "{% if not loop.last %}" + nl + "{% endif %}{% endfor %}"
        )
        answer = (
            "{% if options | length <= 26 %}Answer with one letter: {% else %}Answer with one label: {% endif %}"
            "{% for o in options %}{{ o.label }}{% if not loop.last %}, {% endif %}{% endfor %}."
        )
        return (
            "<|im_start|>system" + nl + self._SYSTEM_PROMPT + "<|im_end|>" + nl
            + "<|im_start|>user" + nl + "<state>" + nl + "{% for image in images %}{{ image }}{% endfor %}{{ state }}" + nl + "</state>" + nl + nl
            + "{{ sep }}{% if type == 'score' %}Rate the state:{% else %}Question:{% endif %} " + instructions + nl + nl
            + "Options:" + nl + options + nl + nl + answer + "<|im_end|>" + nl
            + "<|im_start|>assistant" + nl + "<think>" + nl + nl + "</think>" + nl + nl
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.BASELABS_DECISION)
        # the answer is read at the token of the label of each option (qev/engine.py: _readout), the server checks that each is one token
        self.gguf_writer.add_decision_labels(self._slot_labels())
        # the full attention layers are bidirectional within each block of the prompt, the server evaluates each block in one batch
        self.gguf_writer.add_causal_attention(False)


@ModelBase.register("BaseLabsDecisionModel")
class BaseLabsDecisionVisionModel(Qwen3VLVisionModel):
    pass
