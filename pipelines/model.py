import os
import json
import torch
import argparse
import numpy as np

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E


class AVSR(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None,
        penalty=0., ctc_weight=0.1, lm_weight=0., beam_size=40, device="cuda:0"):
        super(AVSR, self).__init__()
        self.device = device

        if modality == "audiovisual":
            from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E
        else:
            from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        # --- robust token-list loading (handles multiple labels_type cases) ---
        labels_type = getattr(self.train_args, "labels_type", "char")
        #print(f"[AVSR] labels_type = {labels_type}")

        self.token_list = None

        # Helper to normalize char_list value into a Python list of tokens
        def _normalize_char_list(char_list):
            if char_list is None:
                return None
            if isinstance(char_list, list):
                return [str(x) for x in char_list if x is not None]
            if isinstance(char_list, str):
                # If it's a newline-separated string, split lines; otherwise split on whitespace
                lines = [ln.strip() for ln in char_list.splitlines() if ln.strip()]
                if lines:
                    return lines
                # fallback to whitespace split
                parts = [tok for tok in char_list.split() if tok]
                return parts if parts else None
            # fallback: try to coerce to list
            try:
                return [str(x) for x in char_list]
            except Exception:
                return None

        if labels_type == "char":
            # try to take char_list from train_args (common in espnet)
            char_list = getattr(self.train_args, "char_list", None)
            token_list = _normalize_char_list(char_list)
            if token_list:
                self.token_list = token_list
            else:
                raise ValueError(
                    "labels_type is 'char' but no usable 'char_list' found in train_args.\n"
                    f"train_args keys: {list(vars(self.train_args).keys())}"
                )

        elif labels_type == "unigram5000":
            file_path = os.path.join(os.path.dirname(__file__), "tokens", "unigram5000_units.txt")
            # Try UTF-8 first, fallback to latin-1; replace invalid bytes to avoid crashes.
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception:
                with open(file_path, "r", encoding="latin-1", errors="replace") as f:
                    lines = [line.strip() for line in f if line.strip()]

            self.token_list = ['<blank>'] + [w.split()[0] for w in lines] + ['<eos>']

        else:
            # Attempt graceful fallback: try to use train_args.char_list if available
            char_list = getattr(self.train_args, "char_list", None)
            token_list = _normalize_char_list(char_list)
            if token_list:
                #print(f"[AVSR] unknown labels_type '{labels_type}', using train_args.char_list as fallback")
                self.token_list = token_list
            else:
                # fail early with informative message
                raise ValueError(
                    f"Unsupported labels_type '{labels_type}' and no char_list available to fall back on.\n"
                    f"train_args keys: {list(vars(self.train_args).keys())}\n"
                    "Please check your model config (model_conf) or adjust labels_type to 'char' or 'unigram5000'."
                )

        # finally set odim (vocab dimension) and print for debug
        self.odim = len(self.token_list)
        #print(f"[AVSR] loaded {self.odim} tokens")
        # --- end robust token-list loading ---

        self.model = E2E(self.odim, self.train_args)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(device=self.device).eval()

        self.beam_search = get_beam_search_decoder(self.model, self.token_list, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size)
        self.beam_search.to(device=self.device).eval()
        
    def infer(self, data):
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.encode(data[0].to(self.device), data[1].to(self.device))
            else:
                enc_feats = self.model.encode(data.to(self.device))
            nbest_hyps = self.beam_search(enc_feats)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("▁", " ").strip()
        return transcription.replace("<eos>", "")


def get_beam_search_decoder(model, token_list, rnnlm=None, rnnlm_conf=None, penalty=0, ctc_weight=0.1, lm_weight=0., beam_size=40):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty,
    )

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
