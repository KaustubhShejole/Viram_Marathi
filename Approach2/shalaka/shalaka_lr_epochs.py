import argparse
import os
import logging
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as eval_load
from huggingface_hub import login as hf_login
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def postprocess_indictrans_text(preds, labels, source_lang="eng_Latn", target_lang="mar_Deva"):
    prefix = f"{target_lang} {source_lang}"

    def remove_prefix(text: str) -> str:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
        return text.strip()

    preds = [remove_prefix(p) for p in preds]
    # return labels as list-of-lists for BLEU (sacrebleu expects that)
    labels = [[remove_prefix(l)] for l in labels]
    return preds, labels
	
def _safe_extract_comet_score(comet_out: dict) -> Optional[float]:
    # Try common keys used by various COMET wrappers
    for k in ("system_score", "mean_score", "score", "value"):
        if k in comet_out:
            return float(comet_out[k])
    # Some wrappers return {"scores": [...]} or similar
    if "scores" in comet_out:
        return float(np.mean(comet_out["scores"]))
    return None
	
def compute_metrics_indictrans(
    eval_preds,
    tokenizer,
    metric_bleu,
    metric_chrf=None,
    metric_comet=None,
    metric_bleurt=None,
    source_lang="eng_Latn",
    target_lang="mar_Deva",
):
    """
    Robust compute_metrics that accepts either:
      - eval_preds as (preds, labels) tuple (older usage), or
      - eval_preds as transformers.EvalPrediction object with attributes
        .predictions, .label_ids, and (optionally) .inputs when
        include_inputs_for_metrics=True is set in training args.
    """
    # Unpack in a robust way
    if hasattr(eval_preds, "predictions"):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        inputs = getattr(eval_preds, "inputs", None)
    else:
        # fallback if user passed a tuple (preds, labels)
        preds, labels = eval_preds
        inputs = None

    # If generation returned tuple (generated_ids, something) take first
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert tensors -> numpy arrays if necessary
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    preds = to_numpy(preds)
    labels = to_numpy(labels)
    inputs = to_numpy(inputs) if inputs is not None else None

    # Decode predictions (list[str])
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Decode labels: replace -100 -> pad_token_id then decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # If inputs are present (requires include_inputs_for_metrics=True), decode sources
    decoded_sources = None
    if inputs is not None:
        decoded_sources = tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Postprocess (remove prefixes). Note: postprocess returns references as list-of-lists.
    decoded_preds, decoded_labels = postprocess_indictrans_text(decoded_preds, decoded_labels, source_lang, target_lang)

    # ----- BLEU (sacrebleu expects references as list-of-lists) -----
    try:
        bleu_out = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_score = float(bleu_out.get("score", 0.0))
    except Exception as e:
        logger.warning("BLEU computation failed: %s", e)
        bleu_score = 0.0

    # ----- ChrF++ -----
    chrfpp_score = None
    if metric_chrf is not None:
        try:
            chrf_out = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
            chrfpp_score = float(chrf_out.get("score", np.nan))
        except Exception as e:
            logger.warning("ChrF computation failed: %s", e)

    # ----- COMET -----
    comet_score = None
    if metric_comet is not None:
        if decoded_sources is None:
            logger.warning("COMET requested but decoded_sources is None — set include_inputs_for_metrics=True in TrainingArguments")
        else:
            try:
                # COMET wants flat references (not list-of-lists)
                flat_refs = [r[0] if isinstance(r, (list, tuple)) and len(r) > 0 else r for r in decoded_labels]
                comet_out = metric_comet.compute(sources=decoded_sources, predictions=decoded_preds, references=flat_refs)
                comet_score = _safe_extract_comet_score(comet_out)
            except Exception as e:
                logger.warning("COMET computation failed: %s", e)

    # ----- BLEURT -----
    bleurt_score = None
    if metric_bleurt is not None:
        try:
            flat_refs = [r[0] if isinstance(r, (list, tuple)) and len(r) > 0 else r for r in decoded_labels]
            bleurt_out = metric_bleurt.compute(predictions=decoded_preds, references=flat_refs)
            # BLEURT returns "scores" usually
            if "scores" in bleurt_out:
                bleurt_score = float(np.mean(bleurt_out["scores"]))
            elif "score" in bleurt_out:
                bleurt_score = float(bleurt_out["score"])
        except Exception as e:
            logger.warning("BLEURT computation failed: %s", e)

    # ----- Generated length -----
    try:
        prediction_lens = [int(np.count_nonzero(p != tokenizer.pad_token_id)) for p in preds]
        gen_len = float(np.mean(prediction_lens)) if len(prediction_lens) > 0 else 0.0
    except Exception:
        gen_len = 0.0

    result = {
        "bleu": round(bleu_score, 4),
        "chrfpp": round(chrfpp_score, 4) if chrfpp_score is not None else None,
        "comet": round(comet_score, 4) if comet_score is not None else None,
        "bleurt": round(bleurt_score, 4) if bleurt_score is not None else None,
        "gen_len": round(gen_len, 4),
    }
    return result

def load_hf_token(path="hf_token.txt"):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"HF token file not found at {path}. Create it with your token.")

cuda = "3"
model_checkpoint = "ai4bharat/indictrans2-en-indic-dist-200M"
dataset = "thenlpresearcher/iitb_en_indic_marathi_punct_variants_tokenized"
batch_size = 16
eval_steps = 6000
save_steps = 6000
num_train_epochs = 5
learning_rate = 1e-5
save_total_limit = 3
metric_for_best_model = "chrfpp"
greater_is_better = True
source_lang = "eng_Latn"
target_lang = "mar_Deva"
push_to_hub = True
hub_model_id=None

os.environ["CUDA_VISIBLE_DEVICES"] = cuda
os.environ["WANDB_DISABLED"] = "true"

hf_token = load_hf_token()
hf_login(hf_token)

logger.info("Loading model/tokenizer from %s", model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, trust_remote_code=True)

logger.info("Loading dataset %s", dataset)
tokenized_datasets = load_dataset(dataset)

logger.info("Loading metrics")
metric_bleu = eval_load("sacrebleu")
metric_chrf = eval_load("chrf")
metric_comet = None
metric_bleurt = None
try:
    metric_comet = eval_load("comet")
except Exception as e:
    logger.warning("Could not load COMET metric: %s", e)
try:
    metric_bleurt = eval_load("bleurt")
except Exception as e:
    logger.warning("Could not load BLEURT metric: %s", e)
    
model_name = model_checkpoint.rstrip("/").split("/")[-1]
hub_model_id = f"thenlpresearcher/shalaka2_{model_name}_finetuned_{source_lang}_to_{target_lang}"


print(model_name)
print(hub_model_id)
print(push_to_hub)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"shalaka2_{model_name}-en-indic-iitb-finetuned-{source_lang}-to-{target_lang}",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=eval_steps,
    save_steps=save_steps,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=save_total_limit,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    fp16=False,
    logging_steps=100,
    seed=42,
    push_to_hub=push_to_hub,
    hub_model_id=hub_model_id,
    hub_private_repo=False,
    hub_strategy="end",
    load_best_model_at_end=True,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    include_inputs_for_metrics=True,   # IMPORTANT for COMET
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")      

trainer = Seq2SeqTrainer(
    model=model,                           # use local model variable
    args=training_args,                    # pass Seq2SeqTrainingArguments instance
    train_dataset=tokenized_datasets.get("train"),
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,                   # important so EvalPrediction.inputs is set
    data_collator=data_collator,
    compute_metrics=lambda eval_preds: compute_metrics_indictrans(
        eval_preds,
        tokenizer=tokenizer,
        metric_bleu=metric_bleu,
        metric_chrf=metric_chrf,
        metric_comet=metric_comet,
        metric_bleurt=metric_bleurt,
        source_lang=source_lang,
        target_lang=target_lang,
    ),
)  

logger.info("Starting fine-tuning ...")
trainer.train()

logger.info("Pushing final checkpoint to hub: %s", hub_model_id)
trainer.push_to_hub()

logger.info("Training finished")