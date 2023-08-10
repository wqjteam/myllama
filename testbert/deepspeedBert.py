import evaluate
import numpy as np
from deepspeed.ops.transformer import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
import argparse
from transformers import Trainer
from datasets import load_dataset

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--model_name_or_path', type=str, help='姓')
parser.add_argument('--from_tf', type=str, help='名')
model_args = parser.parse_args()
config = {}


# deepspeed --num_gpus=1 run_glue.py \
#   --model_name_or_path bert-base-cased \
#   --task_name mrpc \
#   --do_train \
#   --do_eval \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --num_train_epochs 3 \
#   --output_dir /tmp/mrpc/ \
#   --overwrite_output_dir \
#   --fp16 \
#   --deepspeed ds_config.json


def gen_ds_bert_config(training_args, config):
    bert_config = DeepSpeedTransformerConfig(
        batch_size=4096,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        heads=config.num_attention_heads,
        attn_dropout_ratio=config.attention_probs_dropout_prob,
        hidden_dropout_ratio=config.hidden_dropout_prob,
        num_hidden_layers=config.num_hidden_layers,
        initializer_range=0.02,
        layer_norm_eps=1e-8,
        local_rank=training_args.local_rank,
        fp16=training_args.fp16,
        pre_layer_norm=False,
        huggingface=True,
        training=True
    )
    return bert_config


# 模型替换
def inject_ds_enc_layer(model, training_args, config):
    for i in range(config.num_hidden_layers):
        bert_config = gen_ds_bert_config(training_args, config)
        model.bert.encoder.layer[i] = DeepSpeedTransformerLayer(bert_config)


# model = AutoModelForSequenceClassification.from_pretrained(
#     model_args.model_name_or_path,
#     from_tf=bool(".ckpt" in model_args.model_name_or_path),
#     config=config,
#     cache_dir=model_args.cache_dir,
#     revision=model_args.model_revision,
#     use_auth_token=True if model_args.use_auth_token else None,
# )
# 在model定义后立刻替换
# inject_ds_enc_layer(model, training_args, config)


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

raw_datasets = load_dataset("glue", "mrpc")


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments("test-trainer")


def compute_metrics(eval_preds):
    metric = evaluate.load("gule", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()