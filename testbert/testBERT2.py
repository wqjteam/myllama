import argparse
from functools import partial

import deepspeed
import torch
import torch.utils.data as Data
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding




parser = argparse.ArgumentParser(add_help=True,description='lijing')
parser.add_argument('--model_name_or_path', default="bert-base-uncased",type=str, help='lujing')
parser.add_argument('--batch_size',default=4, type=int, help='lujing')
parser.add_argument('--num_epochs',default=10, type=int, help='lujing')
parser.add_argument('--save_interval',default=100, type=int, help='lujing')
parser.add_argument('--save_dir',default="./save_model/", type=str, help='lujing')
args = parser.parse_args()


transformersmodel=AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
raw_datasets = load_dataset("glue", "mrpc")

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

model, optimizer, _, _ = deepspeed.initialize(model=transformersmodel,
                                                     model_parameters=transformersmodel.parameters())

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def collate_fn(data,data_collator):
    return data_collator(data)

collate_fn_partial = partial(collate_fn, data_collator=data_collator)

train_dataloader = Data.DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn_partial, batch_size=args.batch_size
)

dev_dataloader = Data.DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn_partial, batch_size=args.batch_size
)

#从checkpoint获取模型
# _, client_sd = model.load_checkpoint(args.load_dir, args.ckpt_id)
# step = client_sd['step']

for epoch in range(args.num_epochs):
    for step,batch in enumerate(train_dataloader):
        inputs, labels = batch
        loss = model(inputs, labels)
        model.backward(loss)
        optimizer.step()

        # save checkpoint
        if step % args.save_interval:
            # client_sd['step'] = step
            ckpt_id = loss.item()
            # model.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
            model.save_checkpoint(args.save_dir, ckpt_id)


# _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
# step = client_sd['step']


# # 初始化 TritonInferenceSession
ds_engine = deepspeed.init_inference(model,
                                 mp_size=2,
                                 dtype=torch.half,
                                 checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                 replace_with_kernel_inject=True)
model = ds_engine.module
output = model('Input String')