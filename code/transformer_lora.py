import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from huggingface_hub import login
login(token='your token')

from datasets import Dataset, load_dataset
import pandas as pd
import evaluate
import numpy as np
np.random.seed(3407)
import time
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging


from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model, PeftModel
)

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], padding='max_length', max_length=256,truncation=True) #modify max_length tokenize


def get_data(dev_test_path, random_seed):
    """
    Load train, validation, and test datasets using the datasets library.
    """
    print("Loading datasets...")
    
    ds = load_dataset("Jinyan1/COLING_2025_MGT_multingual")

    train_df = ds['train'] 
    dev_df = ds['dev']    

    tmp = train_df.train_test_split(shuffle = True, seed = random_seed, test_size=0.1)
    train_df = tmp['train']
    val_df = tmp['test']
    dev_test_df = pd.read_json(dev_test_path, lines=True)
    print("Data loaded and split successfully.")
    
    return train_df, val_df, dev_df, dev_test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    train_dataset = train_df
    valid_dataset = valid_df

    tokenizer = AutoTokenizer.from_pretrained(model)   
    #quantization_config = BitsAndBytesConfig(
#     load_in_4bit = True, # enable 4-bit quantization
#     bnb_4bit_quant_type = 'nf4', 
#     bnb_4bit_use_double_quant = True, 
#     bnb_4bit_compute_dtype = torch.bfloat16 
# )


# lora config
    lora_config = LoraConfig(
        r = 16, 
        lora_alpha = 32, 
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.1, 
        bias = 'none', 
        task_type = 'SEQ_CLS'
    )

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        #quantization_config=quantization_config,
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id
    ).bfloat16()
    #model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id 
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)


def test(test_df, model, tokenizer, id2label, label2id):
    
    # load tokenizer from saved model 
    #tokenizer = AutoTokenizer.from_pretrained(model_path)

    # # load best model
    # model = AutoModelForSequenceClassification.from_pretrained(
    #    model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    # )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id        
    #test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_df
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds

def prediction(test_df,  model, tokenizer, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id     
            
    test_dataset = Dataset.from_pandas(test_df)
    #test_dataset = test_df
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)

    # return dictionary of classification report
    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_test_file_path", "-dt", required=True, help="Path to the dev_test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)
#    parser.add_argument("--prediction_dev_file_path", "-dp", required=True, help="Path where to save the prediction dev file.", type=str)

    args = parser.parse_args()

    random_seed = 0
    dev_test_path =  args.dev_test_file_path
    model =  args.model 
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path 

    

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))
    random_seed = 0
    set_seed(random_seed)
    print("Loading data .... \n")

    #get data for train/dev/test sets
    train_df, valid_df, test_df, dev_test_df = get_data(dev_test_path, random_seed)

    # train detector model
    start = time.time()
    fine_tune(train_df, valid_df, f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)
    end = time.time()
    print()
    base_model_path = "your baseline" #for example: meta-llama/Llama-3.2-1B
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    adapter_path = "adapter path after fine-tuning" #for example: f"./meta-llama/Llama-3.2-1B/best/"  
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    results, predictions = test(test_df, model, tokenizer, id2label, label2id)


    logging.info(results)
    # print("results on test set: ", results)
    # predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    # predictions_df.to_json(f"{prediction_path}/llama-16_subtask_{subtask}_dev.jsonl", lines=True, orient='records')


    # # ### infer for the dev test dataset
    print("start to infer on dev_test")
    prediction_dev_test = prediction(dev_test_df, model, tokenizer, id2label, label2id)
    prediction_dev_test_df = pd.DataFrame({'testset_id': dev_test_df['testset_id'], 'label': prediction_dev_test})
    prediction_dev_test_df.to_json(f"/home/nhi.doan/Downloads/NLP701/ai_detection/prediction_data/eval/llama_subtask_b_eval.jsonl", lines=True, orient='records')
    

#python3 ai_detection/code/transformer_lora.py --dev_test_file_path ./en_dev_test_textonly.jsonl --prediction_file_path ai_detection --subtask A --model Qwen/Qwen2.5-1.5B

