from datasets import Dataset, load_dataset
import torch
import numpy as np
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import os
from tqdm import tqdm
import json
os.environ['CUDA_VISIBLE_DEVICES']='0'
from huggingface_hub import login
login(token='your token')
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model, PeftModel
)
import logging
import os
import pandas as pd

def print_vram_with_nvidia_smi():
    os.system("nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv")


def get_data(random_seed):
    """
    Load train, validation, and test datasets using the datasets library.
    """
    print("Loading datasets...")
    
    # Load dataset from the specified dataset name
    ds = load_dataset("Jinyan1/COLING_2025_MGT_en")


    train_df = ds['train']  
    dev_df = ds['dev']  

    tmp = dev_df.train_test_split(shuffle = True, seed = random_seed, test_size='your modify')
    train_df = tmp['train']
    val_df = tmp['test']
    return train_df, val_df, dev_df
class EnsembleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer1, tokenizer3, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer3 = tokenizer3
        self.tokenizer1 = tokenizer1
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text with model3's tokenizer
        inputs_model1 = self.tokenizer1(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        inputs_model3 = self.tokenizer3(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )


        inputs_model1 = {key: value.squeeze(0) for key, value in inputs_model1.items()}
        inputs_model3 = {key: value.squeeze(0) for key, value in inputs_model3.items()}

        return {
            'input_model1': inputs_model1,
            'input_model3': inputs_model3,  # Metrics as tensor
            'label': torch.tensor(label, dtype=torch.float)
        }

class EnsembleInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer1, tokenizer3, ids=None, max_length=256):
        self.texts = texts
        self.tokenizer1 = tokenizer1
        self.tokenizer3 = tokenizer3
        self.ids = ids  
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the text with each model's tokenizer
        inputs_model1 = self.tokenizer1(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        inputs_model3 = self.tokenizer3(
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )


        inputs_model1 = {key: value.squeeze(0) for key, value in inputs_model1.items()}
        inputs_model3 = {key: value.squeeze(0) for key, value in inputs_model3.items()}

        data = {
            'input_model1': inputs_model1,
            'input_model3': inputs_model3,
        }

        if self.ids is not None:
            data['id'] = self.ids[idx]  # Use 'id' as the key

        return data


# Custom Collate Function
def collate_fn(batch):
    input_model1 = {key: torch.stack([example['input_model1'][key] for example in batch]) for key in batch[0]['input_model1']}
    input_model3 = {key: torch.stack([example['input_model3'][key] for example in batch]) for key in batch[0]['input_model3']}
    
    result = {
        'input_model1': input_model1,
        'input_model3': input_model3,
    }

    if 'label' in batch[0]:
        labels = torch.tensor([example['label'] for example in batch], dtype=torch.float)
        result['label'] = labels

    if 'id' in batch[0]:
        ids = [example['id'] for example in batch]
        result['id'] = ids

    return result


# Ensemble Model Class
class EnsembleModel(nn.Module):
    def __init__(self, model1, model3):
        super().__init__()
        
        # Load models
        self.model1_model = model1
        self.model3_model = model3

        # Initialize learnable weights for each model's output
        for param in self.model1_model.parameters():
            param.requires_grad = False
        for param in self.model3_model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(4, 1) 



    def forward(self, input_model1, input_model3):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Move inputs to device
            input_model1 = {key: value.to(device) for key, value in input_model1.items()}
            input_model3 = {key: value.to(device) for key, value in input_model3.items()}

            logits_model1 = self.model1_model(**input_model1).logits
            logits_model3 = self.model3_model(**input_model3).logits
            combined_logits = self.linear(torch.cat((logits_model1, logits_model3), dim=1))

            return combined_logits.squeeze(1)

            

def evaluate_ensemble(ensemble_model, dataloader, criterion):
    ensemble_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble_model.to(device)
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation', leave=False):
            input_model1 = batch['input_model1']
            input_model3 = batch['input_model3']
            labels = batch['label'].to(device)

            outputs = ensemble_model(input_model1, input_model3)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy (for binary classification)
            predictions = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def train_ensemble(ensemble_model, train_dataloader, val_dataloader, optimizer, criterion, epochs=5, save_dir='./ensemble/mono/checkpoints'):
    ensemble_model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble_model.to(device)

    os.makedirs(save_dir, exist_ok=True)  

    best_val_accuracy = 0.0
    trainer_state = {
        "best_model_checkpoint": None,
        "epoch": 0,
        "best_val_accuracy": 0.0,
        "validation_loss": [],
        "validation_accuracy": [],
        "log_history": []
    }
    
    num_batches = len(train_dataloader)
    steps_per_epoch = num_batches // 10  
    global_step = 0

    for epoch in range(epochs):
        ensemble_model.train()
        total_loss = 0
        num_batches = len(train_dataloader)
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for step, batch in enumerate(progress_bar):
            input_model1 = batch['input_model1']
            input_model3 = batch['input_model3']
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = ensemble_model(input_model1, input_model3)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # Evaluation after each epoch
        val_loss, val_accuracy = evaluate_ensemble(ensemble_model, val_dataloader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save checkpoint if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth')
            torch.save(ensemble_model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

            trainer_state["best_model_checkpoint"] = checkpoint_path
            trainer_state["epoch"] = epoch + 1
            trainer_state["best_val_accuracy"] = best_val_accuracy
            trainer_state["validation_loss"].append(val_loss)
            trainer_state["validation_accuracy"].append(val_accuracy)


            trainer_state_path = os.path.join(save_dir, 'trainer_state.json')
            with open(trainer_state_path, 'w') as f:
                json.dump(trainer_state, f)
            print(f"Saved trainer state to {trainer_state_path}")

def predictions(test_dataloader, ensemble_model, prediction_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble_model.eval()
    ensemble_model.to(device)
    all_predictions = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Inference'):
            input_model1 = batch['input_model1']
            input_model3 = batch['input_model3']
            
            input_model1 = {k: v.to(device) for k, v in input_model1.items()}
            input_model3 = {k: v.to(device) for k, v in input_model3.items()}
            
            outputs = ensemble_model(input_model1, input_model3)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).long().cpu().numpy()
            all_predictions.extend(predictions)

            if 'id' in batch:
                all_ids.extend(batch['id'])

    if all_ids:
        prediction_results = [{'id': id_, 'label': int(label)} for id_, label in zip(all_ids, all_predictions)]
    else:
        prediction_results = [{'label': int(label)} for label in all_predictions]

    with open(prediction_path, 'w') as f:
        for item in prediction_results:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", "-", required=True, help="Choose action", type=str) # Here, "train" or "inference"
    args = parser.parse_args()
    action = args.action

    base_model3_path = "model base" #for example: meta-llama/Llama-3.2-1B
    model3 = AutoModelForSequenceClassification.from_pretrained(base_model3_path)
    adapter3_path = "adapter path after fine-tuning" #for example: f"./meta-llama/Llama-3.2-1B/best/"  
    model3 = PeftModel.from_pretrained(model3, adapter3_path)
    tokenizer3 = AutoTokenizer.from_pretrained(base_model3_path)
    tokenizer3.pad_token = tokenizer3.eos_token
    model3.config.pad_token_id = model3.config.eos_token_id 
    
    print_vram_with_nvidia_smi()


    base_model1_path = "model base"#"Qwen/Qwen2.5-1.5B"
    model1 = AutoModelForSequenceClassification.from_pretrained(base_model1_path)

    adapter1_path = "adapter path after fine-tuning" #for example: f"./meta-llama/Llama-3.2-1B/best/"  
    model1 = PeftModel.from_pretrained(model1, adapter1_path)

    tokenizer1 = AutoTokenizer.from_pretrained(base_model1_path)
    tokenizer1.pad_token = tokenizer1.eos_token
    model1.config.pad_token_id = model1.config.eos_token_id 
    ensemble_model = EnsembleModel(model1, model3)
    print_vram_with_nvidia_smi()


    if action == "train": 
        # Load data
        train_df, val_df, dev_df = get_data(0)

        # Extract texts, labels, and metrics
        train_texts = train_df['text']
        train_labels = train_df['label']
        val_texts = val_df['text']
        val_labels = val_df['label']

        train_dataset = EnsembleDataset(train_texts, train_labels, tokenizer1, tokenizer3)
        val_dataset = EnsembleDataset(val_texts, val_labels, tokenizer1, tokenizer3)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        print_vram_with_nvidia_smi()



        criterion = nn.BCEWithLogitsLoss()  
        optimizer = AdamW(ensemble_model.linear.parameters(), lr=1e-5)

        train_ensemble(ensemble_model, train_dataloader, val_dataloader, optimizer, criterion, epochs=1)
    
    elif action == "inference":

        dev_test_df = pd.read_json("./en_dev_test_textonly.jsonl", lines=True)
        
        test_texts = dev_test_df['text'].tolist()

        test_ids = dev_test_df['id'].tolist() if 'id' in dev_test_df.columns else None

        test_dataset = EnsembleInferenceDataset(test_texts, tokenizer1, tokenizer3, ids=test_ids)

        # Create DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble_model.load_state_dict(torch.load('./checkpoints/best_checkpoint.pth', map_location=device, weights_only=True))

        # Run predictions
        prediction_path = './en_devtest_2llm_predictions.jsonl'
        predictions(test_dataloader, ensemble_model, prediction_path)

    else:
        logging.error("Action is not specified")
        raise ValueError("Action is not specified")
