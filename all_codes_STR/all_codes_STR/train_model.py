from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import os
import torch
from transformers import get_scheduler
from datasets import Dataset
from datasets import load_from_disk
import sys
import tqdm
tqdmn=tqdm.tqdm
import mlflow
from scipy.stats import spearmanr, pearsonr
from transformers import XLMRobertaForSequenceClassification
from transformers import RobertaForSequenceClassification
from tqdm import tqdm
import json
from dataloader import DataProcessor
import shutil
device=torch.device("cuda") if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, languages, train_dataloader, dev_dataloader, test_dataloader, device):
        self.languages = languages
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.best_model_paths = {}  # To store paths of the best models for each language

    def train(self, n_epochs, model_checkpoint_path, p_directory, valid_labels):
        valid_predictions={}
        best_valid_accuracy={}
        best_model_epoch={}
        best_model_state_dict={}
        correlation_coefficient={}
        valid_accuracy_list={}
        for lang in self.languages:
            best_valid_accuracy[lang] = -float('inf')
            valid_predictions[lang]=[]
            best_model_epoch[lang] = None
            best_model_state_dict[lang] = None


            with mlflow.start_run(run_name=f"1_stage_fine_tuning_Xlm_roberta_large_monolanguage {lang}") as run:
                torch.manual_seed(7)
                model = XLMRobertaForSequenceClassification.from_pretrained(model_checkpoint_path, num_labels=1)
                model.train().to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
                num_training_steps = (n_epochs * len(self.train_dataloader)) / 16
                lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
                
                train_loss = []
                valid_loss = []
                valid_accuracy_list[lang] = []

                for epoch in tqdm(range(n_epochs)):
                    model.train()
                    for i, batch in enumerate(self.train_dataloader):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        train_loss.append(loss.item())

                    model.eval()
                    for i, batch in enumerate(self.dev_dataloader):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with torch.no_grad():
                            outputs = model(**batch)
                            loss = outputs.loss
                            logits = outputs.logits
                            predictions = logits.cpu().detach().numpy().flatten().tolist()
                            valid_predictions[lang].extend(predictions)
                            valid_loss.append(loss.item())
                    print("predict6ed values:",valid_predictions)
                    print("True values:",valid_labels)
                    # valid_clean_predictions=[item[0] for item in valid_predictions]
                    # val_preds=[value.item() for value in valid_clean_predictions]
                    correlation_coefficient[lang] = pearsonr(valid_predictions[lang], valid_labels[lang])[0]
                    print(correlation_coefficient)
                    valid_accuracy_list[lang].append(correlation_coefficient[lang])

                    if correlation_coefficient[lang] > best_valid_accuracy[lang]:
                        best_valid_accuracy[lang] = correlation_coefficient[lang]
                        best_model_epoch[lang] = epoch + 1
                        best_model_state_dict[lang] = model.state_dict().copy()

                    print(f"Epoch {epoch + 1}: Train Loss: {sum(train_loss) / len(train_loss)}, Valid Accuracy: {correlation_coefficient}, Valid Loss: {sum(valid_loss) / len(valid_loss)}")

                    mlflow.log_metric("train_loss", sum(train_loss) / len(train_loss))
                    mlflow.log_metric("valid_loss", sum(valid_loss) / len(valid_loss))
                    mlflow.log_metric("valid_accuracy", correlation_coefficient[lang])

                    if epoch > 0 and valid_loss[-1] > valid_loss[-2]:
                        break

                # Save the best model's state dictionary
                self.save_best_model(lang, best_model_epoch[lang], model, best_model_state_dict[lang])

                # Delete remaining models
                self.delete_remaining_models(p_directory, lang, best_model_epoch[lang])

                # Log final accuracies for the last epoch
                mlflow.log_metric("final_train_loss", train_loss[-1])
                mlflow.log_metric("final_valid_loss", valid_loss[-1])
                mlflow.log_metric("final_valid_accuracy", valid_accuracy_list[lang][-1])

                # Save valid accuracy for each epoch and language to a JSON file
                with open(f"valid_accuracy_{lang}.json", "w") as f:
                    json.dump(valid_accuracy_list[lang], f)

    def save_best_model(self, lang, epoch, model, state_dict):
        model_directory = os.path.join(p_directory, lang, f"epoch_{epoch}")
        model.save_pretrained(model_directory)
        # os.makedirs(model_directory, exist_ok=True)
        # model_path = os.path.join(model_directory, "model.pt")
        # torch.save(state_dict, model_path)
        # self.best_model_paths[lang] = model_path
        self.best_model_paths[lang]=model_directory

    def delete_remaining_models(self, p_directory, lang, best_epoch):
        for epoch_folder in os.listdir(os.path.join(p_directory, lang)):
            if epoch_folder.startswith("epoch_") and epoch_folder != f"epoch_{best_epoch}":
                path_to_delete = os.path.join(p_directory, lang, epoch_folder)
                if os.path.exists(path_to_delete):
                    shutil.rmtree(path_to_delete)

if __name__ == "__main__":
    # Define your data loaders and other parameters
    languages = ["eng", "amh", "arq", ]
    n_epochs = 1 
    # dataset_root = "/home/naive123/nlp/Sumit/Textural_Relateness/Semantic_Relatedness_SemEval2024/textuaL_relatedness_test_phase/Semantic_Relatedness_SemEval2024/Dataset/Track A"  # Specify the root path to your dataset directory
    dataset_root="C:/Users/panka/Desktop/semantic_textual_relatedness/Semantic_Relatedness_SemEval2024/Track_A"
    max_length = 92  # Your max_length
    batch_size = 8   # Your batch_size
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_checkpoint_path = "xlm-roberta-large"
    p_directory = "saved_models_train_py_format"
    train_dataloader={}
    dev_dataloader={}
    test_dataloader={}
    val_labels={}
    data_val_path={}
    valid_dataset={} 
    label2id = {}  # dictionary to map labels to ids
    id2label = {}  # dictionary to map ids to labels

    for lang in languages:
        # Your data loading code goes here
        # Make sure you have loaded train, dev, and test data for each language
        # Assuming train_data, dev_data, and test_data are populated here
        data_processor = DataProcessor(dataset_root, languages, max_length, batch_size)
        data_val_path[lang] = os.path.join(dataset_root, f"{lang}/{lang}_dev_with_labels.csv")
        valid_dataset[lang] = data_processor._process_dataset(data_val_path[lang], nrows=10)  # Taking only 10 rows
        val_labels[lang] = valid_dataset[lang]["labels"]
        print(val_labels)
        train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang] = data_processor.process_data(lang)
        print(f"Training the model on {lang} language")
        # Define label-to-id and id-to-label mappings
        label2id[lang] = {label: idx for idx, label in enumerate(set(val_labels[lang]))}
        id2label[lang] = {idx: label for label, idx in label2id[lang].items()}

        trainer = Trainer(languages, train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang], device)
        
        # Start the training process
    trainer.train(n_epochs, model_checkpoint_path, p_directory, val_labels)

    # Save the dictionary containing the paths of the best models into a JSON file
    with open("best_model_paths.json", "w") as f:
        json.dump(trainer.best_model_paths, f)
        

# if __name__ == "__main__":
#     # Define your data loaders and other parameters
#     languages = ["eng", "esp", "amh", "arq", "ary", "hau", "kin", "mar", "tel"]
#     n_epochs = 1 
#     dataset_root = "/home/naive123/nlp/Sumit/Textural_Relateness/Semantic_Relatedness_SemEval2024/textuaL_relatedness_test_phase/Semantic_Relatedness_SemEval2024/Dataset/Track A"  # Specify the root path to your dataset directory
#     max_length = 92  # Your max_length
#     batch_size = 8   # Your batch_size
#     device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model_checkpoint_path = "xlm-roberta-large"
#     p_directory = "saved_models_train_py_format"
#     train_dataloader={}
#     dev_dataloader={}
#     test_dataloader={}
#     val_labels={}
#     data_val_path={}
#     valid_dataset={} 
#     label2id = {}  # dictionary to map labels to ids
#     id2label = {}  # dictionary to map ids to labels
#     for lang in languages:
#         # Your data loading code goes here
#         # Make sure you have loaded train, dev, and test data for each language
#         # Assuming train_data, dev_data, and test_data are populated here
#         data_processor = DataProcessor(dataset_root, languages, max_length, batch_size)
#         data_val_path[lang]=os.path.join(dataset_root, f"{lang}/{lang}_dev.csv")
#         valid_dataset[lang]=data_processor._process_dataset(data_val_path[lang])
#         val_labels[lang]= valid_dataset[lang]["labels"]
#         train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang] = data_processor.process_data(lang)
#         print(f"Training the model on {lang} language")
#         # Define label-to-id and id-to-label mappings
#         label2id[lang] = {label: idx for idx, label in enumerate(set(val_labels[lang]))}
#         id2label[lang] = {idx: label for label, idx in label2id[lang].items()}

#         trainer = Trainer(languages, train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang], device)
        
#         # Start the training process
#         trainer.train(n_epochs, model_checkpoint_path, p_directory, val_labels[lang])

#     # Save the dictionary containing the paths of the best models into a JSON file
#     with open("best_model_paths.json", "w") as f:
#         json.dump(trainer.best_model_paths, f)


# class Trainer:
#     def __init__(self, languages, train_dataloader, dev_dataloader, test_dataloader, device):
#         self.languages = languages
#         self.train_dataloader = train_dataloader
#         self.dev_dataloader = dev_dataloader
#         self.test_dataloader = test_dataloader
#         self.device = device
#         self.best_model_paths={}

#     def train(self, n_epochs, model_checkpoint_path, p_directory):
#         # for lang in self.languages:
#         with mlflow.start_run(run_name=f"1_stage_fine_tuning_Xlm_roberta_large_monolanguage {lang}") as run:
#             torch.manual_seed(7)
#             model = XLMRobertaForSequenceClassification.from_pretrained(model_checkpoint_path, num_labels=1)
#             model.train().to(self.device)
#             optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#             num_training_steps = (n_epochs * len(self.train_dataloader)) / 16
#             lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
            
#             train_loss = []
#             valid_loss = []
#             valid_accuracy_list = []
            
#             for epoch in tqdm(range(n_epochs)):
#                 # Training loop
#                 model.train()
#                 for i, batch in tqdm(enumerate(self.train_dataloader)):
#                     # print(batch)
#                     batch = { k: v.to(self.device) for k, v in batch.items()}
#                     outputs = model(**batch)
#                     loss = outputs.loss
#                     loss.backward()
#                     optimizer.step()
#                     lr_scheduler.step()
#                     optimizer.zero_grad()
#                     train_loss.append(loss.item())

#                 # Validation loop
#                 model.eval()
#                 valid_predictions = []
#                 for i, batch in tqdm(enumerate(self.dev_dataloader)):
#                     batch = { k: v.to(self.device) for k, v in batch.items()}
#                     with torch.no_grad():
#                         outputs = model(**batch)
#                         loss = outputs.loss
#                         logits = outputs.logits
#                         predictions = logits.cpu().detach().numpy().flatten().tolist()
#                         valid_predictions.extend(predictions)
#                         valid_loss.append(loss.item())
                
#                 # Compute validation accuracy

#                 correlation_coefficient = pearsonr(valid_predictions, val_labels[lang])[0]
#                 valid_accuracy_list.append(correlation_coefficient)
                
#                 # Log metrics to MLflow
#                 mlflow.log_metric("train_loss", sum(train_loss) / len(train_loss))
#                 mlflow.log_metric("valid_loss", sum(valid_loss) / len(valid_loss))
#                 mlflow.log_metric("valid_accuracy", correlation_coefficient)
                
#                 # Save model checkpoints
#                 model_directory = os.path.join(p_directory, lang, f"epoch_{epoch}")
#                 os.makedirs(model_directory, exist_ok=True)
#                 model.save_pretrained(model_directory)

#                 # Early stopping condition
#                 if epoch > 0 and valid_loss[-1] > valid_loss[-2]:
#                     break

#             # Log final accuracies for the last epoch
#             mlflow.log_metric("final_train_loss", train_loss[-1])
#             mlflow.log_metric("final_valid_loss", valid_loss[-1])
#             mlflow.log_metric("final_valid_accuracy", valid_accuracy_list[-1])

# if __name__ == "__main__":
#     # Define your data loaders and other parameters
#     languages = ["eng", "esp", "amh", "arq", "ary", "hau", "kin", "mar", "tel"]
#     n_epochs = 1
#     dataset_root = "/home/naive123/nlp/Sumit/Textural_Relateness/Semantic_Relatedness_SemEval2024/textuaL_relatedness_test_phase/Semantic_Relatedness_SemEval2024/Dataset/Track A"  # Specify the root path to your dataset directory
#     languages = ["eng", "esp", "amh", "arq", "ary", "hau", "kin", "mar", "tel"]
#     max_length = 92  # Your max_length
#     batch_size = 8   # Your batch_size

#     model_checkpoint_path = "xlm-roberta-large"
#     p_directory = "saved_models_train_py_format"
#     train_dataloader={}
#     dev_dataloader={}
#     test_dataloader={}
#     val_labels={}
#     data_val_path={}
#     valid_dataset={} 
#     for lang in languages:
#     # Your data loading code goes here
#     # Make sure you have loaded train, dev, and test data for each language
#     # Assuming train_data, dev_data, and test_data are populated here
#         data_processor = DataProcessor(dataset_root, languages, max_length, batch_size)
#         data_val_path[lang]=os.path.join(dataset_root, f"{lang}/{lang}_dev.csv")
#         valid_dataset[lang]=data_processor._process_dataset(data_val_path[lang])
#         # print(valid_dataset[lang])
#         val_labels[lang]= valid_dataset[lang]["labels"]
#         train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang] = data_processor.process_data(lang)

#         # Initialize and configure the Trainer
#         trainer = Trainer(languages, train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang], device)
        
#         # Start the training process
#         trainer.train(n_epochs, model_checkpoint_path, p_directory)

