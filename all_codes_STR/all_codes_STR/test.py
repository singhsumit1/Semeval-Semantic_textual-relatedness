import torch
from transformers import RobertaForSequenceClassification, XLMRobertaForSequenceClassification
from tqdm import tqdm
from dataloader import DataProcessor
import os 
import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import torch
from tqdm import tqdm
from scipy.stats import pearsonr

class Predictor:
    def __init__(self, languages, best_models, test_dataloader, device):
        self.languages = languages
        self.best_models = best_models
        self.test_dataloader = test_dataloader
        self.device = device

    def prediction(self, valid_labels):
        correlation_coefficient = {}
        val_predictions={}
        for lang in self.languages:
            val_predictions[lang] = []
            num_labels = 1  # Define num_labels correctly
            print("path of the saved models")
            print(self.best_models)  
            # Load the model
            # model = torch.load(self.best_models[lang])
            # # model = model.to(self.device)
            # model.eval()
            # model = torch.load(self.best_models[lang])
            model= XLMRobertaForSequenceClassification.from_pretrained(self.best_models[lang])
        # ` # Here, you might need to create an instance of your model class
        #     # based on the architecture you used during training
        #     # For example, if you used a BERT model:
        #     # model = YourBERTModelClass()
            
        #     # Update the model's state dictionary
        #     model.load_state_dict(state_dict)
            device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model.to(device)
            # Set the model to evaluation mode
            model.eval()
  
            for batch in tqdm(self.test_dataloader):
                with torch.no_grad():
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    predictions = logits.squeeze().tolist()  # Assuming single-label classification
                    val_predictions[lang].extend(predictions)

            correlation_coefficient[lang] = pearsonr(val_predictions[lang], valid_labels[lang])[0]
            print(f"final accuracy for the {lang} is: {correlation_coefficient[lang]}")

        return correlation_coefficient

# class Predictor:
#     def __init__(self, languages, test_dataloader, device, best_models):
#         self.languages = languages
#         self.test_dataloader = test_dataloader
#         self.device = device
#         self.best_models = best_models

#     def prediction(self, valid_labels):
#         model={}
#         correlation_coefficient={}
#         for lang in self.languages:
#             val_predictions={}
#             num_labels = 1  # You need to define num_labels
#             # Load best model path from the best_models dictionary
#             # best_model_path = self.best_models.get(lang)
#             # if best_model_path is None:
#             #     print(f"No bdef prediction(self, valid_labels):
#         model={}
#         correlation_coefficient={}
#         for lang in self.languages:
#             val_predictions={}
#             num_labels = 1  # You need to define num_labels
#             # Load best model path from the best_models dictionary
#             # best_model_path = self.best_models.get(lang)
#             # if best_model_path is None:
#             #     print(f"No best model found for {lang}")
#             #     continue
#             print(self.best_models)
#             # model[lang] = XLMRobertaForSequenceClassification.from_pretrained(self.best_models[lang], num_labels=num_labels)  
#             model[lang]= torch.load(self.best_models[lang])
#             # model[lang] = model[lang].eval().to(self.device)
            
#             predicted = []
#             for i, batch in enumerate(tqdm(self.test_dataloader)):
#                 with torch.no_grad():
#                     batch = {k: v.to(self.device) for k, v in batch.items()}
#                     print(model[lang])
#                     input_ids=batch["input_ids"]
#                     attention_mask=batch["attention_mask"]
#                     # outputs = model[lang](**batch)
#                     outputs = model[lang](input_ids=input_ids, attention_mask=attention_mask)
#                     logits = outputs.logits
#                     predictions = logits.tolist()
#                     val_predictions[lang].extend(predictions)

#                 # flattened_pred_values = [item for sublist in pred_values for item in sublist]
#                 # predicted += flattened_pred_values
#             print(val_predictions)
#             correlation_coefficient[lang] = pearsonr(val_predictions[lang], valid_labels[lang])[0]
#             print(correlation_coefficient)

            # # Add predicted scores to the test dataset
            # test_pdf[lang]["Pred_Score"] = predicted
            # csv_file_path = f"pred_{lang}_a.csv"
            # final_results_path = os.path.join(results_path, csv_file_path) 
            # test_pdf[lang][['PairID','Pred_Score']].to_csv(final_results_path, index=False)
            # print(f"Prediction completed for {lang}")



if __name__=="__main__":
    # dataset_root = "/home/naive123/nlp/Sumit/Textural_Relateness/Semantic_Relatedness_SemEval2024/textuaL_relatedness_test_phase/Semantic_Relatedness_SemEval2024/Dataset/Track A"  # Specify the root path to your dataset directory
    dataset_root="C:/Users/panka/Desktop/semantic_textual_relatedness/Semantic_Relatedness_SemEval2024/Track_A"
    # languages = ["eng",  "amh", "arq", "ary", "hau", "kin", "mar", "tel"]
    languages = ["eng",  "amh", "arq",]

    max_length = 92  # Your max_length
    batch_size = 8   # Your batch_size

    model_checkpoint_path = "xlm-roberta-large"
    p_directory = "saved_models_train_py_format"
    test_dataloader = {}
    data_val_path = {}
    val_labels = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    valid_dataset = {}
    # val_predictions={}
    label2id = {}  # dictionary to map labels to ids
    id2label = {}
    best_models = {}

    # Load the best models from a JSON file
    with open("best_model_paths.json", "r") as f:
        best_models = json.load(f)
    print(best_models)
    # Initialize DataProcessor
    data_processor = DataProcessor(dataset_root, languages, max_length, batch_size)

    for lang in languages:
        data_val_path[lang] = os.path.join(dataset_root, f"{lang}/{lang}_test_with_labels.csv",)
        valid_dataset[lang] = data_processor._process_dataset(data_val_path[lang], nrows=10)
        val_labels[lang] = valid_dataset[lang]["labels"]
        label2id[lang] = {label: idx for idx, label in enumerate(set(val_labels[lang]))}
        id2label[lang] = {idx: label for label, idx in label2id[lang].items()}
        _, _, test_dataloader[lang] = data_processor.process_data(lang)
        # Assuming you have test_pdf defined somewhere
        # test_pdf = {}  # Define your test data here
        # print(valid_dataset)
    predictor = Predictor(languages, best_models, test_dataloader[lang], device)
    # to predict the accuracy
    predictor.prediction(val_labels)
        # predictor.prediction("Results/2_stage_results_XLM_roberta/csv_files/", "Results/2_stage_results_XLM_roberta/zip_files/")

# import torch
# from transformers import RobertaForSequenceClassification
# from tqdm import tqdm
# from dataloader import DataProcessor
# import os 
# class Predictor:
#     def __init__(self, languages, test_dataloader, device, best_models):
#         # Initialize any variables or parameters here
#         self.languages=languages
#         self.test_dataloader=test_dataloader
#         self.device=device
#         self.best_models=best_models
#         # pass

#     def prediction(self, best_model_path, test_dataloader, test_pdf, results_path, zip_file_path):
#         num_labels =1  # You need to define num_labels
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Assuming you have defined num_labels and test_lan somewhere
        
#         # Your code goes here
#         confusion = torch.zeros(num_labels, num_labels)
#         predicted_for_classification_report = []
#         final_results_path = {}
#         zip_files_path = {}
#         # print(f"path of the {lang} model is: {best_model_path}")
#         model = RobertaForSequenceClassification.from_pretrained(best_models[lang])  
#         model = model.eval().to(device)
#         print("model device")
        
#         predicted = []
#         predictions = {}
#         true_labels = []
#         predicted_for_classification_report = []
        
#         for i, batch in enumerate(tqdm(test_dataloader[lang])):
#             with torch.no_grad():
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 outputs = model(**batch)
#                 logits = outputs.logits
#             pred_values = logits.tolist()
#             flattened_pred_values = [item for sublist in pred_values for item in sublist]
#             predicted += flattened_pred_values
        
#         print(len(predicted))   
#         print(test_pdf[lang])
#         test_pdf[lang]["Pred_Score"] = predicted
#         csv_file_path = f"pred_{lang}_a.csv"
#         final_results_path[lang] = os.path.join(results_path, csv_file_path) 
#         test_pdf[lang][['PairID','Pred_Score']].to_csv(final_results_path[lang], index=False)


# if __name__=="__main__":
#     dataset_root = "/home/naive123/nlp/Sumit/Textural_Relateness/Semantic_Relatedness_SemEval2024/textuaL_relatedness_test_phase/Semantic_Relatedness_SemEval2024/Dataset/Track A"  # Specify the root path to your dataset directory
#     languages = ["eng", "esp", "amh", "arq", "ary", "hau", "kin", "mar", "tel"]
#     max_length = 92  # Your max_length
#     batch_size = 8   # Your batch_size

#     model_checkpoint_path = "xlm-roberta-large"
#     p_directory = "saved_models_train_py_format"
#     test_dataloader={}
#     data_val_path={}
#     val_labels={}
#     device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     valid_dataset={}
#     label2id = {}  # dictionary to map labels to ids
#     id2label = {}
#     best_models={}
#     for lang in languages:
#         data_val_path[lang]=os.path.join(dataset_root, f"{lang}/{lang}_dev.csv")
#         valid_dataset[lang]=data_processor._process_dataset(data_val_path[lang])
#         # print(valid_dataset[lang])
#         val_labels[lang]= valid_dataset[lang]["labels"]
#         label2id[lang] = {label: idx for idx, label in enumerate(set(val_labels[lang]))}
#         id2label[lang] = {idx: label for label, idx in label2id[lang].items()}
#         data_processor = DataProcessor(dataset_root, languages, max_length, batch_size)
#         predictor = Predictor(languages, test_dataloader[lang], device,best_models[lang])
#         predictor.prediction(best_model_path, test_dataloader, test_pdf, "Results/2_stage_results_XLM_roberta/csv_files/", "Results/2_stage_results_XLM_roberta/zip_files/")
