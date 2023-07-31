import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pandas as pd
import glob
import os

from model import BertBaseUncased  # Import your Bert model here
from utils import load_data  # Import your load_data function here

def predict_data(file_path, model_path):
    file_list = glob.glob(os.path.join(file_path, "*.xlsx"))
    for file in file_list:
        label_predicted = []
        data = pd.read_excel(file, engine="openpyxl")
        text = data.loc[data["content"].notnull(), ["content", "pubdate"]]  # Filter out rows with NaN values
        text_pd = text.reindex(columns=['pubdate', 'content', 'label'], fill_value=0).reset_index(drop=True)
        text_p = text_pd.dropna(axis=0, how="any")
        text = np.array(text_pd.loc[:, ["content", "label"]]).tolist()
        test_dataset = Dataset(text)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, drop_last=False)
        model = BertBaseUncased()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        with torch.no_grad():
            testing_bitch = tqdm(test_loader)
            for test_data in testing_bitch:
                testing_text, testing_text_mask, testing_labels = test_data
                outputs = model(testing_text.to(device), testing_text_mask.to(device))
                predict_y = torch.max(outputs.logits, dim=1)[1]
                label_predicted.extend(predict_y.cpu().numpy().tolist())
            label_predicted = np.array(label_predicted)
            label_predicted[label_predicted == 2] = 1
            label_predicted[label_predicted == 3] = 1
            text_p["label"] = pd.DataFrame(label_predicted.tolist())
        # Save the file as .xlsx instead of .xls
        text_p.to_excel(file.replace(".xlsx", ".xlsx"), index=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text):
        self.text = text
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # Create tokenizer instance here

    def __getitem__(self, index):
        text, label = self.text[index]
        text_encoded_text = self.tokenizer.encode_plus(
            str(text),  # Convert to string to handle NaN values
            add_special_tokens=True,
            truncation=True,
            max_length=64,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        ids = text_encoded_text['input_ids'][0]
        mask = text_encoded_text['attention_mask'][0]
        label = int(label)

        return ids, mask, label

    def __len__(self):
        return len(self.text)


if __name__ == "__main__":
    path = r""
    model_path = r"bestModelbase.pth"

    predict_data(path, model_path)
