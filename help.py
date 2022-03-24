import wget
import pandas as pd
import random
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


def set_device():
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    
    return(device)
    
    
def load_clean_data(url='https://raw.githubusercontent.com/ChristinaROK/PreOnboarding_AI_assets/e56006adfac42f8a2975db0ebbe60eacbe1c6b11/data/sample_df.csv'):
    
    wget.download(url)
    sample_df = pd.read_csv('sample_df.csv').dropna()
    
    return sample_df


def label_evenly_balanced_dataset_sampler(df, sample_size):
    """
    데이터 프레임의을 sample_size만큼 임의 추출해 새로운 데이터 프레임을 생성.
    이 때, "label"열의 값들이 동일한 비율(5:5)을 가짐.
    """
    df_0_index = list(df[df['label']==0].index) # label이 0인 데이터의 index 추출
    df_1_index = list(df[df['label']==1].index) # label이 1인 데이터의 index 추출
    sample_index = random.sample(df_0_index, sample_size//2) +\
                   random.sample(df_1_index, sample_size//2) # sample_size의 절반만큼 random sampling 하여 sample data의 index list 구축
    sample = df.loc[sample_index].reset_index(drop=True) # index list를 사용해 sample data 가져오기

    return sample


def custom_collate_fn(batch):
    """
    - input_list: list of string
    - target_list: list of int
    """
    tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")
    input_list = [pair[0] for pair in batch]
    target_list = [pair[1] for pair in batch]
    
    tensorized_input = tokenizer_bert(input_list,
                                      add_special_tokens=True,
                                      padding='longest',
                                      truncation=True,
                                      return_tensors='pt')
    tensorized_label = torch.tensor(target_list)
    
    return tensorized_input, tensorized_label


class CustomDataset:
    """
    - input_data: list of string
    - target_data: list of int
    """
    def __init__(self, input_data:list, target_data:list) -> None:
        self.X = input_data
        self.Y = target_data

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    
class CustomClassifier(nn.Module):
    
    def __init__(self, hidden_size: int, n_label: int):
        super(CustomClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("klue/bert-base")

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        self.classifier = nn.Sequential(nn.Linear(hidden_size, linear_layer_hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(linear_layer_hidden_size, n_label))
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        
        cls_token_last_hidden_states = outputs['pooler_output'] # 마지막 layer의 첫 번째 토큰('[CLS]')의 벡터
        logits = self.classifier(cls_token_last_hidden_states)
        
        return logits