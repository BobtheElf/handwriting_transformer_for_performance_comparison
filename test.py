import torch
import pandas as pd

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

tr_df = pd.read_csv("./digit-recognizer/train.csv")
print(tr_df.head())
