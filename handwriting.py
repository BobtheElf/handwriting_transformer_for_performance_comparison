#libraries
import torch
from torch import nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm

#constants
SEED = 18
BATCH = 512
EPOCHS = 40
LEARN_RATE = 1e-4
CLASSES = 10
P_SIZE = 4
I_SIZE = 28
I_CHAN = 1
HEADS = 8
DROP = 0.001
HIDDEN_DIM = 768
ADAM_W_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION_FUNC = "gelu"
ENCODERS = 4
EMBED_DIM = (P_SIZE ** 2) * I_CHAN * 4 #16 (now 64)
PATCHES = (I_SIZE // P_SIZE) ** 2 #49

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("CUDA")
else:
    print("cpu")

#embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, p_size, patches, drop, i_chan):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels = i_chan,
                out_channels = embed_dim,
                kernel_size = p_size,
                stride = p_size
            ),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, patches + 1, embed_dim)), requires_grad=True)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # Flattened to (B, D, N), then permute to (B, N, D) for transformer
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.drop(x)
        return x
    
model = PatchEmbedding(EMBED_DIM, P_SIZE, PATCHES, DROP, I_CHAN).to(device)
x = torch.randn(512, 1, 28, 28).to(device)
output = model(x)
print(output.shape)

#transformer
class ViT(nn.Module):
    def __init__(self, patches, i_size, classes, p_size, embed_dim, encoders, heads, hidden_dim, drop, activation_func, i_chan):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, p_size, patches, drop, i_chan)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=heads,
                                                   dim_feedforward=256,  # already set to 768 in your constants
                                                   dropout=drop,
                                                   activation=activation_func,
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=classes)
        )
    
    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x
    
model = ViT(PATCHES, I_SIZE, CLASSES, P_SIZE, EMBED_DIM, ENCODERS, HEADS, HIDDEN_DIM, DROP, ACTIVATION_FUNC, I_CHAN).to(device)
x = torch.randn(512, 1, 28, 28).to(device)
output = model(x)
print(model(x).shape)

#data
tr_df = pd.read_csv("./digit-recognizer/train.csv")
ts_df = pd.read_csv("./digit-recognizer/test.csv")
s_df = pd.read_csv("./digit-recognizer/sample_submission.csv")

#testing data samples
tr_df.head()
ts_df.head()
s_df.head()

#splitting data
tr_df, val_df = train_test_split(tr_df, test_size=0.1, random_state=SEED, shuffle=True)

#training classes
class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}
    
class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}
    
class MNISTTestDataset(Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8)
        index = self.indices[idx]
        image = self.transform(image)

        return {"image": image, "index": index}

#examples of dataset before training
plt.figure()
f, r = plt.subplots(1,3)

tr_ds = MNISTTrainDataset(tr_df.iloc[:, 1:].values.astype(np.uint8), tr_df.iloc[:, 0].values, tr_df.index.values)
print(len(tr_ds))
print(tr_ds[0])
r[0].imshow(tr_ds[0]["image"].squeeze(), cmap="gray")
r[0].set_title("Train Image")
print("-"*30)

v_ds = MNISTValDataset(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values, val_df.index.values)
print(len(v_ds))
print(v_ds[0])
r[1].imshow(v_ds[0]["image"].squeeze(), cmap="gray")
r[1].set_title("Val Image")
print("-"*30)

ts_ds = MNISTTestDataset(ts_df.values.astype(np.uint8), ts_df.index.values)
print(len(ts_ds))
print(ts_ds[0])
r[2].imshow(ts_ds[0]["image"].squeeze(), cmap="gray")
r[2].set_title("Test Image")
print("-"*30)

plt.show()

#loading into the dataloader
tr_dl = DataLoader(dataset=tr_ds,
                              batch_size=BATCH,
                              shuffle=True)

v_dl = DataLoader(dataset=v_ds,
                              batch_size=BATCH,
                              shuffle=True)

ts_dl = DataLoader(dataset=ts_ds,
                              batch_size=BATCH,
                              shuffle=False)

#training the model
crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARN_RATE, weight_decay=ADAM_W_DECAY)

start = timeit.default_timer()
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    for idx, img_label in enumerate(tqdm(tr_dl, position=0, leave=True)):
        img = img_label["image"].float().to(device)
        label = img_label["label"].long().to(device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = crit(y_pred, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for idx, img_label in enumerate(tqdm(v_dl, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].long().to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = crit(y_pred, label)
            val_running_loss += loss.item()
    val_loss = val_running_loss / (idx + 1)

    print("-" * 30)
    print(f"Train Loss Epoch {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss Epoch {epoch+1}: {val_loss:.4f}")
    print(f"Train Accuracy EPOCH {epoch + 1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
    print(f"Valid Accuracy EPOCH {epoch + 1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print("-" * 30)

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")

#memory safety with cuda
torch.cuda.empty_cache()

#testing
labels = []
ids = []
imgs = []
model.eval()
with torch.no_grad():
    for idx, sample in enumerate(tqdm(ts_dl, position=0, leave=True)):
        img = sample["image"].to(device)
        ids.extend([int(i)+1 for i in sample["index"]])

        outputs = model(img)

        imgs.extend(img.detach().cpu())
        labels.extend([int(i) for i in torch.argmax(outputs, dim=1)])

#printing results
plt.figure()
f, axarr = plt.subplots(2, 3)
counter = 0
for i in range(2):
    for j in range(3):
        axarr[i][j].imshow(imgs[counter].squeeze(), cmap="gray")
        axarr[i][j].set_title(f"Predicted {labels[counter]}")
        counter += 1

plt.show()