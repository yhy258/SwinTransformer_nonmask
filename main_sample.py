import torch
from torchvision import datasets, transforms
from SwinTransformer import SwinTransformer
from tqdm.notebook import tqdm
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE)

swin = SwinTransformer(class_num=10, C=96, num_heads =  [3, 6, 12, 24], window_size =7,  swin_num_list=[1,1,3,1], norm = True, img_size = 224, dropout = 0.1, ffn_dim = 384).to(DEVICE)

"""
    Dataset
"""
transform = transforms.Compose([transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
])


transform_test = transforms.Compose([transforms.ToTensor(),transforms.Resize(224),transforms.Normalize(0.5, 0.5)])

train_dataset = datasets.CIFAR10(
    root='./.data',
    train=True,
    transform = transform,
    download = True
    )

test_dataset = datasets.CIFAR10(
    root='./.data',
    train=False,
    transform = transform_test,
    download = True
    )

train_dataloader = torch.utils.data.DataLoader(train_dataset, 16, True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, 16, True)


"""
    Train Sample
"""

optim = torch.optim.Adam(params=swin.parameters(),lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
swin.train()

epochs = 30

for epoch in range(epochs):
    losses = []
    print("\n Epoch {}/{}".format(epoch+1, epochs))
    for x, y in tqdm(train_dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = swin(x)
        loss = criterion(pred,y)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("This Loss is : {}".format(np.mean(losses)))
