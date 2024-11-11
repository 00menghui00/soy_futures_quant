#1_0待改进：均方误差损失函数不行，空白像素点会极大地拉低这个均方误差
'''
你也可以用收益率作为output：（第六日收盘价-第六日开盘价）/第六日开盘价*100%
两个建议：
1.用一张图作为输入，里面包含3日或者5日的K线图，这样比你输入单独的3张或者5张K线图更好，因为可以反映多个K线图之间的位置关系。
2.损失函数调整为分类问题，判断上涨和下跌，看正确率。或者将损失函数设置为次日收益率，看MAPE。
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import time

class KLineImageDataset(Dataset):
    def __init__(self, image_folder, sequence_length=5, transform=None):
        self.image_folder = image_folder  # 传入图片文件夹
        self.sequence_length = sequence_length  # 获取序列长度
        self.transform = transform  # 对图像数据进行预处理或数据增强
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))  # 获取文件并按字母排序

    def __len__(self):
        return len(self.image_paths) - self.sequence_length

    def __getitem__(self, idx):
        images = []
        for i in range(self.sequence_length):
            img_path = self.image_paths[idx + i]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        target = Image.open(self.image_paths[idx + self.sequence_length]).convert('RGB')#确保目标对象是第六日的k线图
        if self.transform:
            target = self.transform(target)
        return torch.stack(images), target


class KLinePredictionModel(nn.Module):
    def __init__(self):
        super(KLinePredictionModel, self).__init__()
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 卷积提取器
        self.lstm = nn.LSTM(128 * 16 * 16, 512, batch_first=True)  # 调整LSTM输入尺寸，适应较小的图像尺寸
        self.fc = nn.Linear(512, 128 * 16 * 16)  # 全连接层
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )  # 反卷积恢复原始图像

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.conv_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out)
        fc_out = fc_out.view(batch_size, 128, 16, 16)  # 调整输出尺寸，以适应较小的图像尺寸
        output = self.deconv(fc_out)
        return output


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU，如果有的话
    model = KLinePredictionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    image_folder = r"F:\Futures_Quant\preprocess\gen_single_day_K_photo\soy_bean_pic"
    sequence_length = 5
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 降低图片分辨率至64x64
        transforms.ToTensor()
    ])
    dataset = KLineImageDataset(image_folder, sequence_length=sequence_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)  # 禁用多线程加速

    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0

        for images, target in dataloader:
            images, target = images.to(device), target.to(device)

            output = model(images)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()  # 每次训练后清空缓存

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch[{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}, Duration: {epoch_duration:.2f}s')


if __name__ == '__main__':
    main()
