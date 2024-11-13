import pandas as pd
import os
import chardet
import numpy as np
import torch.nn as nn
import torch
import argparse
# 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_len=512):
        super(TransformerModel, self).__init__()
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, model_dim))
        
        # Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # 输出层
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.conv = nn.Sequential(nn.Conv1d(1, model_dim, kernel_size=3),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(model_dim))
    def forward(self, x):
        # 嵌入层
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.embedding(x)
        
        # 添加位置编码
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer 编码器
        x = self.transformer(x)
        
        # 输出层
        x = self.fc_out(x)
        x = x.view(x.shape[0],-1)
        #print('out',x.shape)
        
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool wear detection algorithm')
    parser.add_argument('--csv_file_path', type=str,default='./peakfreq_20240308.csv',help='path to the file')
    parser.add_argument('--model_path', type=str,default='./best_model.pth',help='path to the file')
    parser.add_argument('--thereshold', type=int,default=5,help='threshold')
    parser.add_argument('--tool_id', type=int,default=2,help='tool id')  
    # args parse
    args = parser.parse_args()

    #数据读入
    csv_file_path = args.csv_file_path
    model_path = args.model_path

    with open(csv_file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding=encoding)
    # 将时间列转换为 datetime 类型
    df['time'] = pd.to_datetime(df['time'])

    # 初始化一个空列表来存储DataFrame
    dataframes = []

    if 'cutterSpacing' in df.columns:
        df_2_cutter = df[df['cutterSpacing'] == args.tool_id]  # 假设2号刀在'cutterSpacing'列中的值为整数2
        dataframes.append(df_2_cutter)
        print('遍历中请等待........')
    else:
        print(f"警告：文件 {csv_file_path} 中不存在 'cutterSpacing' 列，已跳过。")


    # 使用pd.concat合并所有DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    # 按照'time'列进行升序排列
    combined_df = combined_df.sort_values(by='time', ascending=True)


    # 打印列名，检查数据
    print("原始列名:", combined_df.columns.tolist())

    # 处理列名中的空格
    combined_df.columns = combined_df.columns.str.strip()

    # 再次检查列名
    print("处理后的列名:", combined_df.columns.tolist())

    # 读取特征、时间
    features = combined_df.iloc[:, 1:120].values
    time = combined_df.iloc[:, 0].values

    print(features.shape)
    print(time)

    #超参
    input_dim = 117  # 输入特征维度
    model_dim = 64  # Transformer 模型维度
    num_heads = 4   # 多头注意力机制的头数
    num_layers = 3  # Transformer 层数
    output_dim = 2  # 输出特征维度
    max_len = 64   # 最大序列长度
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 预测
    model.to(device)
    features_tensor = torch.tensor(features, dtype=torch.float32)  
    features_tensor = features_tensor.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        predictions = model(features_tensor)
        print(predictions.shape)
        _, predictions = torch.max(predictions, dim=-1)
        predictions = predictions.cpu().numpy()

    print(np.sum(predictions==1))
    # 标记异常数据
    combined_df['predicted_status'] = predictions
    combined_df['is_abnormal'] = combined_df['predicted_status'] == 0

    # 保存为 CSV 文件
    combined_df.to_csv('combined_df_with_is_abnormal.csv', index=False)

    # 统计首次提示异常时与总时长的占比

    abnormal_df = combined_df[combined_df['is_abnormal']]
    df = combined_df[['time', 'is_abnormal']]

    # 将 'time' 列和 'is_abnormal' 列分别存储在两个列表中
    time_list = df['time'].tolist()
    is_abnormal_list = df['is_abnormal'].tolist()


    ## 计算首次提示异常的时间 
    lenght = len(time_list)
    threshold = args.thereshold
    sums = 0
    i = 0
    ans = []
    first_abnormal_time = None

    while i < lenght:
        if is_abnormal_list[i] == True:
            sums += 1
            ans.append(time_list[i])
            i += 1
            if sums >= threshold:
                first_abnormal_time = ans[0]
                break
        else:
            i += 1
            sums = 0
            ans = []

    # 打印结果以验证
    print("First Abnormal Time:", first_abnormal_time)

    total_time = (combined_df['time'].max() - combined_df['time'].min()).total_seconds() / 3600

    if pd.notna(first_abnormal_time):
        time_to_first_abnormal = (first_abnormal_time - combined_df['time'].min()).total_seconds() / 3600
        ratio = time_to_first_abnormal / total_time
    else:
        ratio = 0

    print(f"首次提示异常时与总时长的占比: {ratio:.4f}")


