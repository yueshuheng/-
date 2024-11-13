import argparse
import os
import pandas as pd
import chardet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Effective data selection')
    parser.add_argument('--folder_path', type=str,default='../', help='Path to the folder containing the data')
    parser.add_argument('--save_path',type=str,default='./', help='Path to save the selected data')

    # args parse
    args = parser.parse_args()
    # 设置文件夹路径
    folder_path = args.folder_path
    save_path =args.save_path

    # 获取文件夹中的所有文件列表
    all_files = os.listdir(folder_path)

    # 筛选出所有CSV文件
    csv_files = [file for file in all_files if file.endswith('.csv')]

    # 初始化一个空列表来存储DataFrame
    dataframes = []

    # 遍历每个CSV文件
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        
        df = pd.read_csv(file_path, encoding=encoding)
        # 检查是否存在'cutterSpacing'列，并筛选出2号刀的数据
        if 'cutterSpacing' in df.columns:
            df_2_cutter = df[df['cutterSpacing'] == 2]  # 假设2号刀在'cutterSpacing'列中的值为整数2
            dataframes.append(df_2_cutter)
            print('遍历中请等待........')
        else:
            print(f"警告：文件 {csv_file} 中不存在 'cutterSpacing' 列，已跳过。")

    # 使用pd.concat合并所有DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    # 按照'time'列进行降序排列
    combined_df = combined_df.sort_values(by='time', ascending=False)

    # 输出合并后的DataFrame的前几行以验证结果
    print(combined_df.head())
    
    output_file_path = os.path.join(save_path, 'combined_2_cutter_data.csv')
    combined_df.to_csv(output_file_path, index=False)