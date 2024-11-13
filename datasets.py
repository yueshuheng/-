import argparse
import os
import pandas as pd
import chardet
from datetime import timedelta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data labeling')
    parser.add_argument('--file_path_lables', type=str,default='C:/Users/U/Desktop/特征数据/刀具换刀日志(TEST-20240304-20240318).csv',help='path to the file')
    parser.add_argument('--file_path_data', type=str,default='./combined_2_cutter_data.csv',help='path to the file')
    parser.add_argument('--housre_gap', type=int, default=6, help='hours')
    # args parse
    args = parser.parse_args()
    file_path_data = args.file_path_data
    file_path_lable = args.file_path_lables
    hours = args.housre_gap
    # 检测文件编码
    with open(file_path_data, 'rb') as f:
        result = chardet.detect(f.read())
        encoding_data = result['encoding']

    with open(file_path_lable, 'rb') as f:
        result = chardet.detect(f.read())
        encoding_label = result['encoding']

    print(encoding_data,encoding_label)
    # 读取 peakfreq_日期.csv 文件
    peakfreq_df = pd.read_csv(file_path_data, encoding=encoding_data)

    # 读取 刀具换刀日志.csv 文件
    tool_change_df = pd.read_csv(file_path_lable, encoding=encoding_label)

    # 将时间列转换为 datetime 类型
    peakfreq_df['time'] = pd.to_datetime(peakfreq_df['time'])
    tool_change_df['\t换刀时间\t'] = pd.to_datetime(tool_change_df['\t换刀时间\t'])
    tool_change_df = tool_change_df[tool_change_df['\t刀位号\t']==2]


    def label_data(row, tool_change_times):
        for idx, change_time in tool_change_times.iterrows():
            before_time = change_time['\t换刀时间\t'] - timedelta(hours=hours)
            after_time = change_time['\t换刀时间\t'] + timedelta(hours=hours)
            
            if before_time <= row['time'] < change_time['\t换刀时间\t']:
                return 0  # 磨损标签
            elif change_time['\t换刀时间\t'] <= row['time'] < after_time:
                return 1  # 正常标签
        
        return None  # 如果不在任何换刀时间的前后6小时内，则返回 None


    # 创建一个新的列 'label' 并应用标记函数
    peakfreq_df['label'] = peakfreq_df.apply(lambda row: label_data(row, tool_change_df), axis=1)
    # 删除未标记的数据
    peakfreq_df = peakfreq_df.dropna(subset=['label'])
    # 保存标记后的数据
    peakfreq_df.to_csv('labeled_peakfreq_data.csv', index=False)