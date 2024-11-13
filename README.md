#算法使用文档

#目标刀位的数据筛选
python  data_selective.py --folder_path  /特征数据的文件夹

#2号刀样本打标和分割
python  datasets.py 
样本打标原理：换刀前n个小时为磨损 后n个小时为正常  当前n=4
#磨损检测
python Detection.py --csv_file_path  /特征数据的文件夹  
# -
刀具磨损、深度学习、声反射
