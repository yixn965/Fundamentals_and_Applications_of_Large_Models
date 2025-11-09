import pandas as pd

# 加载数据文件并进行预处理
file_path = 'data/en-cn/cmn.txt'

# 读取文件并处理每一行，提取英文和中文句子
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 每行数据使用制表符分割，提取英文和中文部分
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            english_sentence = parts[0].strip()
            chinese_sentence = parts[1].strip()
            data.append([english_sentence, chinese_sentence])

# 创建 DataFrame 保存提取的句子
df = pd.DataFrame(data, columns=['English', 'Chinese'])

# 将处理后的英文和中文句子分别保存为两个文件
df['English'].to_csv('data/english_sentences.txt', index=False, header=False)
df['Chinese'].to_csv('data/chinese_sentences.txt', index=False, header=False)

# 显示前几行以验证处理是否正确
print(df.head())