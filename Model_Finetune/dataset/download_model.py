from modelscope import snapshot_download

# model_dir = snapshot_download(
#     'Qwen/Qwen3-VL-8B-Instruct',
#     cache_dir='./cache'
# )
# print(model_dir)

model_dir = snapshot_download(
    'Qwen/Qwen3-VL-2B-Instruct',
    cache_dir='./cache'
)
print(model_dir)