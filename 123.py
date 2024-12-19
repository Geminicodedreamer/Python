import torch

# # 检查CUDA是否可用
# if torch.cuda.is_available():
#     # 获取CUDA设备数量
#     device_count = torch.cuda.device_count()
#     print("CUDA可用，共有 {} 个CUDA设备可用:".format(device_count))
#     for i in range(device_count):
#         device = torch.device("cuda:{}".format(i))
#         print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

#         # 获取当前设备的显存使用情况
#         total_memory = torch.cuda.get_device_properties(device).total_memory
#         allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
#         reserved_memory = torch.cuda.memory_reserved(device)    # 已保留的显存
#         free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
#         print("  - 总显存: {:.2f} GB".format(total_memory / (1024 ** 3)))
#         print("  - 已分配的显存: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
#         print("  - 已保留的显存: {:.2f} GB".format(reserved_memory / (1024 ** 3)))
#         print("  - 剩余可用显存: {:.2f} GB".format(free_memory / (1024 ** 3)))
# else:
#     print("CUDA 不可用")


print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
# Create two tensors with 1GB memory footprint each, initialized randomly, in fp16 format
# For a tensor of float16 (2 bytes), 1GB of memory can hold 1GB / 2B = 500M elements
tensor_size = 512 * 1024 * 1024
x = torch.randn(tensor_size, dtype=torch.float16, device='cuda')
y = torch.randn(tensor_size, dtype=torch.float16, device='cuda')

# Record current memory footprint, and reset max memory counter
current_memory = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

print(len(x))
print(len(y))


def compute(x, y):
    return (x + 1) * (y + 1)


z = compute(x, y)

# Record the additional memory (both peak memory and persistent memory) after calculating the resulting tensor
additional_memory = torch.cuda.memory_allocated() - (current_memory + 1e9)
peak_memory = torch.cuda.max_memory_allocated()
additional_peak_memory = peak_memory - (current_memory + 1e9)

print(f"Additional memory used: {additional_memory / (1024 ** 3)} GB")
print(f"Additional peak memory used: {
      additional_peak_memory / (1024 ** 3)} GB")
