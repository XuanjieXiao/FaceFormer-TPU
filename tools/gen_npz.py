import torch
import numpy as np
import os

# 设置目标文件夹路径
test_path = '../models/testInput'

# 检查文件夹是否存在
if not os.path.exists(test_path):
    # 文件夹不存在，创建文件夹
    os.makedirs(test_path)
    print("文件夹已创建：", test_path)
else:
    print("文件夹已存在：", test_path)

random_tensor = torch.rand(1, 262144)
random_tensor2 = torch.rand(1, 490, 512)
one_hot_random_tensor = torch.rand(1, 8)
vertice_input = torch.rand(1, 490, 64)
hidden_states = torch.rand(1, 490, 64)
memory_mask = torch.rand(490, 490)
embedding = torch.rand(1, 490, 64)

np.savez(os.path.join(test_path, 'input_encoder_1.npz'), **{'audio_feature':random_tensor})
np.savez(os.path.join(test_path, 'input_encoder_2.npz'), **{'hidden_states': random_tensor2,'one_hot':one_hot_random_tensor})
np.savez(os.path.join(test_path, 'input_ppe.npz'), **{'embedding': embedding})
np.savez(os.path.join(test_path, 'input_decoder.npz'), **{'vertice_input': vertice_input,'hidden_states':hidden_states,'memory_mask':memory_mask})
print("测试数据已生成：", test_path)