import torch
import linear_w8a8

# Example input and weights
input_tensor = torch.randint(-128, 127, (1024, 2048), dtype=torch.int8, device='cuda')
weights_tensor = torch.randint(-128, 127, (2048, 4096), dtype=torch.int8, device='cuda')
print("input_tensor", input_tensor)
print("weights_tensor", weights_tensor)
# Perform the linear operation
output_tensor = linear_w8a8.linear_w8a8(input_tensor, weights_tensor)
# print("output_tensor", output_tensor, output_tensor.shape)

lin_out = torch.nn.functional.linear(input_tensor.float(), weights_tensor.float().T)

print(torch.allclose(output_tensor.float(), lin_out, atol=1e-3))

