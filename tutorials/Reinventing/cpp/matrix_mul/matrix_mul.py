import torch

a = torch.tensor([[0,1,2],[1,2,3]])
b = a.T
out = torch.zeros(2,2)

for k in range(3):
    for c in range(2):
        temp = b[k][c]
        for r in range(2):
            out[r][c]+=a[r][k]*temp
print(out)
