import torch
from torch.utils.data import DataLoader, TensorDataset
    
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)

def test_dataloader(num_workers):
    try:
        dataloader = DataLoader(dataset, batch_size=10, num_workers=num_workers)
        for batch in dataloader:
            pass
        print(f"Success with num_workers = {num_workers}")
    except Exception as e:
        print(f"Failed with num_workers = {num_workers}: {e}")
