from torch.utils.data.dataloader import default_collate
from collections import defaultdict

def my_collate(batch):
    d = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            d[k].append(v)
    for k, v in d.items():
        if k != "answers":
            d[k] = default_collate(v)
    return d