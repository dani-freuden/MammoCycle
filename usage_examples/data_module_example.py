#%%
from src.data.module import CycleDataModule




# %%
data_dir = "path/to/data"
height = 1024
width = 640
batch_size = 32
num_workers = 8

data_module = CycleDataModule(
    data_dir=data_dir,
    height=height,
    width=width,
    batch_size=batch_size,
    num_workers=num_workers,
)

