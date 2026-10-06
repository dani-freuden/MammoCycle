#%%
from src.data.module import CycleDataModule




# %%
data_dir = "path/to/data"
image_size = 240
batch_size = 32
num_workers = 8

data_module = CycleDataModule(
    data_dir=data_dir,
    image_size=image_size,
    batch_size=batch_size,
    num_workers=num_workers,
)

