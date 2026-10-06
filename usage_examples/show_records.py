"""Print the records of the first few patients, and the shape of one batch per split."""
import argparse
import os
from typing import Annotated
import typer
from typer import Option
from dotenv import load_dotenv


from src.data.module import CycleDataModule

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')

def main(
    data_dir: Annotated[str,Option(prompt=True,help='Path to the data directory')] = DATA_DIR,
    max_records: Annotated[int,Option(prompt=True,help='Maximum number of records to load per split')] = 3,
):

    datamodule = CycleDataModule(data_dir, max_records=max_records, num_workers=0)
    datamodule.setup('fit')

    for name, loader in (('train', datamodule.train_dataloader()), ('dev', datamodule.val_dataloader())):
        print(f'== {name}: {len(loader.dataset)} items')

        for record in loader.dataset.records:
            dates = ' -> '.join(str(exam.date) for exam in record.exams)
            print(f'{record.patient_id}  {record.side} {record.view:<3}  {dates}')

        frames = next(iter(loader))['frames']
        print(f'first batch: frames {tuple(frames.shape)}, values in [{frames.min():.2f}, {frames.max():.2f}]\n')


if __name__ == '__main__':
    typer.run(main)
