import os
import shutil
from pathlib import Path


def fetch_dataset(info, dest_dir):
    filename = info['filename']
    url = info['url']
    unzip = info['unzip']

    temp_folder = Path("./temp").absolute()
    if not temp_folder.exists():
        os.makedirs(temp_folder)

    os.system(f'wget -P {temp_folder} {url}')

    temp_path = temp_folder / filename

    if unzip is True:
        if ".tar.gz" in info['url']:
            os.system(f'tar xvf {temp_path}.tar.gz -C {temp_folder}')
        elif '.zip' in info['url']:
            os.system(f'unzip {temp_path.with_suffix(".zip")} -d {temp_folder}')
        elif '.gz' in info['url']:
            os.system(f'gunzip {temp_path}.gz')

    source_path = temp_folder / filename
    dest_path = dest_dir / filename

    shutil.move(source_path, dest_path)
    shutil.rmtree(temp_folder)
