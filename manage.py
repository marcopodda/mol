import argparse
from pathlib import Path

from core.config import Config
from core.datasets.preprocess import run_preprocess
from tasks import TASKS

from rdkit import rdBase
rdBase.DisableLog("rdApp.*")


def command_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    sub_preprocess = sub.add_parser('preprocess', help="Preprocess a given dataset.")
    sub_preprocess.add_argument("--dataset-name", default="ZINC", help="Dataset name.")
    sub_preprocess.set_defaults(command='preprocess')

    sub_run = sub.add_parser('run', help="Run a task.")
    sub_run.add_argument("--task", required=True, choices=TASKS.keys(), help="Task name.")
    sub_run.add_argument("--dataset-name", default="ZINC", help="Dataset name.")
    sub_run.add_argument("--config-file", default="config.yml", help="Config file.")
    sub_run.add_argument("--output-dir", default="RESULTS", help="Output folder.")
    sub_run.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
    sub_run.add_argument("--gpu", default=1, help="GPU number.")
    sub_run.set_defaults(command='run')

    return parser


if __name__ == "__main__":
    parser = command_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(args.dataset_name)
    elif args.command == "run":
        task = TASKS[args.task]
        task(args)
