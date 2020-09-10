import argparse

from core.datasets.preprocess import run_preprocess
from tasks.pretraining import run as pretrain
from tasks.translation import run as translate

from rdkit import rdBase
rdBase.DisableLog("rdApp.*")


def command_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    sub_preprocess = sub.add_parser('preprocess', help="Preprocess a given dataset.")
    sub_preprocess.add_argument("--dataset-name", default="ZINC", help="Dataset name.")
    sub_preprocess.set_defaults(command='preprocess')

    sub_pretrain = sub.add_parser('pretrain', help="Pretrain.")
    sub_pretrain.add_argument("--dataset-name", default="ZINC", help="Dataset name.")
    sub_pretrain.add_argument("--hparams-file", default="hparams.yml", help="HParams file.")
    sub_pretrain.add_argument("--output-dir", default="RESULTS", help="Output folder.")
    sub_pretrain.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
    sub_pretrain.add_argument("--gpu", default=1, help="GPU number.")
    sub_pretrain.set_defaults(command='pretrain')

    sub_translate = sub.add_parser('translate', help="Run a task.")
    sub_translate.add_argument("--pretrain-from", default="moses", help="Dataset name.")
    sub_translate.add_argument("--dataset-name", default="ZINC", help="Dataset name.")
    sub_translate.add_argument("--hparams-file", default="hparams.yml", help="HParams file.")
    sub_translate.add_argument("--output-dir", default="RESULTS", help="Output folder.")
    sub_translate.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
    sub_translate.add_argument("--gpu", default=1, help="GPU number.")
    sub_translate.set_defaults(command='translate')

    return parser


if __name__ == "__main__":
    parser = command_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(args.dataset_name)
    elif args.command == "pretrain":
        pretrain(args)
    elif args.command == "translate":
        translate(args)
