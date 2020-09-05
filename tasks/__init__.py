from tasks.pretraining.experiment import run as pretrain
from tasks.translation.experiment import run as translate


TASKS = {
    "pretraining": pretrain,
    "translation": translate
}