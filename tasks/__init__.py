from tasks.autoencoding.experiment import run_train as autoencode
from tasks.pretraining.experiment import run_train as pretrain
# from tasks.translation.experiment import run_train as translate

TASKS = {
    "pretrain": pretrain,
    "autoencode": autoencode
}