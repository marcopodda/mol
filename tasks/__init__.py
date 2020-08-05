from tasks.pretraining.experiment import run as pretrain
from tasks.autoencoding.experiment import run as autoencode
from tasks.generation.experiment import run as generate
from tasks.optimization.experiment import run as optimize
from tasks.translation.experiment import run as translate


TASKS = {
    "pretrain": pretrain,
    "autoencode": autoencode,
    "generate": generate,
    "optimize": optimize,
    "translate": translate
}