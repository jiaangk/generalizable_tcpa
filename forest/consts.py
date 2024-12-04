"""Setup constants, ymmv."""

PIN_MEMORY = True
NON_BLOCKING = True
BENCHMARK = True
MAX_THREADING = 40
SHARING_STRATEGY = 'file_descriptor'  # file_system or file_descriptor

DEBUG_TRAINING = False

DISTRIBUTED_BACKEND = 'gloo'  # nccl would be faster, but require gpu-transfers for indexing and stuff

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
tiny_imagenet_mean = [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
tiny_imagenet_std = [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

DATASET_SETTING = {
    'CIFAR10': {
        'in_channels': 3,
        'num_classes': 10,
        'resolution_is_fix': True
    },
    'CIFAR100': {
        'in_channels': 3,
        'num_classes': 100,
        'resolution_is_fix': True
    },
    'CIFAR100_20': {
        'in_channels': 3,
        'num_classes': 20,
        'resolution_is_fix': True
    },
    'MNIST': {
        'in_channels': 1,
        'num_classes': 10,
        'resolution_is_fix': True
    },
    'CelebA': {
        'in_channels': 3,
        'num_classes': 2,
        'resolution_is_fix': False
    },
    'custom': {
        'in_channels': 3,
        'num_classes': 10,
        'resolution_is_fix': False
    },
    'ImageNet': {
        'in_channels': 3,
        'num_classes': 10,
        'resolution_is_fix': False
    }
}

PRETRAINED_MODELS = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'inception_v3': 2048,
    'convnext_tiny': 768,
    'convnext_small': 768,
    'convnext_base': 1024,
    'convnext_large': 1536,
    'efficientnet_v2_s': 1280,
    'efficientnet_v2_m': 1280,
    'efficientnet_v2_l': 1280,
    'swin_v2_t': 768,
    'swin_v2_s': 768,
    'swin_v2_b': 1024,
    'dinov2_vits14': 384,
    'dinov2_vits14_reg': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitb14_reg': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitl14_reg': 1024,
    'dinov2_vitg14': 1536,
    'dinov2_vitg14_reg': 1536,
    'clip_vitb32': 512,
    'clip_vitb16': 512,
    'clip_vitl14': 768,
    'align': 640
}

DB_CONSTRUCT = """
CREATE TABLE poisoned_result (
    ID                    INTEGER PRIMARY KEY,
    [Feature Extractor]   TEXT,
    Loss                  TEXT,
    Model                 TEXT,
    [Eval Model]          TEXT,
    Dataset               TEXT,
    Seed                  TEXT,
    Recipe                TEXT,
    Realistic             INTEGER,
    Defense               TEXT,
    [Training Aug]        INTEGER,
    [Poison Aug]          INTEGER,
    pbatch                INTEGER,
    epochs                INTEGER,
    eps                   INTEGER,
    Budget                NUMERIC,
    Restarts              INTEGER,
    Retrain               INTEGER,
    Clusters              INTEGER,
    [Cluster Model]       TEXT,
    [Sp Target]           INTEGER,
    [Training Targets]    INTEGER,
    [Valid Targets]       INTEGER,
    [Clean adv. loss]     REAL,
    [Clean fool acc.]     REAL,
    [Clean orig. loss]    REAL,
    [Clean orig. acc]     REAL,
    [Poisoned adv. loss]  REAL,
    [Poisoned fool acc.]  REAL,
    [Poisoned orig. loss] REAL,
    [Poisoned orig. acc]  REAL,
    [Clean valid acc.]    REAL,
    [Poisoned valid acc.] REAL
);
"""

DB_INSERT = """
INSERT INTO poisoned_result (
    [Feature Extractor], Loss, Model, [Eval Model], Dataset, Seed, Recipe, Realistic, Defense, [Training Aug], 
    [Poison Aug], pbatch, epochs, eps, Budget, Restarts, Retrain, Clusters, [Cluster Model],
    [Sp Target], [Training Targets], [Valid Targets], [Clean adv. loss], [Clean fool acc.], [Clean orig. loss], 
    [Clean orig. acc], [Poisoned adv. loss], [Poisoned fool acc.], [Poisoned orig. loss], [Poisoned orig. acc], 
    [Clean valid acc.], [Poisoned valid acc.]
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""