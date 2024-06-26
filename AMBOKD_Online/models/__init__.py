from .EEGNet import EEGNet
from .TSception import TSception
from .MCGRAM import MCGRAM
from .MobileNet import MobileNet
from .ResNet50 import ResNet50
from .EfficientNet import EfficientNet_b0
from .CMM import CMM
from .MLB import MLB
from .AMM import AMM
from .EML import EML
from .DML import DML
from .KDCL import KDCL
from .MMOKD import MMOKD

teacher_model_dict = {
    'EEGNet': (EEGNet,'./models/EEGNet_0229'),
    'TSception': (TSception,'./models/TSception_0229'),
    'MCGRAM': (MCGRAM,'./models/MCGRAM_0229'),
    'MobileNet': (MobileNet,'./models/MobileNet_0229'),
    'ResNet50': (ResNet50,'./models/ResNet50_0229'),
    'EfficientNet': (EfficientNet_b0,'./models/EfficientNet_0229'),
}

model_dict = {
    'CMM': CMM,
    'MLB': MLB,
    'AMM': AMM,
    'EML': EML,
    'DML': DML,
    'KDCL': KDCL,
    'MMOKD': MMOKD,

}