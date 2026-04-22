from keras_hub.src.models.phi4_mini.phi4_mini_backbone import Phi4MiniBackbone
from keras_hub.src.models.phi4_mini.phi4_mini_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, Phi4MiniBackbone)
