from anti_chimera.data.scene_hint import SceneHintBuilder
from anti_chimera.data.scene_hint_minimal import MinimalSceneHintBuilder


def build_scene_hint_builder(data_cfg):
    mode = str(data_cfg.get('scene_hint_mode', 'minimal')).lower()
    if mode == 'minimal':
        return MinimalSceneHintBuilder(
            max_objects=int(data_cfg['max_objects']),
            depth_bins=int(data_cfg['depth_bins']),
            image_size=int(data_cfg['image_size']),
        )
    return SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )
