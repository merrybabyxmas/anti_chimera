from anti_chimera.data.scene_hint import SceneHintBuilder


def build_scene_hint_builder(data_cfg):
    return SceneHintBuilder(
        max_objects=int(data_cfg['max_objects']),
        depth_bins=int(data_cfg['depth_bins']),
        image_size=int(data_cfg['image_size']),
    )
