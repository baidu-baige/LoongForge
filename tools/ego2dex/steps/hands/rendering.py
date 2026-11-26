# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Retain every rendered geom while assigning dense segmentation IDs."""

from steps.retarget.rendering import make_renderer as original_make_renderer


def make_renderer(model, width, height):
    renderer = original_make_renderer(model, width, height)
    update = renderer.update_scene

    def update_scene(*args, **kwargs):
        import copy
        import mujoco

        option = copy.copy(kwargs.get("scene_option")) if kwargs.get("scene_option") is not None else mujoco.MjvOption()
        option.sitegroup[:] = 0
        option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        kwargs["scene_option"] = option
        result = update(*args, **kwargs)
        # Vendor sensor decorations can leave gaps in MuJoCo's scene segids.
        # The renderer's lookup assumes IDs < scene.ngeom. Dense reassignment
        # changes IDs only; it never changes geometry, visibility or masks.
        for i in range(renderer.scene.ngeom):
            renderer.scene.geoms[i].segid = i
        return result

    renderer.update_scene = update_scene
    return renderer


def render_geom_subset_mask(model, data, renderer, camera, geom_ids):
    import mujoco
    import numpy as np

    ids = np.asarray(geom_ids, dtype=int)
    if not len(ids):
        return np.zeros((renderer.height, renderer.width), dtype=bool)
    groups = model.geom_group.copy()
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    keep = np.zeros(model.ngeom, bool)
    keep[ids] = True
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    try:
        model.geom_group[:] = np.where(keep, 0, 5)
        renderer.update_scene(data, camera=camera, scene_option=option)
        depth = renderer.render().copy()
        far = float(model.vis.map.zfar * model.stat.extent)
        return np.isfinite(depth) & (depth > 0) & (depth < far * 0.999)
    finally:
        model.geom_group[:] = groups
        renderer.disable_depth_rendering()


def visible_category_masks(model, data, renderer, camera, categories):
    import mujoco
    import numpy as np

    groups = model.geom_group.copy()
    masks = []
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=camera)
        full = renderer.render().copy()
        far = float(model.vis.map.zfar * model.stat.extent)
        for ids in categories:
            keep = np.zeros(model.ngeom, bool)
            keep[np.asarray(ids, dtype=int)] = True
            model.geom_group[:] = np.where(keep, 0, 5)
            renderer.update_scene(data, camera=camera, scene_option=option)
            depth = renderer.render().copy()
            masks.append(np.isfinite(depth) & (depth > 0) & (depth < far * 0.999) & (abs(depth - full) < 1e-5))
        return masks
    finally:
        model.geom_group[:] = groups
        renderer.disable_depth_rendering()
