class Mode3D(object):
    r"""3D coordinate mode helper and constants.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in Camera:

    .. code-block:: none

                z front (yaw=-0.5*pi)
                ^
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (yaw=0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (yaw=0)

    the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAMERA = 1
    DEPTH = 2

    @staticmethod
    def get_mode(mode):
        """Normalize a mode name/enum to a Mode3D constant.

        Args:
            mode: One of {'lidar','camera','depth'} or a Mode3D constant.

        Returns:
            int: Mode3D.LIDAR | Mode3D.CAMERA | Mode3D.DEPTH
        """
        if mode == 'lidar':
            return Mode3D.LIDAR
        elif mode == 'camera':
            return Mode3D.CAMERA
        elif mode == 'depth':
            return Mode3D.DEPTH
        else:
            assert mode in (Mode3D.LIDAR, Mode3D.CAMERA, Mode3D.DEPTH)
            return mode

    @staticmethod
    def get_params(mode, key_name=None):
        """Return useful axis mappings for a given 3D coordinate mode.

        Args:
            mode: Mode name or constant (see ``get_mode``).
            key_name: Optional key to return a single value. When None, return a dict.

        Returns:
            dict | Any: A dict with
                - rot_axis (int): rotation axis index for yaw/pitch/roll
                - flip_axis (tuple[int,int]): horizontal/vertical BEV axes
                - bev_axis (tuple[int,int,int,int,int]): indices to build (cx,cy,w,h,angle)
            or a single field if ``key_name`` is provided.
        """
        mode = Mode3D.get_mode(mode)
        if mode == Mode3D.LIDAR:
            params = dict(rot_axis=2, flip_axis=(1, 0), bev_axis=(0, 1, 3, 4, 8))
        elif mode == Mode3D.CAMERA:
            params = dict(rot_axis=1, flip_axis=(0, 2), bev_axis=(0, 2, 3, 5, -7))
        elif mode == Mode3D.DEPTH:
            params = dict(rot_axis=2, flip_axis=(0, 1), bev_axis=(0, 1, 3, 4, 8))
        else:
            assert False
        return params if key_name is None else params[key_name]
