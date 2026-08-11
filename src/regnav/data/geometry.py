import numpy as np
from numpy.typing import NDArray


def _wrap_yaw(yaw: NDArray[np.floating]) -> NDArray[np.float32]:
    return np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)


def poses_to_local(
    positions: NDArray[np.floating],
    yaws: NDArray[np.floating],
    anchor: int,
) -> NDArray[np.float32]:
    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [N,2]")
    if yaws.shape != (len(positions),):
        raise ValueError("yaws must have shape [N]")

    yaw = yaws[anchor]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, sine], [-sine, cosine]], np.float32)
    xy = (positions - positions[anchor]) @ rotation.T
    return np.column_stack((xy, _wrap_yaw(yaws - yaw))).astype(np.float32)


def resample_future(
    timestamps: NDArray[np.floating],
    poses: NDArray[np.floating],
    query_times: NDArray[np.floating],
) -> NDArray[np.float32]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float32)
    query_times = np.asarray(query_times, dtype=np.float64)
    if timestamps.ndim != 1 or poses.shape != (len(timestamps), 3):
        raise ValueError("timestamps and poses must have shapes [N] and [N,3]")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if query_times.ndim != 1 or np.any(np.diff(query_times) < 0):
        raise ValueError("query_times must be increasing")
    if len(query_times) and (
        query_times[0] < timestamps[0] or query_times[-1] > timestamps[-1]
    ):
        raise ValueError("query_times must be within the available range")

    result = np.empty((len(query_times), 3), np.float32)
    result[:, 0] = np.interp(query_times, timestamps, poses[:, 0])
    result[:, 1] = np.interp(query_times, timestamps, poses[:, 1])
    result[:, 2] = _wrap_yaw(
        np.interp(query_times, timestamps, np.unwrap(poses[:, 2]))
    )
    return result


def route_goal_from_trajectory(
    trajectory: NDArray[np.floating], turn_threshold: float = 0.35
) -> NDArray[np.float32]:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) == 0:
        raise ValueError("trajectory must have shape [N,3] with at least one pose")
    yaw = float(_wrap_yaw(trajectory[-1:, 2])[0])
    if yaw > turn_threshold:
        turn = [1.0, 0.0, 0.0]
    elif yaw < -turn_threshold:
        turn = [0.0, 0.0, 1.0]
    else:
        turn = [0.0, 1.0, 0.0]
    return np.concatenate((trajectory[-1], np.array(turn, np.float32)))
