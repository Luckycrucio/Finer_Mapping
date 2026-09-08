"""Small rosbag2 helpers shared by the frame-caching and fusion passes."""
import rosbag2_py


def open_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, type_map


def iter_topic(bag_path, topic):
    """Yield deserialized messages on `topic` in log order from a fresh read."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader, type_map = open_bag(bag_path)
    if topic not in type_map:
        raise KeyError(f"{topic} not found in bag; available: {sorted(type_map)}")
    msg_type = get_message(type_map[topic])
    while reader.has_next():
        this_topic, data, _t = reader.read_next()
        if this_topic == topic:
            yield deserialize_message(data, msg_type)
