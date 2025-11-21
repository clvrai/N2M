import numpy as np

def rotate_points_xy(points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    return points @ rotation_matrix

def rotate_points_se2(points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    points[:2] = points[:2] @ rotation_matrix
    points[2] = points[2] - angle
    return points

def rotate_points_xythetaz(points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    points[:2] = points[:2] @ rotation_matrix
    points[2] = points[2] - angle
    return points

def translate_points_xy(points, radius, angle):
    translation_matrix = radius * np.array([np.cos(angle), np.sin(angle), 0])
    return points + translation_matrix

def translate_points_xythetaz(points, radius, angle):
    translation_matrix = radius * np.array([np.cos(angle), np.sin(angle), 0, 0])
    return points + translation_matrix

def apply_rotation_se2(point_cloud, target_point, rotation):
    min_angle = rotation['min_angle'] * np.pi / 180
    max_angle = rotation['max_angle'] * np.pi / 180

    angle = np.random.uniform(min_angle, max_angle)
    point_cloud[:, :3] = rotate_points_xy(point_cloud[:, :3], angle)
    target_point = rotate_points_se2(target_point, angle)
    return point_cloud, target_point

def apply_rotation_xy(point_cloud, target_point, rotation):
    min_angle = rotation['min_angle'] * np.pi / 180
    max_angle = rotation['max_angle'] * np.pi / 180

    angle = np.random.uniform(min_angle, max_angle)
    point_cloud[:, :3] = rotate_points_xy(point_cloud[:, :3], angle)
    target_point = rotate_points_xy(target_point, angle)
    return point_cloud, target_point

def apply_translation_xy(point_cloud, target_point, translation):
    radius = np.random.uniform(0, translation['radius'])
    angle = np.random.uniform(0, 2 * np.pi)
    point_cloud[:, :3] = translate_points_xy(point_cloud[:, :3], radius, angle)
    target_point = translate_points_xy(target_point, radius, angle)
    return point_cloud, target_point

def apply_rotation_xythetaz(point_cloud, target_point, rotation):
    min_angle = rotation['min_angle'] * np.pi / 180
    max_angle = rotation['max_angle'] * np.pi / 180

    angle = np.random.uniform(min_angle, max_angle)
    point_cloud[:, :3] = rotate_points_xy(point_cloud[:, :3], angle)
    target_point = rotate_points_xythetaz(target_point, angle)
    return point_cloud, target_point

def apply_translation_xythetaz(point_cloud, target_point, translation):
    radius = np.random.uniform(0, translation['radius'])
    angle = np.random.uniform(0, 2 * np.pi)
    point_cloud[:, :3] = translate_points_xy(point_cloud[:, :3], radius, angle)
    target_point = translate_points_xythetaz(target_point, radius, angle)
    return point_cloud, target_point

def apply_augmentations(point_cloud, target_point, augmentations):
    if 'rotation_xy' in augmentations:
        point_cloud, target_point = apply_rotation_xy(point_cloud, target_point, augmentations['rotation_xy'])
    if 'rotation_se2' in augmentations:
        point_cloud, target_point = apply_rotation_se2(point_cloud, target_point, augmentations['rotation_se2'])
    if 'rotation_xythetaz' in augmentations:
        point_cloud, target_point = apply_rotation_xythetaz(point_cloud, target_point, augmentations['rotation_xythetaz'])
    if 'translation_xy' in augmentations:
        point_cloud, target_point = apply_translation_xy(point_cloud, target_point, augmentations['translation_xy'])
    if 'translation_xythetaz' in augmentations:
        point_cloud, target_point = apply_translation_xythetaz(point_cloud, target_point, augmentations['translation_xythetaz'])
    return point_cloud, target_point

def fix_point_cloud_size(point_cloud, pointnum):
    if point_cloud.shape[0] > pointnum:
        indices = np.random.choice(point_cloud.shape[0], pointnum, replace=False)
        point_cloud = point_cloud[indices]
    elif point_cloud.shape[0] < pointnum:
        padding = np.zeros((pointnum - point_cloud.shape[0], point_cloud.shape[1]))
        point_cloud = np.vstack([point_cloud, padding])
    return point_cloud

# def translate_se2_point_cloud(point_cloud, se2_transform):
#     point_cloud[:, :3] = point_cloud[:, :3] + np.array([se2_transform[0], se2_transform[1], 0])
#     point_cloud[:, :3] = rotate_points_xy(point_cloud[:, :3], se2_transform[2])
#     return point_cloud

def translate_se2_target(target_point, se2_transform):
    target_point = rotate_points_se2(target_point, se2_transform[2])
    target_point = target_point + np.array([se2_transform[0], se2_transform[1], 0])
    return target_point