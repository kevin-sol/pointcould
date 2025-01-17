import open3d as o3d
import numpy as np

#point_cloud = o3d.io.read_point_cloud("e:/code/Volumetric-Video-Streaming/longdress_viewdep_vox12/longdress_viewdep_vox12.ply")
# print(pcd)
point_cloud = o3d.io.read_point_cloud("E:/CODE/pointcloud_video-main/model_compare/data/Videos/longdress/longdress/Ply/longdress_vox10_1051.ply")

coord_frame0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])

points_np = np.asarray(point_cloud.points)
points_np[:, 2] *= -1

# 将修改后的NumPy数组重新赋值给点云
point_cloud.points = o3d.utility.Vector3dVector(points_np)
# o3d.visualization.draw_geometries([coord_frame,coord_frame0,pcd])

# point_cloud.translate(np.array([-45.2095, 7.18301, -54.3561]))
scaling_factor = 0.0018 # 缩放为0.1倍
point_cloud.scale(0.00179523,np.array([0, 0, 0]))

# 创建变换矩阵
rotation_angle = -np.pi / 2 # 90 度

# 定义绕 Y 轴的旋转矩阵
rotation_matrix = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],[0, 1, 0], [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
])

# 对点云进行旋转
point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
print("success")
