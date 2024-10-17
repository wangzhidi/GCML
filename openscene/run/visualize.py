import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

def plot_bounding_box(aabb,fig_data):

    xmin, ymin, zmin = aabb[0]
    xmax, ymax, zmax = aabb[1]
    points = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax]
    ])
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # xmin faces
        [4, 5], [5, 7], [7, 6], [6, 4],  # xmax faces
        [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
    ]

    for edge in edges:
        fig_data.append(go.Scatter3d(
            x=[points[edge[0], 0], points[edge[1], 0]],
            y=[points[edge[0], 1], points[edge[1], 1]],
            z=[points[edge[0], 2], points[edge[1], 2]],
            mode='lines',
            line=dict(width=5, color='black'),
            showlegend=False
        ))

def visualize(points, colors, classes, labelset):



    # 假设 points 是形状为 [n_points, 3] 的点云数组
    # categories 是形状为 [n_points] 的类别数组

    # 1. 过滤出类别为0的点
    mask = classes == 0
    filtered_points = points[mask]
    center=None
    aabb=None
    # 2. 计算密度
    if len(filtered_points) > 0:
        # 使用 KDTree 来计算点的密度
        kdtree = KDTree(filtered_points)
        
        # 选择一个合适的半径，例如0.1，来计算密度
        radius = 1
        counts = np.array([len(kdtree.query_ball_point(point, radius)) for point in filtered_points])

        # 3. 找到密度最高的点
        max_density_index = np.argmax(counts)
        center = filtered_points[max_density_index]
        print(f"position for {labelset[0]}: {center}")

        # 假设 center_point 是找到的中心点，radius 是定义的搜索半径
        radius = 1  # 根据需求调整

        # 过滤出在半径范围内的点
        distances = np.linalg.norm(filtered_points - center, axis=1)
        boundary_mask = (distances <= radius)

        # 获取边界点
        boundary_points = filtered_points[boundary_mask]
        aabb=np.array([np.min(boundary_points,axis=0),np.max(boundary_points,axis=0)])

    else:
        print("positive class not found")



    # # 缩减点云大小，随机采样部分点
    # sample_indices = np.random.choice(len(points), size=len(points)//1, replace=False)
    # sampled_points = points[sample_indices]
    # sampled_colors = colors[sample_indices]
    # # 找到所有类别
    # unique_classes = torch.unique(classes).numpy()
    # unique_classes=unique_classes[:-1]

    # class_zero_count=np.count_nonzero(classes==0)
    # eps = 20  # DBSCAN半径参数
    # min_samples = int(class_zero_count/50+50)  # DBSCAN最小样本数参数
    # centers = {}
    # bounding_boxes = {}
    # # 暂时只识别0号目标类别的中心
    # unique_classes=np.array([0])
    # for cls in unique_classes:
    #     # 获取属于该类别的点
    #     class_mask = classes == cls
    #     class_points = points[class_mask]
            
    #     # 使用DBSCAN去噪和分离实例
    #     db = DBSCAN(eps=eps, min_samples=min_samples).fit(class_points)
    #     labels = db.labels_
        
    #     # 获取所有有效簇
    #     unique_labels = set(labels)
    #     if -1 in unique_labels:
    #         unique_labels.remove(-1)  # 去除噪音簇
        
    #     centers[cls] = []
    #     bounding_boxes[cls] = []
        
    #     for label in unique_labels:
    #         # 获取属于该簇的点
    #         cluster_mask = labels == label
    #         cluster_points = class_points[cluster_mask]
            
    #         # 计算中心
    #         center = cluster_points.mean(axis=0)
    #         # centers[cls].append(center)
    #         centers[cls].append(concentrated_point)
    #         # 计算边界框
    #         min_coords = cluster_points.min(axis=0)
    #         max_coords = cluster_points.max(axis=0)
    #         bounding_box = (min_coords, max_coords)
    #         bounding_boxes[cls].append(bounding_box)

    # print("Centers:", centers)
    # print("Bounding Boxes:", bounding_boxes)


    # visualize
    fig_data = []
    fig_data.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                mode='markers', marker=dict(size=3, color='grey', opacity=1.0)))
    fig_data.append(go.Scatter3d(x=filtered_points[:, 0], y=filtered_points[:, 1], z=filtered_points[:, 2],
                                mode='markers', marker=dict(size=3, color='green', opacity=1.0)))
    # plot center and bounding boxes
    if center is not None:
        fig_data.append(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode='markers', name=labelset[0], marker=dict(size=12, color='yellow')))
        plot_bounding_box(aabb,fig_data)
    
    min_values = points.min(axis=0)  
    max_values = points.max(axis=0)  
    fig = go.Figure(data=fig_data)
    fig.update_layout(scene=dict(xaxis=dict(range=[min_values[0], max_values[0]], autorange=False),
                                yaxis=dict(range=[min_values[1], max_values[1]], autorange=False),
                                zaxis=dict(range=[min_values[2], max_values[2]], autorange=False)),
                    scene_aspectmode='manual',
                    scene_aspectratio=dict(x=1.0, 
                                           y=(max_values[1]-min_values[1])/(max_values[0]-min_values[0]),
                                            z=(max_values[2]-min_values[2])/(max_values[0]-min_values[0])))
    fig.update_layout(template='none')
    fig.write_html(f"{labelset[0]}.html")

    return center,aabb

