# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import Point, LineString
# import numpy as np
# import os
#
#
# def split_road_network(input_path, output_nodes_path, output_edges_path):
#     """
#     将道路网络数据拆分为节点和边
#
#     参数:
#     input_path: 输入的道路形状文件完整路径 (不带扩展名)
#     output_nodes_path: 输出节点文件完整路径 (不带扩展名)
#     output_edges_path: 输出边文件完整路径 (不带扩展名)
#     """
#     # 读取道路网络数据
#     print("读取道路网络数据...")
#     roads = gpd.read_file(f"{input_path}.shp")
#
#     # 创建输出目录（如果不存在）
#     os.makedirs(os.path.dirname(output_nodes_path), exist_ok=True)
#     os.makedirs(os.path.dirname(output_edges_path), exist_ok=True)
#
#     # 创建空的节点和边列表
#     nodes = []
#     edges = []
#     node_id = 0
#     edge_id = 0
#     node_coordinates = set()  # 用于存储唯一的节点坐标
#
#     print("处理道路数据...")
#     total_roads = len(roads)
#     # 遍历每条道路
#     for idx, road in roads.iterrows():
#         if idx % 1000 == 0:  # 每处理1000条道路打印一次进度
#             print(f"处理进度: {idx}/{total_roads} ({(idx / total_roads * 100):.2f}%)")
#
#         if isinstance(road.geometry, LineString):
#             # 获取道路的坐标点
#             coords = list(road.geometry.coords)
#
#             # 处理该道路的所有点
#             road_nodes = []
#             for coord in coords:
#                 # 如果该坐标还没有被处理过
#                 if coord not in node_coordinates:
#                     node_coordinates.add(coord)
#                     # 创建新节点
#                     node = {
#                         'node_id': node_id,
#                         'geometry': Point(coord),
#                         'longitude': coord[0],
#                         'latitude': coord[1]
#                     }
#                     nodes.append(node)
#                     road_nodes.append(node_id)
#                     node_id += 1
#                 else:
#                     # 找到已存在节点的ID
#                     for n in nodes:
#                         if (n['geometry'].x, n['geometry'].y) == coord:
#                             road_nodes.append(n['node_id'])
#                             break
#
#             # 创建边
#             for i in range(len(road_nodes) - 1):
#                 edge = {
#                     'edge_id': edge_id,
#                     'from_node': road_nodes[i],
#                     'to_node': road_nodes[i + 1],
#                     'geometry': LineString([coords[i], coords[i + 1]]),
#                     'length': Point(coords[i]).distance(Point(coords[i + 1])),
#                     'road_type': road['fclass'] if 'fclass' in road else None,
#                     'name': road['name'] if 'name' in road else None
#                 }
#                 edges.append(edge)
#                 edge_id += 1
#
#     print(f"发现 {len(nodes)} 个节点和 {len(edges)} 条边")
#
#     # 创建GeoDataFrames
#     print("创建节点GeoDataFrame...")
#     nodes_gdf = gpd.GeoDataFrame(nodes)
#     nodes_gdf.set_geometry('geometry', inplace=True)
#     nodes_gdf.crs = roads.crs  # 设置坐标参考系统
#
#     print("创建边GeoDataFrame...")
#     edges_gdf = gpd.GeoDataFrame(edges)
#     edges_gdf.set_geometry('geometry', inplace=True)
#     edges_gdf.crs = roads.crs  # 设置坐标参考系统
#
#     # 保存为形状文件
#     print("保存节点形状文件...")
#     nodes_gdf.to_file(f"{output_nodes_path}.shp")
#
#     print("保存边形状文件...")
#     edges_gdf.to_file(f"{output_edges_path}.shp")
#
#     print("处理完成！")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 设置输入输出路径
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#
#     # 输入文件路径
#     input_path = os.path.join(current_dir, "rawRoadnet", "gis_osm_roads_free_1")
#
#     # 输出文件路径
#     output_dir = os.path.join(current_dir, "re", "processedRoadnet")  # 创建新的输出目录
#     output_nodes_path = os.path.join(output_dir, "nodes")
#     output_edges_path = os.path.join(output_dir, "edges")
#
#
#
#     split_road_network(input_path, output_nodes_path, output_edges_path)
#
