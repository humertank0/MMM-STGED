import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time


def process_road_chunk(args):
    """处理道路块的辅助函数"""
    chunk, node_dict, start_node_id, start_edge_id = args
    local_nodes = []
    local_edges = []
    current_node_id = start_node_id
    current_edge_id = start_edge_id

    for _, road in chunk.iterrows():
        if isinstance(road.geometry, LineString):
            coords = list(road.geometry.coords)
            road_nodes = []

            for coord in coords:
                coord_key = (coord[0], coord[1])
                if coord_key not in node_dict:
                    node = {
                        'node_id': current_node_id,
                        'geometry': Point(coord),
                        'longitude': coord[0],
                        'latitude': coord[1]
                    }
                    local_nodes.append(node)
                    node_dict[coord_key] = current_node_id
                    road_nodes.append(current_node_id)
                    current_node_id += 1
                else:
                    road_nodes.append(node_dict[coord_key])

            for i in range(len(road_nodes) - 1):
                edge = {
                    'edge_id': current_edge_id,
                    'from_node': road_nodes[i],
                    'to_node': road_nodes[i + 1],
                    'geometry': LineString([coords[i], coords[i + 1]]),
                    'length': Point(coords[i]).distance(Point(coords[i + 1])),
                    'road_type': road['fclass'] if 'fclass' in road else None,
                    'name': road['name'] if 'name' in road else None
                }
                local_edges.append(edge)
                current_edge_id += 1

    return local_nodes, local_edges


def split_road_network(input_path, output_nodes_path, output_edges_path):
    """将道路网络数据拆分为节点和边"""
    start_time = time.time()

    # 读取道路网络数据
    print("读取道路网络数据...")
    roads = gpd.read_file(f"{input_path}.shp")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_nodes_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_edges_path), exist_ok=True)

    # 使用字典存储节点坐标到节点ID的映射
    node_dict = {}

    # 将数据分成多个块以进行并行处理
    n_chunks = os.cpu_count()
    chunk_size = len(roads) // n_chunks + 1
    road_chunks = [roads[i:i + chunk_size] for i in range(0, len(roads), chunk_size)]

    # 准备并行处理的参数
    chunk_args = [(chunk, node_dict, i * chunk_size * 1000, i * chunk_size * 1000)
                  for i, chunk in enumerate(road_chunks)]

    # 并行处理道路数据
    print("并行处理道路数据...")
    all_nodes = []
    all_edges = []

    with ThreadPoolExecutor(max_workers=n_chunks) as executor:
        results = list(executor.map(process_road_chunk, chunk_args))

        for nodes, edges in results:
            all_nodes.extend(nodes)
            all_edges.extend(edges)

    print(f"发现 {len(all_nodes)} 个节点和 {len(all_edges)} 条边")

    # 使用pandas DataFrame来提高性能
    print("创建GeoDataFrames...")
    nodes_gdf = gpd.GeoDataFrame(all_nodes)
    nodes_gdf.set_geometry('geometry', inplace=True)
    nodes_gdf.crs = roads.crs

    edges_gdf = gpd.GeoDataFrame(all_edges)
    edges_gdf.set_geometry('geometry', inplace=True)
    edges_gdf.crs = roads.crs

    # 保存文件
    print("保存文件...")
    nodes_gdf.to_file(f"{output_nodes_path}.shp")
    edges_gdf.to_file(f"{output_edges_path}.shp")

    end_time = time.time()
    print(f"处理完成！总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "rawRoadnet", "gis_osm_roads_free_1")
    output_dir = os.path.join(current_dir,"processedRoadnet")
    output_nodes_path = os.path.join(output_dir, "nodes")
    output_edges_path = os.path.join(output_dir, "edges")

    split_road_network(input_path, output_nodes_path, output_edges_path)
