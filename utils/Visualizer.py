"""
量子网络可视化工具
基于 NetworkX 原生绘图功能
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取 utils 路径
project_root = os.path.dirname(current_dir)              # 获取 QuantumRouting 路径
if project_root not in sys.path:
    sys.path.append(project_root)
# ==========================================

import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
from core.Network import QuantumNetworkManager
from utils.Constants import RES_CONFIG


def visualize_network(network_manager: QuantumNetworkManager,
                      save_path: Optional[str] = None,
                      figsize=(10, 8)):
    """
    使用 NetworkX 原生函数可视化量子网络
    """
    graph = network_manager.graph

    # 获取节点位置
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos:
        # 如果没有位置信息，使用弹簧布局作为备选
        pos = nx.spring_layout(graph, seed=42)

    # 分类节点
    high_nodes = []
    low_nodes = []

    # 安全获取配置阈值
    high_rate_threshold = RES_CONFIG.get("HIGH", {}).get("rate", 100)

    for node_id in graph.nodes():
        node = network_manager.get_node_object(node_id)
        if node and node.max_rate >= high_rate_threshold:
            high_nodes.append(node_id)
        else:
            low_nodes.append(node_id)

    plt.figure(figsize=figsize)

    # 绘制核心节点 (红色)
    if high_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=high_nodes,
                             node_color='#FF6B6B', node_size=400, # 稍微调好看了点颜色
                             label=f'Core (Repeater)')

    # 绘制边缘节点 (蓝色)
    if low_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=low_nodes,
                             node_color='#4ECDC4', node_size=200,
                             label=f'Edge (EndNode)')

    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3, width=1, style='dashed')

    # 绘制标签
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')

    # 设置标题和图例
    plt.title(f'Quantum Network Topology\n{graph.number_of_nodes()} Nodes, {graph.number_of_edges()} Links')
    plt.legend()
    plt.axis('off') # 关闭坐标轴显示，更美观
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualizer] Network topology saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_path(network_manager: QuantumNetworkManager,
                   path: list,
                   save_path: Optional[str] = None,
                   figsize=(10, 8)):
    """
    可视化路径
    """
    graph = network_manager.graph
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos:
        pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=figsize)

    # 1. 绘制背景网络 (淡化)
    nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=100, alpha=0.3)
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', alpha=0.2)

    # 2. 绘制路径节点 (高亮)
    nx.draw_networkx_nodes(graph, pos, nodelist=path,
                         node_color='red', node_size=300, alpha=0.9, label='Path Nodes')

    # 3. 绘制路径边 (高亮箭头)
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                         edge_color='red', width=2.5, alpha=0.9)

    # 绘制标签
    nx.draw_networkx_labels(graph, pos, font_size=8)

    plt.title(f'Routing Path: {" -> ".join(map(str, path))}')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualizer] Path visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()
