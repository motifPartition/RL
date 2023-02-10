
import argparse
from pathlib import Path

import networkx as nx
import nxmetis
from networkx import conductance

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, GATConv, GlobalAttention, graclus, avg_pool, global_mean_pool
from torch_geometric.utils import to_networkx, k_hop_subgraph, degree

import snap
import subprocess
import random as random_
import sys
import pandas as pd

import numpy as np
from numpy import random

import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy.spatial import Delaunay

#import random_p
import copy
import math
import timeit
import os
from itertools import combinations


def graph_delaunay_from_points(points):
    mesh = Delaunay(points, qhull_options="QJ")
    mesh_simp = mesh.simplices

    edges = []
    for i in range(len(mesh_simp)):
        edges += combinations(mesh_simp[i], 2)
	
    e = list(set(edges))
    return nx.Graph(e)

# Pytorch geometric Delaunay mesh with n random points in the unit square


def random_delaunay_graph(n):
	#Get n points that have random x and y
    points = np.random.random_sample((n, 2))

    g = graph_delaunay_from_points(points)
    adj_sparse = nx.to_scipy_sparse_matrix(g, format='coo')

    row = adj_sparse.row
    col = adj_sparse.col

    one_hot = []
    for i in range(g.number_of_nodes()):
        one_hot.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(one_hot), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)
    
    return graph_torch

# Build a pytorch geometric graph with features [1,0] form a networkx graph
    
def torch_from_graph(g):

    adj_sparse = nx.to_scipy_sparse_matrix(g, format='coo')
    row = adj_sparse.row
    col = adj_sparse.col

    one_hot = []
    for i in range(g.number_of_nodes()):
        one_hot.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(one_hot), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    degs = np.sum(adj_sparse.todense(), axis=0)
    first_vertices = np.where(degs == np.min(degs))[0]
    first_vertex = np.random.choice(first_vertices)
    change_vertex(graph_torch, first_vertex)

    return graph_torch

# Training dataset made of Delaunay graphs generated from random points in
# the unit square and their coarser graphs

def delaunay_dataset_with_coarser(n, n_min, n_max):
    dataset = []

    while len(dataset) < n:
        number_nodes = np.random.choice(np.arange(n_min, n_max + 1, 2))
        #print('number_nodes : ',number_nodes)

        g = random_delaunay_graph(number_nodes)
        dataset.append(g)
        while g.num_nodes > 200:
            cluster = graclus(g.edge_index)
            coarse_graph = avg_pool(
                cluster,
                Batch(
                    batch=torch.zeros(
                        g.num_nodes),
                    x=g.x,
                    edge_index=g.edge_index))
            g1 = Data(x=coarse_graph.x, edge_index=coarse_graph.edge_index)
            dataset.append(g1)
            g = g1

    for i,g in enumerate(dataset):
        torch.save(g, f'./temp_edge/{i}')

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    return loader

# Build a pytorch geometric graph with features [1,0] form a sparse matrix


def torch_from_sparse(adj_sparse):

    row = adj_sparse.row
    col = adj_sparse.col

    features = []
    for i in range(adj_sparse.shape[0]):
        features.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(features), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    return graph_torch

# Training dataset made of SuiteSparse graphs and their coarser graphs


def suitesparse_dataset_with_coarser(n, n_min, n_max):
    dataset, picked = [], []
    for graph in os.listdir(os.path.expanduser(
            'drl-graph-partitioning/suitesparse_train/')):

        if len(dataset) > n or len(picked) >= len(os.listdir(os.path.expanduser(
                'drl-graph-partitioning/suitesparse_train/'))):
            break
        picked.append(str(graph))
        # print(str(graph))
        matrix_sparse = mmread(
            os.path.expanduser(
                'drl-graph-partitioning/suitesparse_train/' +
                str(graph)))
        gnx = nx.from_scipy_sparse_matrix(matrix_sparse)
        if nx.number_connected_components(gnx) == 1 and gnx.number_of_nodes(
        ) > n_min and gnx.number_of_nodes() < n_max:
            g = torch_from_sparse(matrix_sparse)
            g.weight = torch.tensor([1] * g.num_edges)
            dataset.append(g)
            while g.num_nodes > 200:
                cluster = graclus(g.edge_index)
                coarse_graph = avg_pool(
                    cluster,
                    Batch(
                        batch=torch.zeros(
                            g.num_nodes),
                        x=g.x,
                        edge_index=g.edge_index))
                g1 = Data(x=coarse_graph.x, edge_index=coarse_graph.edge_index)
                dataset.append(g1)
                g = g1

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader

# Cut of the input graph


def cut(graph):
    cut = torch.sum((graph.x[graph.edge_index[0],
                             :2] != graph.x[graph.edge_index[1],
                                            :2]).all(axis=-1)).detach().item() / 2
    return cut

# Change the feature of the selected vertex


def change_vertex(state, vertex):
    if (state.x[vertex, :2] == torch.tensor([1., 0.])).all():
        state.x[vertex, 0] = torch.tensor(0.)
        state.x[vertex, 1] = torch.tensor(1.)
    else:
        state.x[vertex, 0] = torch.tensor(1.)
        state.x[vertex, 1] = torch.tensor(0.)

    return state

# Reward to train the DRL agent


def reward_NC_2(state, vertex):
    new_state = state.clone()
    new_state = change_vertex(new_state, vertex)
    return normalized_cut_2(state) - normalized_cut_2(new_state)

# Normalized cut of the input graph


def normalized_cut(graph):
    cut, da, db = volumes(graph)
    if da == 0 or db == 0:
        return 2
    else:
        return cut * (1 / da + 1 / db)
	
def normalized_cut_2(graph):
    cut, da, db, count_a, count_b = volumes_2(graph)
    if count_b == 0 or count_a == 0:
        return 2
    else:
        return cut * (1 / count_a + 1 / count_b)

def normalized_cut_3(T, cut, da, db, count_a, count_b):
    if count_b < T or count_b < T:
        return -10
    else:
        return cut * (1 / count_a + 1 / count_b)

# Coarsen a pytorch geometric graph, then find the cut with METIS and
# interpolate it back


def partition_metis_refine(graph):
    cluster = graclus(graph.edge_index)

    coarse_graph = avg_pool(
        cluster,
        Batch(
            batch=graph.batch,
            x=graph.x,
            edge_index=graph.edge_index))

    coarse_graph_nx = to_networkx(coarse_graph, to_undirected=True)
    _, parts = nxmetis.partition(coarse_graph_nx, 2)
    mparts = np.array(parts)

    coarse_graph.x[np.array(parts[0])] = torch.tensor([1., 0.])
    coarse_graph.x[np.array(parts[1])] = torch.tensor([0., 1.])
    _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
    graph.x = coarse_graph.x[inverse]
    return graph

def partition_motif_refine(graph, motif="clique3"):
    
	file_1 = open('graph.txt','w')

	for node1, node2 in zip(graph.edge_index[0].numpy(), graph.edge_index[1].numpy()):
		file_1.write(f'{int(node1)} {int(node2)}\n')

	file_1.close()

	seed_node = random_.sample( range(graph.x.size(0)) ,1 )[0]

	out = subprocess.Popen(["../../Snap-6.0/examples/localmotifcluster/localmotifclustermain", "-d:N", "-i:./graph.txt", f"-m:{motif}", f"-s:{seed_node}"], stdout=subprocess.PIPE)
	out, _ = out.communicate()
	out = out.decode("utf-8") 
	out = out.split('\n')
	out = out[9:-4]

	parts = [[],[]]
	part = 0

	for node in out:
		if 'Global' in node or 'local' in node:
			part = 1
			continue
		else:
			node_local = int(node.split('\t')[1])
			parts[part].append(node_local)

	parts.append( list( set(graph.edge_index[0].numpy()) - set(parts[0]+parts[1]) ) )

	graph.x[np.array(parts[0])] = torch.tensor([1., 0.])
	graph.x[np.array(parts[1])] = torch.tensor([0., 1.])
	graph.x[np.array(parts[2])] = torch.tensor([0., 0.])

	return graph

def partition_motif_refine_2(graph, motif="clique3"):
    
    file_1 = open('graph_.txt','w')

    for node1, node2 in zip(graph.edge_index[0].numpy(), graph.edge_index[1].numpy()):
        file_1.write(f'{int(node1)} {int(node2)}\n')

    file_1.close()

    part_3_bis = [0]
    stop = False
    seed_boolean = -1

    while len(part_3_bis) != 0 and not stop:

        if seed_boolean == -1:
            seed_node = random_.sample( range(graph.x.size(0)) ,1 )[0]
        else:
            if seed_boolean == len(part_3_bis):
                seed_boolean = random_.sample( range(seed_boolean) ,1 )[0]
                seed_node = part_3_bis[seed_boolean]
                stop = True
            else:
                seed_node = part_3_bis[seed_boolean]
                seed_boolean += 1

        out = subprocess.Popen(["../../Snap-6.0/examples/localmotifcluster/localmotifclustermain", "-d:N", "-i:./graph_.txt", f"-m:{motif}", f"-s:{seed_node}"], stdout=subprocess.PIPE)
        out, _ = out.communicate()
        out = out.decode("utf-8") 
        out = out.split('\n')
        out = out[9:-4]

        parts = [[],[]]
        part = 0

        for node in out:
            if 'Global' in node or 'local' in node:
                part = 1
                continue
            else:
                node_local = int(node.split('\t')[1])
                parts[part].append(node_local)

        part_3 = list( set(graph.edge_index[0].numpy()) - set(parts[0]+parts[1]) )

        if seed_boolean == -1:
            part_3_bis = list(part_3)
            seed_boolean = 0

    parts.append( part_3 )

    graph.x[np.array(parts[0])] = torch.tensor([1., 0.])
    graph.x[np.array(parts[1])] = torch.tensor([0., 1.])
    graph.x[np.array(parts[2])] = torch.tensor([0., 0.])

    return graph


# Subgraph around the cut


def k_hop_graph_cut(graph, k, va, vb, count_a, count_b):
	nei = torch.where((graph.x[graph.edge_index[0], :2] !=
					graph.x[graph.edge_index[1], :2]).all(axis=-1))[0]

	neib = graph.edge_index[0][nei]

	data_cut = k_hop_subgraph(neib, k, graph.edge_index, relabel_nodes=True)
	data_small = k_hop_subgraph(neib, k - 1, graph.edge_index, relabel_nodes=True)

	data_removed = list(torch.where( (graph.x == torch.tensor([0,0])).all(axis=-1) )[0].numpy())

	nodes_boundary = list(set(data_cut[0].numpy()).difference(data_small[0].numpy()))
	nodes_boundary += data_removed
	nodes_boundary = list(set(nodes_boundary))

	boundary_features = torch.tensor([1. if i.item() in nodes_boundary else 0. for i in data_cut[0]]).reshape(data_cut[0].shape[0], 1)

	#_, va, vb, count_a, count_b = volumes_2(graph)
	e = torch.ones(data_cut[0].shape[0], 1)
	nnz = graph.num_edges

	subprocess.call(["../../Snap-6.0/examples/motifs/motifs", "-i:./graph.txt", f"-m:{3}", f"-o:all"])
	df = pd.read_csv('all-counts.tab',sep='\t')
	counts = df['Count'].sum()


	features = torch.cat((graph.x[data_cut[0]], boundary_features, torch.true_divide(
		va, nnz) * e, torch.true_divide(vb, nnz) * e, torch.true_divide(count_a, counts) * e, torch.true_divide(count_b, counts) * e), 1)

	g_red = Batch(
		batch=torch.zeros(
			data_cut[0].shape[0],
			dtype=torch.long),
		x=features,
		edge_index=data_cut[1])

	return g_red, data_cut[0]

# Volumes of the partitions


def volumes(graph):
    ia = torch.where(
        (graph.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]
    ib = torch.where(
        (graph.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]
    degs = degree(
        graph.edge_index[0],
        num_nodes=graph.x.size(0),
        dtype=torch.uint8)
    da = torch.sum(degs[ia]).detach().item()
    db = torch.sum(degs[ib]).detach().item()


    cut = torch.sum((graph.x[graph.edge_index[0],
                             :2] != graph.x[graph.edge_index[1],
                                            :2]).all(axis=-1)).detach().item() / 2
    return cut, da, db

def volumes_2(graph):
	ia = torch.where(
		(graph.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]

	ib = torch.where(
		(graph.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]

	degs = degree(
		graph.edge_index[0],
		num_nodes=graph.x.size(0),
		dtype=torch.uint8)

	da = torch.sum(degs[ia]).detach().item()
	db = torch.sum(degs[ib]).detach().item()

	#Build ia and ib in a file
	old2new_1 = {}
	for idx, i in enumerate(ia.numpy()):
		old2new_1[i] = idx

	old2new_2 = {}
	for idx, i in enumerate(ib.numpy()):
		old2new_2[i] = idx

	file_1 = open('graph_a.txt','w')
	file_2 = open('graph_b.txt','w')

	for i,j in zip(graph.edge_index[0].numpy(), graph.edge_index[1].numpy()):
		if i in old2new_1 and j in old2new_1:
			file_1.write(f'{int(old2new_1[i])} {int(old2new_1[j])}\n')
		elif i in old2new_2 and j in old2new_2:
			file_2.write(f'{int(old2new_2[i])} {int(old2new_2[j])}\n')

	file_1.close()
	file_2.close()	

	subprocess.call(["../../Snap-6.0/examples/motifs/motifs", "-i:./graph_a.txt", f"-m:{3}", f"-o:ia"])
	ia_df = pd.read_csv('ia-counts.tab',sep='\t')
	counts_ia = ia_df['Count'].sum()

	subprocess.call(["../../Snap-6.0/examples/motifs/motifs", "-i:./graph_b.txt", f"-m:{3}", f"-o:ib"])
	ib_df = pd.read_csv('ib-counts.tab',sep='\t')
	counts_ib = ib_df['Count'].sum()

	cut = torch.sum((graph.x[graph.edge_index[0],
								:2] != graph.x[graph.edge_index[1],
											:2]).all(axis=-1)).detach().item() / 2
	return cut, da, db, counts_ia, counts_ib


def eval_basic(ac, graph, k):
    g = graph.clone()
    g = partition_motif_refine(g)
    _, volA, volB, count_a, count_b = volumes_2(g)

    g, counts = ac_eval_refine_basic(ac, g, k, volA, volB, count_a, count_b)

    return g, counts

def ac_eval_refine_basic(ac, graph_t, k, volA, volB, count_a, count_b):
    graph = graph_t.clone()
    g0 = graph_t.clone()
    data = k_hop_graph_cut(graph, k, volA, volB, count_a, count_b)
    graph_cut, positions = data[0], data[1]

    subprocess.call(["../../Snap-6.0/examples/motifs/motifs", "-i:./graph.txt", f"-m:{3}", f"-o:all"])
    df = pd.read_csv('all-counts.tab',sep='\t')
    counts = df['Count'].sum()

    len_episod = int(cut(graph))

    peak_reward = 0
    peak_time = 0
    total_reward = 0
    actions = []

    e = torch.ones(graph_cut.num_nodes, 1)
    nnz = graph.num_edges
    cut_sub = len_episod

    for i in range(len_episod):

        with torch.no_grad():
            policy = ac(graph_cut)

        probs = policy.view(-1).clone().detach().numpy()
        flip = np.argmax(probs)

        old_nc = cut_sub * (torch.true_divide(1, count_a) +
                            torch.true_divide(1, count_b))

        new_graph_cut = graph_cut.clone()
        new_graph_cut = change_vertex(new_graph_cut, flip)
        _, volA, volB, count_a, count_b = volumes_2(new_graph_cut)

        new_graph = graph.clone()
        new_graph = change_vertex(new_graph, positions[flip])
        cut_sub = int(cut(new_graph))

        new_nc = cut_sub * (torch.true_divide(1, count_a) +
                            torch.true_divide(1, count_b))

        total_reward += (old_nc - new_nc).item()

        actions.append(flip)

        graph_cut = new_graph_cut
        graph = new_graph

        graph_cut.x[:, 3] = torch.true_divide(volA, nnz)
        graph_cut.x[:, 4] = torch.true_divide(volB, nnz)
        graph_cut.x[:, 5] = torch.true_divide(count_a, counts)
        graph_cut.x[:, 6] = torch.true_divide(count_b, counts)

        if i >= 1 and actions[-1] == actions[-2]:
            break
        if total_reward > peak_reward:
            peak_reward = total_reward
            peak_time = i + 1

    for t in range(peak_time):
        g0 = change_vertex(g0, positions[actions[t]])

    return g0, counts

def eval_different_partition(ac, graph, k):
    g = graph.clone()
    g = partition_motif_refine_2(g)
    _, volA, volB, count_a, count_b = volumes_2(g)

    g, counts = ac_eval_refine_basic(ac, g, k, volA, volB, count_a, count_b)

    return g, counts

def eval_different_reward(ac, graph, k, T):
    g = graph.clone()
    g = partition_motif_refine(g)
    _, volA, volB, count_a, count_b = volumes_2(g)

    g, counts = ac_eval_refine_basic_2(ac, g, k, volA, volB, count_a, count_b, T)

    return g, counts

def ac_eval_refine_basic_2(ac, graph_t, k, volA, volB, count_a, count_b, T):
    graph = graph_t.clone()
    g0 = graph_t.clone()
    data = k_hop_graph_cut(graph, k, volA, volB, count_a, count_b)
    graph_cut, positions = data[0], data[1]

    subprocess.call(["../../Snap-6.0/examples/motifs/motifs", "-i:./graph.txt", f"-m:{3}", f"-o:all"])
    df = pd.read_csv('all-counts.tab',sep='\t')
    counts = df['Count'].sum()

    len_episod = int(cut(graph))

    peak_reward = 0
    peak_time = 0
    total_reward = 0
    actions = []

    e = torch.ones(graph_cut.num_nodes, 1)
    nnz = graph.num_edges
    cut_sub = len_episod

    for i in range(len_episod):

        with torch.no_grad():
            policy = ac(graph_cut)

        probs = policy.view(-1).clone().detach().numpy()
        flip = np.argmax(probs)

        old_nc = normalized_cut_3(T, cut_sub, volA, volB, count_a, count_b)

        new_graph_cut = graph_cut.clone()
        new_graph_cut = change_vertex(new_graph_cut, flip)
        _, volA, volB, count_a, count_b = volumes_2(new_graph_cut)

        new_graph = graph.clone()
        new_graph = change_vertex(new_graph, positions[flip])
        cut_sub = int(cut(new_graph))

        new_nc = normalized_cut_3(T, cut_sub, volA, volB, count_a, count_b)
        
        if old_nc < 0:
            total_reward += (old_nc).item()
        elif new_nc < 0:
            total_reward += (new_nc).item()
        else:
            total_reward += (old_nc - new_nc).item()

        actions.append(flip)

        graph_cut = new_graph_cut
        graph = new_graph

        graph_cut.x[:, 3] = torch.true_divide(volA, nnz)
        graph_cut.x[:, 4] = torch.true_divide(volB, nnz)
        graph_cut.x[:, 5] = torch.true_divide(count_a, counts)
        graph_cut.x[:, 6] = torch.true_divide(count_b, counts)

        if i >= 1 and actions[-1] == actions[-2]:
            break
        if total_reward > peak_reward:
            peak_reward = total_reward
            peak_time = i + 1

    for t in range(peak_time):
        g0 = change_vertex(g0, positions[actions[t]])

    return g0, counts


def partition_metis(graph, graph_nx):
    obj, parts = nxmetis.partition(graph_nx, 2)
    mparts = np.array(parts)
    graph.x[parts[0]] = torch.tensor([1., 0.])
    graph.x[parts[1]] = torch.tensor([0., 1.])

    return graph

# Refining the cut on the subgraph around the cut


def ac_eval_refine(ac, graph_t, k, gnx, volA, volB):
    graph = graph_t.clone()
    g0 = graph_t.clone()
    data = k_hop_graph_cut(graph, k, gnx, volA, volB)
    graph_cut, positions = data[0], data[1]

    len_episod = int(cut(graph))

    peak_reward = 0
    peak_time = 0
    total_reward = 0
    actions = []

    e = torch.ones(graph_cut.num_nodes, 1)
    nnz = graph.num_edges
    cut_sub = len_episod
    for i in range(len_episod):
        with torch.no_grad():
            policy = ac(graph_cut)
        probs = policy.view(-1).clone().detach().numpy()
        flip = np.argmax(probs)

        dv = gnx.degree[positions[flip].item()]
        old_nc = cut_sub * (torch.true_divide(1, volA) +
                            torch.true_divide(1, volB))
        if graph_cut.x[flip, 0] == 1.:
            volA = volA - dv
            volB = volB + dv
        else:
            volA = volA + dv
            volB = volB - dv
        new_nc, cut_sub = update_nc(
            graph, gnx, cut_sub, positions[flip].item(), volA, volB)
        total_reward += (old_nc - new_nc).item()

        actions.append(flip)

        change_vertex(graph_cut, flip)
        change_vertex(graph, positions[flip])

        graph_cut.x[:, 3] = torch.true_divide(volA, nnz)
        graph_cut.x[:, 4] = torch.true_divide(volB, nnz)

        if i >= 1 and actions[-1] == actions[-2]:
            break
        if total_reward > peak_reward:
            peak_reward = total_reward
            peak_time = i + 1

    for t in range(peak_time):
        g0 = change_vertex(g0, positions[actions[t]])

    return g0

# Compute the update for the normalized cut


def update_nc(graph, gnx, cut_total, v1, va, vb):
    c_v1 = 0
    for v in gnx[v1]:
        if graph.x[v, 0] != graph.x[v1, 0]:
            c_v1 += 1
        else:
            c_v1 -= 1
    cut_new = cut_total - c_v1
    return cut_new * (torch.true_divide(1, va) +
                      torch.true_divide(1, vb)), cut_new

# Evaluation of the DRL model on the coarsest graph


def ac_eval(ac, graph, perc):
    graph_test = graph.clone()
    error_bal = math.ceil(graph_test.num_nodes * perc)
    cuts = []
    nodes = []
    # Run the episod
    for i in range(int(graph_test.num_nodes / 2 - 1 + error_bal)):
        policy, _ = ac(graph_test)
        policy = policy.view(-1).detach().numpy()
        flip = random.choice(torch.arange(0, graph_test.num_nodes), p=policy)
        graph_test = change_vertex(graph_test, flip)
        if i >= int(graph_test.num_nodes / 2 - 1 - error_bal):
            cuts.append(cut(graph_test))
            nodes.append(flip)
    if len(cuts) > 0:
        stops = np.argwhere(cuts == np.min(cuts))
        stops = stops.reshape((stops.shape[0],))
        if len(stops) == 1:
            graph_test.x[nodes[stops[0] + 1:]] = torch.tensor([1., 0.])
        else:
            diff = [np.abs(i - int(len(stops) / 2 - 1)) for i in stops]
            min_dist = np.argwhere(diff == np.min(diff))
            min_dist = min_dist.reshape((min_dist.shape[0],))
            stop = np.random.choice(stops[min_dist])
            graph_test.x[nodes[stop + 1:]] = torch.tensor([1., 0.])

    return graph_test

# Partitioning provided by SCOTCH


# Deep neural network for the DRL agent


class Model(torch.nn.Module):
	def __init__(self, units):
		super(Model, self).__init__()

		self.units = units
		self.common_layers = 1
		self.critic_layers = 1
		self.actor_layers = 1
		self.activation = torch.tanh

		self.conv_first = SAGEConv(7, self.units)
		self.conv_common = nn.ModuleList(
		    [SAGEConv(self.units, self.units)
		     for i in range(self.common_layers)]
		)
		self.conv_actor = nn.ModuleList(
		    [SAGEConv(self.units,
		              1 if i == self.actor_layers - 1 else self.units)
		     for i in range(self.actor_layers)]
		)
		self.conv_critic = nn.ModuleList(
		    [SAGEConv(self.units, self.units)
		     for i in range(self.critic_layers)]
		)
		self.final_critic = nn.Linear(self.units, 1)

	def forward(self, graph):
		x, edge_index, batch = graph.x, graph.edge_index, graph.batch

		do_not_flip = torch.where(x[:, 2] != 0.)
		x = self.activation(self.conv_first(x, edge_index))
		for i in range(self.common_layers):
		    x = self.activation(self.conv_common[i](x, edge_index))

		x_actor = x
		for i in range(self.actor_layers):
		    x_actor = self.conv_actor[i](x_actor, edge_index)
		    if i < self.actor_layers - 1:
		        x_actor = self.activation(x_actor)
		x_actor[do_not_flip] = torch.tensor(-np.Inf)
		x_actor = torch.log_softmax(x_actor, dim=0)
		

		if not self.training:
		    return x_actor

		x_critic = x.detach()
		for i in range(self.critic_layers):
		    x_critic = self.conv_critic[i](x_critic, edge_index)
		    if i < self.critic_layers - 1:
		        x_critic = self.activation(x_critic)
		x_critic = self.final_critic(x_critic)
		x_critic = torch.tanh(global_mean_pool(x_critic, batch))

		return x_actor



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', default='./temp_edge/', type=str)
    parser.add_argument(
        "--nmin",
        default=200,
        help="Minimum graph size",
        type=int)
    parser.add_argument(
        "--nmax",
        default=500,
        help="Maximum graph size",
        type=int)
    parser.add_argument(
        "--ntest",
        default=100,
        help="Number of test graphs",
        type=int)
    parser.add_argument("--hops", default=3, help="Number of hops", type=int)
    parser.add_argument(
        "--units",
        default=5,
        help="Number of units in conv layers",
        type=int)
    parser.add_argument(
        "--units_conv",
        default=[
            30,
            30,
            30,
            30],
        help="Number of units in conv layers",
        nargs='+',
        type=int)
    parser.add_argument(
        "--units_dense",
        default=[
            30,
            30,
            20],
        help="Number of units in linear layers",
        nargs='+',
        type=int)
    parser.add_argument(
        "--attempts",
        default=3,
        help="Number of attempts in the DRL",
        type=int)
    parser.add_argument(
        "--dataset",
        default='delaunay',
        help="Dataset type: delaunay, suitesparse, graded l, hole3, hole6",
        type=str)

    torch.manual_seed(1)
    np.random.seed(2)

    args = parser.parse_args()
    outdir = args.out + '/'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    n_min = args.nmin
    n_max = args.nmax
    n_test = args.ntest
    hops = args.hops
    units = args.units
    trials = args.attempts
    hid_conv = args.units_conv
    hid_lin = args.units_dense
    dataset_type = args.dataset

    T = 200

    # Choose the model to load according to the parameter 'dataset_type'
    model_basic = Model(units)
    model_basic.load_state_dict(torch.load('../temp_edge/model_partitioning_delaunay_basic'))

    model_different_partition = Model(units)
    model_different_partition.load_state_dict(torch.load('../temp_edge/model_partitioning_delaunay_new_partition'))

    model_different_reward = Model(units)
    model_different_reward.load_state_dict(torch.load('../temp_edge/model_partitioning_delaunay_new_reward'))

    print('Models loaded\n')
    results = pd.DataFrame(columns=['method','volume_1','volume_2','num_motifs_1','num_motifs_2','conductance','cut','normalized_cut','num_motifs_all','v','num_nodes_1','num_nodes_2','num_nodes_all'])
    i = 0
    while i < n_test:
        # Choose the dataset type according to the parameter 'dataset_type'
        if dataset_type == 'delaunay':
            n_nodes = np.random.choice(np.arange(n_min, n_max))
            g_inter = random_delaunay_graph(n_nodes)
            g_inter.batch = torch.zeros(g_inter.num_nodes)
            i += 1
        print('Graph:', i, '  Vertices:', g_inter.num_nodes, '  Edges:', g_inter.num_edges)
        nnz = g_inter.num_edges

        g = g_inter.clone()
        #Basic
        graph_basic, counts = eval_basic(model_basic, g, hops)
        cdrl, va, vb, c_a, c_b = volumes_2(graph_basic)
        gnx = to_networkx(graph_basic, to_undirected=True)

        ia = torch.where(
		(graph_basic.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ib = torch.where(
            (graph_basic.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ia = set([i.item() for i in ia])
        ib = set([i.item() for i in ib])

        if len(ia) == 0:
            conduct = conductance(gnx,ib)
        elif len(ib) == 0:
            conduct = conductance(gnx,ia)
        else:
            conduct = conductance(gnx,ia,ib)

        if c_b == 0 or c_a == 0:
            norm =  2
        else:
            norm = cdrl * (1 / c_b + 1 / c_a)
        
        results.loc[len(results)] = ['model_basic',va, vb, c_a, c_b, conduct, cdrl, norm, counts, nnz, ia, ib, graph_basic.x.size(0)]

        g = g_inter.clone()
        #Different parition
        graph_different_partition, counts_2 = eval_different_partition(model_different_partition, g, hops)
        cdrl_2, va_2, vb_2, c_a_2, c_b_2 = volumes_2(graph_different_partition)
        gnx = to_networkx(graph_different_partition, to_undirected=True)

        ia = torch.where(
		(graph_different_partition.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ib = torch.where(
            (graph_different_partition.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ia = set([i.item() for i in ia])
        ib = set([i.item() for i in ib])

        
        if len(ia) == 0:
            conduct = conductance(gnx,ib)
        elif len(ib) == 0:
            conduct = conductance(gnx,ia)
        else:
            conduct = conductance(gnx,ia,ib)

        if c_b_2 == 0 or c_a_2 == 0:
            norm =  2
        else:
            norm = cdrl_2 * (1 / c_b_2 + 1 / c_a_2)
        
        results.loc[len(results)] = ['model_diff_partition',va_2, vb_2, c_a_2, c_b_2, conduct, cdrl_2, norm, counts, nnz, ia, ib, graph_different_partition.x.size(0)]

        g = g_inter.clone()
        #Different reward
        graph_different_reward, counts_3 = eval_different_reward(model_different_reward, g, hops, T)
        cdrl_3, va_3, vb_3, c_a_3, c_b_3 = volumes_2(graph_different_reward)
        gnx = to_networkx(graph_different_reward, to_undirected=True)

        ia = torch.where(
		(graph_different_reward.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ib = torch.where(
            (graph_different_reward.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ia = set([i.item() for i in ia])
        ib = set([i.item() for i in ib])

        if len(ia) == 0:
            conduct = conductance(gnx,ib)
        elif len(ib) == 0:
            conduct = conductance(gnx,ia)
        else:
            conduct = conductance(gnx,ia,ib)
 
        if c_b_3 == 0 or c_a_3 == 0:
            norm =  2
        else:
            norm = cdrl_3 * (1 / c_b_3 + 1 / c_a_3)
        
        results.loc[len(results)] = ['model_diff_reward',va_3, vb_3, c_a_3, c_b_3, conduct, cdrl_3, norm, counts, nnz, ia, ib, graph_different_reward.x.size(0)]

        g = g_inter.clone()
        # Original one
        graph_baseline = partition_motif_refine(g)
        cdrl_4, va_4, vb_4, c_a_4, c_b_4 = volumes_2(graph_baseline)
        gnx = to_networkx(graph_baseline, to_undirected=True)

        ia = torch.where(
		(graph_baseline.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ib = torch.where(
            (graph_baseline.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ia = set([i.item() for i in ia])
        ib = set([i.item() for i in ib])

        
        if len(ia) == 0:
            conduct = conductance(gnx,ib)
        elif len(ib) == 0:
            conduct = conductance(gnx,ia)
        else:
            conduct = conductance(gnx,ia,ib)

        if c_b_4 == 0 or c_a_4 == 0:
            norm =  2
        else:
            norm = cdrl_4 * (1 / c_b_4 + 1 / c_a_4)
        
        results.loc[len(results)] = ['local_partition',va_4, vb_4, c_a_4, c_b_4, conduct, cdrl_4, norm, counts, nnz, ia, ib, graph_baseline.x.size(0)]

        g = g_inter.clone()
        # Partitioning with METIS
        gnx = to_networkx(g, to_undirected=True)
        g_metis = partition_metis(g, gnx)

        cdrl_5, va_5, vb_5, c_a_5, c_b_5 = volumes_2(g_metis)
        
        ia = torch.where(
		(g_metis.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ib = torch.where(
            (g_metis.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]

        ia = set([i.item() for i in ia])
        ib = set([i.item() for i in ib])

        
        if len(ia) == 0:
            conduct = conductance(gnx,ib)
        elif len(ib) == 0:
            conduct = conductance(gnx,ia)
        else:
            conduct = conductance(gnx,ia,ib)

        if c_b_5 == 0 or c_a_5 == 0:
            norm =  2
        else:
            norm = cdrl_5 * (1 / c_b_5 + 1 / c_a_5)
        
        results.loc[len(results)] = ['metis',va_5, vb_5, c_a_5, c_b_5, conduct, cdrl_5, norm, counts, nnz, ia, ib, g_metis.x.size(0)]

    print('Done')
    results.to_csv('results_2.csv',index=False)