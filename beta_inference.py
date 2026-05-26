import sys
from config_ import cfg

import model_o_att_e
import utils_n_nself_b
import torch

import numpy as np

import time
from tqdm import *

cfg = cfg()
cfg.get_args()
cfgs = cfg.update_train_configs()

from seu_tkg import sinkhorn, cal_sims
from CSLS_test_4 import *
from CSLS_test_3r import *

import os.path as osp
from kneed import KneeLocator

spe_string = 'evolve_model_beta'

from pathlib import Path

folder_name = f"model/{spe_string}"
folder_path = Path.cwd() / folder_name

file_path = str(Path.cwd()) + '/dataset/'
filename = 'BETA_E/'

# Check and create folder
folder_path.mkdir(parents=True, exist_ok=True)
print(f"Folder ready at: {folder_path}")

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([])
    time_int = set([])
    time = set([])
    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            #head, r, tail, ts, te = [int(item) for item in para]
            head, r, tail = int(eval(para[0])), int(eval(para[1])), int(eval(para[2]))
            ts = para[3]
            te = para[4]
            if ts == "~":
                ts = 0
            else:
                ts = int(eval(ts))
            if te == '~':
                te = 0
            else:
                te = int(eval(te))
            # if ts != 0 and te == 0:
            #     te = ts
            # if te != 0 and ts == 0:
            #     ts = te
            t_int = (ts, te)
            entity.add(head)
            entity.add(tail)
            rel.add(r)
            time_int.add(t_int)
            time.add(ts)
            time.add(te)
            triples.append([head, r, tail, ts, te])
    return entity, rel, triples, time_int, time

entity_1, rel_1, triples_1, time_int_1, time_1 =\
      load_triples('/home/jiayun/Desktop/BETA/data/BETA/triples_1')
entity_2, rel_2, triples_2, time_int_2, time_2 = \
load_triples('/home/jiayun/Desktop/BETA/data/BETA/triples_2')

def update_int_set(time_int):
    time_int_ = set()
    for i in time_int:
        time_int_.add(i)
        ts = i[0]
        te = i[1]
        if te == 0:
            time_int_.add((ts, ts))
        if ts == 0:
            time_int_.add((te, te))
    return time_int_

time_int = time_int_1 | time_int_2
time_int = update_int_set(time_int)
time_index_dic = dict(zip(list(time_int), np.arange(0, len(time_int), 1)))

int_index = time_index_dic[(0, 0)]

time_set = time_1 | time_2
print(len(time_set))

time_dic = dict(zip(np.arange(len(time_set)), time_set))
time_dic_i = dict(zip(time_set, np.arange(len(time_set))))

#here, we would like to loop from 10 to 20.
snap_shots = [f'{i}/'for i in range(10, 20)]

#get with the all_pair_dic.
def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair

sup_pairs = load_alignment_pair(file_path + 'BETA_E/sup_pairs')
ref_pairs = load_alignment_pair(file_path + 'BETA_E/ref_pairs')
all_pair = np.vstack((np.array(sup_pairs), np.array(ref_pairs)))
# all_pair = np.array(all_pair)
all_pair_dic = dict(zip(all_pair[:, 0], all_pair[:, 1]))
all_pair_dic_i = dict(zip(all_pair[:, 1], all_pair[:, 0]))

print(len(sup_pairs), len(ref_pairs), len(all_pair))

dev_pair_list = []
entity_list_e = []
entity_list_eh = []
score_list = []

def gen_dev(entity_set, all_pair_dic):
    rec = []
    for ent in entity_set:
        ent_c = all_pair_dic[ent]
        rec.append([ent, ent_c])
    return rec

def gen_dev_sub(dev_pair, entity_set):
    """generate a subset for the general part of the dev pair"""
    rec = []
    for pair in dev_pair:
        if pair[0] in entity_set:
            rec.append(pair)
    return np.array(rec)

def gen_ent_1(triples, ent_set):
    """generate entities with one hop"""
    rec = set()
    for triple in triples:
        head, tail = triple[0], triple[2]
        if head in ent_set:
            rec.add(head)
            rec.add(tail)
        if tail in ent_set:
            rec.add(head)
            rec.add(tail)
    return rec

def gen_top_seeds(score_s):
    """generate the top score seeds based on the bi-directional version"""
    rec = {}
    for i in tqdm(range(len(score_s))):
        max_index = np.argmax(score_s[i])
        #max_val = np.max(score_st[i])
        max_val = score_s[i][max_index].numpy()
        max_index_c = np.argmax(score_s[:, max_index])
    # max_val_c = np.max(score_st[:, max_index])
        max_val_c = score_s[:, max_index][max_index_c].numpy()
        if max_index_c == i:
            rec[i, max_index] = (max_val + max_val_c) / 2
    rec_sort = dict(sorted(rec.items(), key=lambda item: item[1], reverse=True))
    return rec_sort

def asymmetric_modest_score(s_A, s_B, mu=0.2, sigma=0.1):
    g_A = np.exp(-((s_B - mu) ** 2) / (2 * sigma ** 2))
    g_B = np.exp(-((s_A - mu) ** 2) / (2 * sigma ** 2))
    score = s_A * g_A + s_B * g_B
    return score.numpy().item()

def pair_sort(rec_sort, score_1, mu, sigma=0.15):
    """
    generate with the part rec sort and the general pair score
    score_1 is the inverse direction of the rec_sort.
    """
    rec_score = []
    rec_c = {}
    count = 0
    for pair, score in tqdm(rec_sort.items()):
        index, index_c = pair
        index_max = np.argmax(score_1[index])
        index_max_c = np.argmax(score_1[:, index_c])
        if index_max == index_max_c == index_c:
            count += 1
            continue
        else:
            score_c = score_1[index][index_c]
        as_score = asymmetric_modest_score(score, score_c, mu=mu, sigma=sigma)
        rec_c[pair] = as_score
        rec_score.append(as_score)
    rec_c = sorted(rec_c.items(), key=lambda item: item[1], reverse=True)
    return rec_c, rec_score

def check_acc(rec_sort, num):
    rec = []
    count = 0
    count_c = 0
    for pair, score_list in rec_sort:
        rec.append(pair)
        count += 1
        if pair[0] == pair[1]:
            count_c += 1
        if count == num:
            break 
    return rec, count_c / count

def gen_threshold(score, s=10):
    """generate the expected thereshold for the score distribution"""
    scores = np.array(score)  # e.g., output from your asymmetric_modest_score
    scores_sorted = np.sort(scores)[::-1]  # descending
    x = np.arange(len(scores_sorted))
    y = scores_sorted

    knee = KneeLocator(x, y, curve="convex", direction="decreasing", S=s)
    elbow_index = knee.knee
    threshold = scores_sorted[elbow_index]
    return elbow_index, threshold

def transfer_pair(rec_pair, pair_dic_yi, pair_dic_wi, all_pair_dic):
    """transfer the pair into the real index"""
    train_pair = []
    count = 0
    for pair in rec_pair:
        ent, ent_c = pair[0], pair[1]
        train_pair.append([pair_dic_yi[ent], pair_dic_wi[ent_c]])
        #check if correct.
        if all_pair_dic[pair_dic_yi[ent]] == pair_dic_wi[ent_c]:
            count += 1
    train_pair = np.array(train_pair)
    return train_pair, count / len(train_pair)

def gen_garph_stat(adj_matrix, rel_features, adj_features, time_features, time_int_features):
    """generate the corresponding info for the graph propagation"""
    adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
    rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
    time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data

    time_int_matrix, time_int_val = np.stack(time_int_features.nonzero(), axis=1), \
        time_int_features.data
    
    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    time_size = time_features.shape[1]
    time_int_size = time_int_features.shape[1]
    triple_size = len(adj_matrix)

    return adj_matrix, rel_matrix, rel_val, ent_matrix, ent_val, time_matrix, time_val, time_int_matrix,\
    time_int_val, node_size, rel_size, time_size, time_int_size, triple_size

def convert_rank(rank_list, dev_pair_dic):
    """convert the rank to the corresponding indices in the counterpart KG"""
    rec = []
    for list in rank_list:
        rec_list = []
        for item in list:
            rec_list.append(dev_pair_dic[item])
        rec.append(np.array(rec_list))
    return rec

def gen_rank_score(rank, score_m):
    rec = []
    for i, rank_list in enumerate(rank):
        score_s = score_m[i]
        rec_ = {}
        for j, ent in enumerate(rank_list):
            rec_[ent] = score_s[j]
        rec.append(rec_)
    return rec

def index_mod(all_index_wiki, wiki_index_i, top):
    rec = set()
    for index in all_index_wiki[0]:
        for i in range(top):
            rec.add(wiki_index_i[index[i].item()])
    return np.array(list(rec))

import faiss

def custom_top_k_gpu(x_s, x_t, batch_size=1000):
    """top-k correspondence computation for the efficient faiss framework
    for the comparison direction, we do from x_s to x_t, (ent_s, top_k)
    using the gpu to compute it"""
    res = faiss.StandardGpuResources() 
    #res.setTempMemory(500 * 1024 * 1024)
    index = faiss.IndexFlatIP(x_t.shape[-1])
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
    index_gpu.add(x_t)

    # Process queries in batches to handle memory efficiently
    batch_size = 2000  # Adjust based on your memory capacity
    all_distances = []
    all_indices = []

    for start in range(0, x_s.shape[0], batch_size):
        end = min(start + batch_size, x_s.shape[0])
        distances, indices = index_gpu.search(x_s[start:end], 10)
        all_distances.append(distances)
        all_indices.append(indices)

    all_distances = np.vstack(all_distances)
    all_indices = np.vstack(all_indices)
    all_indices = torch.LongTensor(all_indices)
    return all_indices.unsqueeze(0)

def check_p(entity_a, all_indices, index_i, top):
    count = 0
    for i, j in enumerate(all_indices[0]):
        for k in range(top):
            if entity_a[i] not in all_pair_dic:
                continue
            if all_pair_dic[entity_a[i]] == index_i[j[k].item()]:
                count += 1
                break
    return count / len(all_indices[0])

def check_pi(entity_a, all_indices, index_i, top):
    count = 0
    for i, j in enumerate(all_indices[0]):
        for k in range(top):
            if entity_a[i] not in all_pair_dic_i:
                continue
            if all_pair_dic_i[entity_a[i]] == index_i[j[k].item()]:
                count += 1
                break
    return count / len(all_indices[0])

def build_search_bi(entity_1an, output, dev_pair, iter, top, wiki_index_i, yago_index_i):
    """build the bi-direcitonal part of search and record the general part of the recall"""
    entity_start = entity_1an
    entity_start_wiki = None
    for i in range(iter):
        print(i)
        all_indices_wiki = custom_top_k_gpu(output[entity_start], output[dev_pair[:, 1]])
        #get the top performance. 
        print(check_p(entity_start, all_indices_wiki, wiki_index_i, top))
        #modify the index for wiki.
        wiki_real_index = index_mod(all_indices_wiki, wiki_index_i, top)
        if entity_start_wiki is not None:
            wiki_index_iter = set(list(wiki_real_index)) | set(list(entity_start_wiki))
            wiki_index_iter_n = np.array(list(wiki_index_iter))
            wiki_real_index = wiki_index_iter_n
        print(len(wiki_real_index))
        all_indices_yago = custom_top_k_gpu(output[wiki_real_index], output[dev_pair[:, 0]])
        #the yago part performance
        print(check_pi(wiki_real_index, all_indices_yago, yago_index_i, top))
        #update the initilized entity.
        yago_real_index_1 = index_mod(all_indices_yago, yago_index_i, top)
        yago_index_iter = set(list(entity_1an)) | set(list(yago_real_index_1))
        yago_index_iter_n = np.array(list(yago_index_iter))
        print(len(yago_index_iter_n))
        entity_start = yago_index_iter_n
        entity_start_wiki = wiki_real_index
    return set(list(entity_start)), set(list(entity_start_wiki))

def check_cov(ent_1, ent_2, all_pair_dic):
    count = 0
    for ent in ent_1:
        if ent not in all_pair_dic:
            continue
        if all_pair_dic[ent] in ent_2:
            count += 1
    return count / len(ent_1)

def gen_ent_hop(triples, entity):
    """get the entity where we have within the one hop"""
    rec = set()
    for triple in triples:
        head, tail = triple[0], triple[2]
        if head in entity:
            rec.add(head)
            rec.add(tail)
        if tail in entity:
            rec.add(head)
            rec.add(tail)
    return rec

def gen_set_test(dev_pair_select, entity_1ha, entity_2ha):
    rec = set()
    rec_w = set()
    for pair in dev_pair_select:
        if pair[0] in entity_1ha:
            rec.add(pair[0])
        if pair[1] in entity_2ha:
            rec_w.add(pair[1])
    return rec, rec_w

def gen_refine_index(dev_pair_select, real_set):
    rec = []
    for i, ent in enumerate(real_set):
        if ent in dev_pair_select[:, 0]:
            rec.append(i)
    return np.array(rec)

def gen_refine_index_r(dev_pair_select, real_set):
    rec = []
    for ent in real_set:
        if ent in dev_pair_select[:, 0]:
            rec.append(ent)
    return np.array(rec)

import tensorflow as tf

# sys.path.append('/home/jiayun/Desktop/DualMatch')
# from seu_tkg import sinkhorn, cal_sims

def cal_sims_(entity_1a, entity_1b, feature, right=None):
    entity_1a = np.array(list(entity_1a))
    entity_1b = np.array(list(entity_1b))
    if right is None:
        feature_a = tf.gather(indices=entity_1a, params=feature)
        feature_b = tf.gather(indices=entity_1b, params=feature)
    else:
        feature_a = feature[:right]
        feature_b = feature[right:]
    fb = tf.transpose(feature_b, [1, 0])
    return tf.matmul(feature_a, fb)

def sinkhorn(sims, eps=1e-6):
    with tf.device('/CPU:0'):
        sims = tf.exp(sims * 50.0)
        for _ in range(10):
            sims = sims / (tf.reduce_sum(sims, axis=1, keepdims=True) + eps)
            sims = sims / (tf.reduce_sum(sims, axis=0, keepdims=True) + eps)
    return sims

def gen_index_rank_score(score_s, entity_1, entity_2, index_rec):
    """generate with {index: args}"""
    score_s = np.array(score_s)
    rec = {}
    count = 0
    for i, score in enumerate(score_s):
        if i not in index_rec:
            continue
        index_sort = np.argsort(-score_s[i])
        index_sort = index_sort[:100]
        #print(index_sort)
        score_c = score_s[i][index_sort]
        # if all_pair_dic[entity_1[i]] == entity_2[index_sort[0]]:
        #     print(1)
        #     count += 1
        rec_ = {}
        for j, index in enumerate(index_sort):
            rec_[entity_2[index]] = score_c[j].item()
        rec[entity_1[i]] = rec_
    return rec, count

from collections import Counter

def combine_ranks_multi(rank_dic_list):
    """
    rank_dic_list: List[Dict[int, Dict[Any, float]]]
    """
    rec = {}

    # collect all outer keys
    all_keys = set()
    for rd in rank_dic_list:
        all_keys.update(rd.keys())

    for i in tqdm(all_keys):
        counter = Counter()

        for rd in rank_dic_list:
            if i in rd:
                counter += Counter(rd[i])

        # retain single-existing results naturally
        if counter:
            rec[i] = dict(
                sorted(counter.items(), key=lambda item: item[1], reverse=True)
            )
    return rec

def gen_acc(score_dic, score_dic_c):
    """generate with the part of the accuracy splited"""
    count = 0
    count_c = 0
    count_i = 0
    count_ic = 0
    for i, j in score_dic.items():
        #if i in ent_set_ec1:
        if i in score_dic_c:
            count_i += 1
            if all_pair_dic[i] == list(j.keys())[0]:
                count_ic += 1
        else:
            count += 1
            if all_pair_dic[i] == list(j.keys())[0]:
                count_c += 1

    #the accuracy for outer, inner and the general.
    return  count_c / count, count_ic / count_i, (count_c + count_ic) / (count + count_i)

def gen_acc_s(score_s, index_e):
    """generate with the part of the accuracy splited"""
    count = 0
    count_c = 0
    count_e = 0
    count_ec = 0
    for i in range(len(score_s)):
        #if i in ent_set_ec1:
        if i in index_e:
            count_e += 1
            if i == np.argmax(score_s[i]):
                count_ec += 1
        else:
            count += 1
            if i == np.argmax(score_s[i]):
                count_c += 1

    #the accuracy for outer, inner and the general.
    return  count_c / count, count_ec / count_e, (count_c + count_ec) / (count + count_e)

def gen_refine_dev(dev_pair_select, entity_set):
    rec = []
    for i, pair in enumerate(dev_pair_select):
        if pair[0] in entity_set:
            rec.append(i)
    return rec

def gen_refine_set(dev_pair_select, entity_set, new_ent):
    rec_e = []
    rec_ei = []
    rec_o = []
    rec_a = []
    for i, pair in enumerate(dev_pair_select):
        if pair[0] in entity_set and pair[0] not in new_ent:
            rec_e.append(i)
        elif pair[0] in new_ent:
            rec_ei.append(i)
        else:
            rec_o.append(i)
        rec_a.append(i)
    return [set(rec_e), set(rec_ei), set(rec_o), set(rec_a)]

def split_score_dic(score_dic, score_dic_c):
    rec = {}
    rec_c = {}
    for i in score_dic:
        if i in score_dic_c:
            rec[i] = score_dic[i]
        else:
            rec_c[i] = score_dic[i]
    #in score_dic, out score_dic.
    return rec, rec_c

def split_score_dic_c(score_dic, score_dic_c, score_dic_p):
    ent_in = set(list(score_dic_c.keys())) - set(list(score_dic_p.keys()))
    rec_i = {}
    rec_c = {}
    rec_in = {}
    for i in score_dic:
        if i in score_dic_c and i not in ent_in:
            rec_i[i] = score_dic[i]
        elif i in ent_in:
            rec_in[i] = score_dic[i]
        else:
            rec_c[i] = score_dic[i]
    #in score_dic, new appear, out score_dic.
    return rec_i, rec_in, rec_c

def gen_acc_m(rank_dic, all_pair_dic):
    """generate the score for hit m for the rank dic"""
    count = 0
    count_5 = 0
    count_10 = 0
    for index, index_score in rank_dic.items():
        index_c = all_pair_dic[index]
        if index_c == list(index_score.keys())[0]:
            count += 1
            count_5 += 1
            count_10 += 1
        elif index_c in list(index_score.keys())[:5]:
            count_5 += 1
            count_10 += 1
        elif index_c in list(index_score.keys())[:10]:
            count_10 += 1
    return count / len(rank_dic), count_5 / len(rank_dic), count_10/ len(rank_dic)

def split_score_ent(score_dic, ent_i):
    """for the previous part, we only split with the inner and outer"""
    rec_i = {}
    rec_o = {}
    for i in score_dic:
        if i in ent_i:
            rec_i[i] = score_dic[i]
        else:
            rec_o[i] = score_dic[i]
    #in score_dic, out score_dic.
    return rec_i, rec_o

def split_score_ent_c(score_dic, ent_i, ent_n):
    """for the previous part, we only split with the inner and outer"""
    rec_i = {}
    rec_n = {}
    rec_o = {}
    for i in score_dic:
        if i in ent_i:
            rec_i[i] = score_dic[i]
        elif i in ent_n:
            rec_n[i] = score_dic[i]
        else:
            rec_o[i] = score_dic[i]
    #in score_dic, out score_dic.
    return rec_i, rec_n, rec_o

def unique_set_conf(sim, entity_start_n, entity_wiki_n, thres):
    """
    select part of the entity_start and entity_wiki
    to ensure the accuracy and recall.
    """
    count = 0
    rec_a = set()
    rec_b = set()
    for i in range(len(sim)):
        index_a = np.argmax(sim[i])
        index_sort = np.argsort(sim[i])
        index_2 = index_sort[-2]
        #index_a = int(tf.argmax(sim[i]).numpy())
        #score_m = (sim[i, index_a].numpy() + sim[index_a, i].numpy()) / 2
        if i == np.argmax(sim[:, index_a])\
              and sim[i, index_a] > thres:
            rec_a.add(entity_start_n[i])
            rec_b.add(entity_wiki_n[index_a])
            if all_pair_dic[entity_start_n[i]] == entity_wiki_n[index_a]:
                count += 1

    return count / len(rec_a), rec_a, rec_b

ent_size = 0
eval_epoch = 3
node_hidden = 50
rel_hidden = 50
#time_hidden = int(node_hidden / 2)
time_hidden = 50
batch_size = 512
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
device = 'cuda:0'
#print(rel_size)

training_time = 0.
grid_search_time = 0.
time_encode_time = 0.

rec_score_list = []
dev_pair_list = []
entity_e = []
entity_eh = []

for j, snap_shot in enumerate(snap_shots):
    print('___________________________')
    print(j)

    #for the previous snapshot, we test with the current 
    #part of the dev pair and record the result score dict.

    #we start with the second snapshot for the general evaluation pipeline.
    #the base model version. 
    if j == 0:
        #load with the model of the current snapshot.
        train_pair, dev_pair, adj_matrix, adj_features, rel_features,\
    time_features, time_int_features, radj,\
        time_dict_i, time_int_dict_i, triples_1, triples_2, entity_1a, entity_2a = \
    utils_n_nself_b.load_data_i(file_path + filename, snap_shot, train_ratio=4000, \
                           time_dict_i=time_dic_i, time_int_dict_i=time_index_dic)
        
        dev_pair_list.append(dev_pair)
        
        adj_matrix, rel_matrix, rel_val, ent_matrix, ent_val, time_matrix, time_val, time_int_matrix,\
    time_int_val, node_size, rel_size, time_size, time_int_size, triple_size \
        = gen_garph_stat(adj_matrix, rel_features, adj_features, time_features, time_int_features)

        inputs = [adj_matrix, train_pair]

        model_o = \
        model_o_att_e.OverAll(node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
                        rel_size=rel_size, rel_hidden=rel_hidden,
                        time_size=time_size, time_int_size=time_int_size,
                        ent_matrix=ent_matrix, ent_val=ent_val,
                        rel_matrix=rel_matrix, rel_val=rel_val,
                        time_matrix=time_matrix, time_val=time_val, out_dim=4, interval_index=int_index,
                        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
                        triple_size=triple_size, dropout_rate=dropout_rate,
                        depth=depth, device=device)

        model_o = model_o.to(device)

        # param_time = torch.load('/home/jiayun/Desktop/temp_evo_refine/model_over_t40.pth')
        # param_rel =\
        # torch.load('/home/jiayun/Desktop/temp_evo_refine/model/evolve_rel_40/model_rel_40_0.pth')
        # param_overall = model_o.state_dict()
        # for key, value in param_rel.items():
        #     if key in param_overall:
        #         param_overall[key] = value

        # for key, value in param_time.items():
        #     if key in param_overall and key != 'ent_emb' and key != 'rel_emb':
        #         param_overall[key] = value

        # #load with the general part of the dict.
        # model_o.load_state_dict(param_overall)
        model_o.load_state_dict\
        (torch.load\
         (f"{str(folder_path)}/model_refine_10.pth"))
        
        model_o.use_temp = True
        model_o.only_rel = True
        model_o.only_temp = True

        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
            #output = tf.convert_to_tensor(output)
            sim = cal_sims(dev_pair, output)
            score_sr = sinkhorn(sim)
            hits_, acc, total_mean, wrong_index_0, wrong_rank_0, wrong_rank_index_0, wrong_scores_0, rank_0\
                = multi_thread_cal(score_sr.numpy(), 20, [1, 5, 10])
            print(hits_, acc, total_mean)
        model_o.train()
        del output
        del sim
        del score_sr
        gc.collect()

        #generate with the part of the result_score_dic for these.
        dev_pair_dic_r1 = dict(zip(np.arange(0, len(dev_pair_list[j])), dev_pair_list[j][:, 1]))
        rank_c1 = convert_rank(rank_0, dev_pair_dic_r1)
        rank_cs1 = gen_rank_score(rank_c1, wrong_scores_0[0])
        #here, the score dict is what we have in the following part generation.
        rank_dic_cs1 = dict(zip(dev_pair_list[j][:, 0], rank_cs1))

        rec_score_list.append(rank_dic_cs1)
    
    #then we evaluate only adj update version.

    #here, we evaluate the part of the finetuning version.
    else:
        #the current snapshot for the comparison.
        train_pair, dev_pair, adj_matrix, adj_features, rel_features,\
    time_features, time_int_features, radj,\
        time_dict_i, time_int_dict_i, triples_1, triples_2, entity_1a, entity_2a = \
    utils_n_nself_b.load_data_i(file_path + filename, snap_shot, train_ratio=4000, \
                           time_dict_i=time_dic_i, time_int_dict_i=time_index_dic)
        
        dev_pair_list.append(dev_pair)
        
        adj_matrix, rel_matrix, rel_val, ent_matrix, ent_val, time_matrix, time_val, time_int_matrix,\
    time_int_val, node_size, rel_size, time_size, time_int_size, triple_size \
        = gen_garph_stat(adj_matrix, rel_features, adj_features, time_features, time_int_features)

        inputs = [adj_matrix, train_pair]

        model_o = \
        model_o_att_e.OverAll(node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
                        rel_size=rel_size, rel_hidden=rel_hidden,
                        time_size=time_size, time_int_size=time_int_size,
                        ent_matrix=ent_matrix, ent_val=ent_val,
                        rel_matrix=rel_matrix, rel_val=rel_val,
                        time_matrix=time_matrix, time_val=time_val, out_dim=4, interval_index=int_index,
                        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
                        triple_size=triple_size, dropout_rate=dropout_rate,
                        depth=depth, device=device)

        model_o = model_o.to(device)

        model_o.load_state_dict\
                (torch.load(f"{str(folder_path)}/model_refine_{snap_shot[:-1]}.pth"))

        model_o.use_temp = True
        model_o.only_rel = True
        model_o.only_temp = True

        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
        model_o.train()

        entity_e.append([entity_1a])
        entity_eh.append([entity_1hop])
        
        wiki_index_i = dict(zip(np.arange(0, len(dev_pair), 1), dev_pair[:, 1]))
        yago_index_i = dict(zip(np.arange(0, len(dev_pair), 1), dev_pair[:, 0]))
        entity_1an = np.array(list(entity_1a))

        entity_start, entity_start_wiki = build_search_bi(entity_1an,\
                                 output, dev_pair, 2, 1, wiki_index_i, yago_index_i)

        print(len(entity_start), len(entity_start_wiki))

        entity_wiki = entity_start_wiki | entity_2a
        print(len(entity_wiki))

        #here, we get the part for the selected entity_wiki part.
        entity_start_n = np.array(list(entity_start))
        entity_wiki_n = np.array(list(entity_wiki))

        sim = cal_sims_(entity_start_n, entity_wiki_n, output)
        score_s = sinkhorn(sim)
        print(sim.shape)

        acc, rec_a, rec_b = unique_set_conf(sim, entity_start_n, entity_wiki_n, 0.8)
        print(acc)

        entity_start = rec_a
        selected_wiki_ent = rec_b

        del sim
        del score_s
        gc.collect()

        print('original coverage vs. selected coverage')
        print(check_cov(entity_start, entity_wiki, all_pair_dic))
        print(check_cov(entity_start, selected_wiki_ent, all_pair_dic))
        
        #we use entity_1ha and entity_2hop for real sim part.
        entity_1ha = gen_ent_hop(triples_1, entity_start)
        entity_1hop = gen_ent_hop(triples_1, entity_1a)
        entity_2ha = gen_ent_hop(triples_2, entity_wiki)
        entity_2hop = gen_ent_hop(triples_2, selected_wiki_ent)

        print(check_cov(entity_1ha, entity_2ha, all_pair_dic))
        print(check_cov(entity_1ha, entity_2hop, all_pair_dic))

        print(len(entity_1ha), len(entity_1hop), len(entity_2ha), len(entity_2hop))

        entity_1has, entity_2has = gen_set_test(dev_pair_list[j], entity_1ha, entity_2hop)
        #the entity only in the subset for dev pair
        print(len(entity_1has), len(entity_2has))

        index_rec = gen_refine_index(dev_pair_list[j], entity_1has)
        print(len(index_rec))

        ts = time.time()

        sim = cal_sims_(entity_1has, entity_2has, output)
        score_s = sinkhorn(sim)

        print(score_s.shape)
        
        #get with the updated part of score dict.
        index_record_score_, count = \
            gen_index_rank_score(score_s, list(entity_1has), list(entity_2has), index_rec) 
        
        del output
        del sim
        del score_s
        gc.collect()
        
        rank_com_dic_sp = combine_ranks_multi(rec_score_list)
        #new_ent = set(list(index_record_score_.keys())) - set(list(rank_com_dic_sp.keys()))
        rec_score_list.append(index_record_score_)
        rank_com_dic_s = combine_ranks_multi(rec_score_list)
        
        set_new = set(list(dev_pair_list[j][:, 0])) - set(list(dev_pair_list[j-1][:, 0]))

        index_1hop = gen_refine_index_r(dev_pair_list[j], entity_1hop)
        #the eovlving, new, outer, all.
        index_1hop_e = set(list(index_1hop)) - set_new
        index_1hop_n = set_new
        index_1hop_o = set(list(dev_pair_list[j][:, 0])) - set(list(index_1hop))
        index_1hop_all = set(list(dev_pair_list[j][:, 0]))
        print(len(index_1hop_e), len(index_1hop_n), len(index_1hop_o), len(index_1hop_all))

        #then we do the general part of the evaluation 
        #for the previous one and the current one.
        #here, we split it into the evolving related and not related 
        #part entities.

        print('generate score')
        # print(gen_acc(rank_com_dic_sp, index_record_score_))
        # print(gen_acc(rank_com_dic_s, index_record_score_))
        #the previous part of the curernt.
        print('previous combined')
        rec_i, rec_o = split_score_ent(rank_com_dic_sp, index_1hop_e)
        print(gen_acc_m(rec_i, all_pair_dic))
        print(gen_acc_m(rec_o, all_pair_dic))
        print(gen_acc_m(rank_com_dic_sp, all_pair_dic))

        #the part of the current combined.
        print('current combined')
        rec_i, rec_n, rec_o = split_score_ent_c(rank_com_dic_s, index_1hop_e, index_1hop_n)
        print(gen_acc_m(rec_i, all_pair_dic))
        print(gen_acc_m(rec_n, all_pair_dic))
        print(gen_acc_m(rec_o, all_pair_dic))
        print(gen_acc_m(rank_com_dic_s, all_pair_dic))

test_list = [0.4]

from collections import defaultdict
from tqdm import tqdm

def combine_ranks_multi_m(rank_dic_list):
    """
    rank_dic_list: List[Dict[int, Dict[Any, float]]]
    """
    rec = {}

    # collect all outer keys
    all_keys = set()
    for rd in rank_dic_list:
        all_keys.update(rd.keys())

    for i in tqdm(all_keys):
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)

        for rd in rank_dic_list:
            if i in rd:
                for k, v in rd[i].items():
                    sum_dict[k] += v
                    count_dict[k] += 1

        # compute mean
        if sum_dict:
            mean_dict = {
                k: sum_dict[k] / count_dict[k]
                for k in sum_dict
            }

            rec[i] = dict(
                sorted(mean_dict.items(), key=lambda item: item[1], reverse=True)
            )

    return rec

def refine_dict_p_(comb_p, comb_c, index_score, thres=0.01):
    """
    this one we refine the dict only based on the 
    combined score diff with a threshold.
    """
    rec = {}
    for index, score_dic in comb_c.items():
        if index not in comb_p:
            rec[index] = score_dic
        else:
            #less than thres retain the original.
            if index in index_score:
                #the current conf
                a = list(index_score[index].values())[0] - list(index_score[index].values())[1]
                #the previous conf
                b = list(comb_p[index].values())[0] - list(comb_p[index].values())[1]
                #if  np.abs(a - b) < thres:
                if a - b > thres:
                    #rec[index] = comb_p[index]
                    rec[index] = index_score[index]
                elif b - a > thres:
                    rec[index] = comb_p[index]
                else:
                    rec[index] = score_dic
            else:
                rec[index] = score_dic
    return rec

#the final inference
for value in test_list:
    rec_list = []
    for i in range(9):
        #target_p = rec_p[i]
        if i == 0:
            target_p = score_list[0]
            target_comb = combine_ranks_multi_m(score_list[: 2])
        else:
            target_p = rank_dic_update
            target_comb = combine_ranks_multi_m([rank_dic_update, score_list[i+1]])
        rank_dic_update = refine_dict_p_(target_p, target_comb, score_list[i + 1], value)
        rec_list.append(rank_dic_update)
        
    #refine_dict[value] = rec_list
    print('________________________________________')
    print(value)
    for j in range(9):
        set_new = set(list(dev_pair_list[j+1][:, 0])) - set(list(dev_pair_list[j][:, 0]))

        index_1hop = gen_refine_index_r(dev_pair_list[j+1], entity_eh[j][0])
        #the eovlving, new, outer, all.
        index_1hop_e = set(list(index_1hop)) - set_new
        index_1hop_n = set_new
        index_1hop_o = set(list(dev_pair_list[j+1][:, 0])) - set(list(index_1hop))
        index_1hop_all = set(list(dev_pair_list[j+1][:, 0]))
        print(len(index_1hop_e), len(index_1hop_n), len(index_1hop_o), len(index_1hop_all))

        #then test the part of the original part of the general result.
        #here we create the reuslt.
        result = rec_list[j]
        rec_i, rec_n, rec_o = split_score_ent_c(result, index_1hop_e, index_1hop_n)
        print(gen_acc_m(rec_i, all_pair_dic))
        print(gen_acc_m(rec_n, all_pair_dic))
        print(gen_acc_m(rec_o, all_pair_dic))
        print(gen_acc_m(result, all_pair_dic))

