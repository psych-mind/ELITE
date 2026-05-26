import sys

from config_ import cfg
import model_o_att_e
import utils_n_nself
import torch

import numpy as np

import time
from tqdm import *

cfg = cfg()
cfg.get_args()
cfgs = cfg.update_train_configs()

from seu_tkg import sinkhorn, cal_sims
from CSLS_test_4 import *

import os.path as osp
from kneed import KneeLocator

spe_string = 'evolve_model_yago'

from pathlib import Path

folder_name = f"model/{spe_string}"
folder_path = Path.cwd() / folder_name

# Check and create folder
folder_path.mkdir(parents=True, exist_ok=True)
print(f"Folder ready at: {folder_path}")

file_path = str(Path.cwd()) + '/dataset/'
filename = 'YAGO_WIKI180K_E/'

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([])
    time_int = set([])
    time = set([])
    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            head, r, tail, ts, te = [int(item) for item in para]
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
      load_triples(file_path + 'YAGO-WIKI180K/triples_1')
entity_2, rel_2, triples_2, time_int_2, time_2 = \
load_triples(file_path + 'YAGO-WIKI180K/triples_2')

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

#here, we would like to loop from 40 to 50.
snap_shots = [f'{i}/'for i in range(40, 50)]

#get with the all_pair_dic.
def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair

all_pair = load_alignment_pair(file_path + filename + 'ref_pairs')
all_pair = np.array(all_pair)
all_pair_dic = dict(zip(all_pair[:, 0], all_pair[:, 1]))
all_pair_dic_i = dict(zip(all_pair[:, 1], all_pair[:, 0]))

dev_pair_list = []

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

# sen_list = [50]

# mid_score = [0.5, 0.6, 0.7, 0.8]

# for sen in sen_list:
#     for score_m in mid_score:
# dir_path = f"{str(folder_path)}/{sen}/{score_m}"

# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)
# print(sen, score_m)

#here, we test with the part of 80 0.5 setting.
print('_________________________')

for j, snap_shot in enumerate(snap_shots):

    train_pair, dev_pair, adj_matrix, adj_features, rel_features,\
    time_features, time_int_features, radj,\
        time_int_dict_i, triples_1, triples_2, entity_1a, entity_2a = \
    utils_n_nself.load_data_i(file_path + filename, snap_shot, train_ratio=2000, \
                            time_int_dict_i=time_index_dic)
    
    #the first snapshot, and create the part with only evolving part entity eval.
    #here, we choose all the entities instead of only choosing a subset.
    if j == 0:
        # select_num = len(dev_pair) // 4
        # np.random.seed(42)
        # numbers = np.random.choice(len(dev_pair), size=select_num, replace=False)
        # select_numbers = np.array(numbers)
        # dev_pair_select = dev_pair[select_numbers]
        # #generate with the part of the selected eovlving ones.
        # #dev_pair_s = gen_dev_sub(dev_pair_select, entity_ha1)
        # print(dev_pair_select[:5], len(dev_pair_select))
        dev_pair_ = dev_pair
        dev_pair_list.append(dev_pair)
        continue
    else:
        #here, we modify with the dev pair to include with the new entities.
        #do the evaluation from the second snapshot.
        #previous and the current.
        dev_set_1 = set(list(dev_pair_[:, 0]))
        dev_set_2 = set(list(dev_pair[:, 0]))
        dev_app = gen_dev(dev_set_2 - dev_set_1, all_pair_dic)
        print(len(dev_app))
        dev_pair_ = dev_pair
        # dev_pair_select = np.vstack((dev_pair_select, dev_app))
        # print(dev_pair_select[:5], len(dev_pair_select))
        dev_pair_list.append(dev_pair)

        _, _, triples_1t, _, _ =\
    load_triples(file_path + filename + f'{snap_shot[:-1]}/triples_1')
        
        _, _, triples_n, _, _ = \
            load_triples(file_path + filename + 'triples_nt1')

        triples_1o = triples_n + triples_1t
        print(len(triples_1o), len(triples_n), len(triples_1t))

        entity_ha1 = gen_ent_1(triples_1o, entity_1a)
        print(len(entity_ha1))  
        #the dev pair for eval.
        dev_pair_s = gen_dev_sub(dev_pair, entity_ha1)
        #then we first test the original version then the updated version.

        #load the model.
        adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
        #adj_matrix_t = np.stack(adj_matrix_t.nonzero(), axis=1)

        rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
        ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
        time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data

        time_int_matrix, time_int_val = np.stack(time_int_features.nonzero(), axis=1), \
            time_int_features.data

        node_size = adj_features.shape[0]
        rel_size = rel_features.shape[1]
        time_size = time_features.shape[1]
        time_int_size = time_int_features.shape[1]
        ent_size = 0

        triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)
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
        # model_o.use_temp = False
        # model_o.only_rel = True
        # model_o.only_temp = False

        #load with the general paramters for rel and time.
        #the first one is separate, then the combined evaluation.
        if j == 1:
        #     param_time = torch.load('/home/jiayun/Desktop/temp_evo_refine/model_over_t40.pth')
        #     param_rel =\
        # torch.load('/home/jiayun/Desktop/temp_evo_refine/model/evolve_rel_40/model_rel_40_0.pth')
        #     param_overall = model_o.state_dict()
        #     for key, value in param_rel.items():
        #         if key in param_overall:
        #             param_overall[key] = value

        #     for key, value in param_time.items():
        #         if key in param_overall and key != 'ent_emb' and key != 'rel_emb':
        #             param_overall[key] = value

        #     #load with the general part of the dict.
        #     model_o.load_state_dict(param_overall)
            #load with the general part of the model.
            model_o.load_state_dict\
        (torch.load\
         (f"{str(folder_path)}/model_refine_10.pth"))
            
        else:

            model_o.load_state_dict\
                (torch.load(f"{str(folder_path)}/model_refine_{snap_shots[j-1][:-1]}.pth"))
            #print(int(snap_shot[:-1]-1))

        inputs = [adj_matrix, train_pair]

        #evalute with the combined manner.
        model_o.use_temp = True
        model_o.only_rel = False
        model_o.only_temp = False
        
        ts = time.time()
        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
            #output = tf.convert_to_tensor(output)
            sim = cal_sims(dev_pair_s, output)
            score_s = sinkhorn(sim)
            print(multi_thread_cal_(score_s.numpy(), 20, [1, 5, 10]))
        model_o.train()

        del output
        del sim
        del score_s
        gc.collect()
        te = time.time()
        print(te - ts)
        
        ts = time.time()
        #evalute with temporal.
        model_o.use_temp = False
        model_o.only_rel = False
        model_o.only_temp = True
        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
            #output = tf.convert_to_tensor(output)
            sim = cal_sims(dev_pair_s, output)
            score_st = sinkhorn(sim)
            print(multi_thread_cal_(score_st.numpy(), 20, [1, 5, 10]))
        model_o.train()

        del output
        del sim
        gc.collect()
        te = time.time()
        print(te - ts)
        
        ts = time.time()
        #evaluate with relation.
        model_o.use_temp = False
        model_o.only_rel = True
        model_o.only_temp = True

        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
            #output = tf.convert_to_tensor(output)
            sim = cal_sims(dev_pair_s, output)
            score_sr = sinkhorn(sim)
            print(multi_thread_cal_(score_sr.numpy(), 20, [1, 5, 10]))
        model_o.train()

        del output
        del sim
        gc.collect()
        te = time.time()
        print(te - ts)
        
        ts = time.time()
        #then generate the overall seeds for relation and temporal training.
        #the bi-directional confident seeds.
        rec_sort_rel = gen_top_seeds(score_sr)
        rec_sort_time = gen_top_seeds(score_st)

        print(len(rec_sort_rel), len(rec_sort_time))

        #rel conf time not conf
        rec_sort_t, score_t = pair_sort(rec_sort_rel, score_st, 0.8)
        #time conf rel not conf
        rec_sort_r, score_r = pair_sort(rec_sort_time, score_sr, 0.8)

        del score_st
        del score_sr
        gc.collect()
        te = time.time()
        print(te - ts)
        
        #here, we select with the proper thereshold and indices 
        #for general selection of the seeds.
        index_r, thres_r = gen_threshold(score_r, 50)
        index_t, thres_t = gen_threshold(score_t, 50)
        print(index_r, thres_r)
        print(index_t, thres_t)

        #then we generate the seeds and report the seeds accuracy.
        rec_r, acc_r = check_acc(rec_sort_r, index_r)
        rec_t, acc_t = check_acc(rec_sort_t, index_t)
        print(acc_r, acc_t)

        #generate the real index.
        pair_dic_y = dict(zip(dev_pair_s[:, 0], np.arange(len(dev_pair_s))))
        pair_dic_yi = dict(zip(np.arange(len(dev_pair_s)), dev_pair_s[:, 0]))
        pair_dic_w = dict(zip(dev_pair_s[:, 1], np.arange(len(dev_pair_s))))
        pair_dic_wi = dict(zip(np.arange(len(dev_pair_s)), dev_pair_s[:, 1]))

        train_rel, acc_r = transfer_pair(rec_r, pair_dic_yi, pair_dic_wi, all_pair_dic)
        train_time, acc_t = transfer_pair(rec_t, pair_dic_yi, pair_dic_wi, all_pair_dic)
        print(len(train_rel), len(train_time))

        train_g = np.vstack((train_rel, train_time))
        print(train_g.shape)

        k = 2000  # number of samples you want

        idx = np.random.choice(train_g.shape[0], size=k, replace=False)
        train_g = train_g[idx]
        print(train_g.shape)

        #evalute with the combined manner.
        model_o.use_temp = True
        model_o.only_rel = False
        model_o.only_temp = False

        opt = torch.optim.RMSprop(model_o.parameters(), lr=lr, weight_decay=0)
        print('model constructed')

        #here, we train with the general model.
        epoch_r = 6

        start = time.time()
        # tic = time.time()
        for i in trange(epoch_r):
            np.random.shuffle(train_g)
            for pairs in [train_g[i * batch_size:(i + 1) * batch_size] for i in
                            range(len(train_g) // batch_size + 1)]:
                inputs = [adj_matrix, pairs]
                output, loss = model_o(inputs)
                #loss_ent = align_loss(pairs, output_ent, node_size)
                #print(loss_r, loss_t)
                print(loss)
                #loss_c = (loss_r + loss_t) / 2
                loss.backward()
                #loss_r.backward(retain_graph=True)
                #loss_tem.backward()
                opt.step()
                opt.zero_grad()
        
        print('after')
        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
            #output = tf.convert_to_tensor(output)
            sim = cal_sims(dev_pair_s, output)
            score_s = sinkhorn(sim)
            print(multi_thread_cal_(score_s.numpy(), 20, [1, 5, 10]))
        model_o.train()

        del output
        del sim
        del score_s
        gc.collect()
        
        ts = time.time()
        model_o.eval()
        with torch.no_grad():
            output, loss = model_o(inputs)
            output = output.cpu().numpy()
            output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-5)
            #output = tf.convert_to_tensor(output)
            sim = cal_sims(dev_pair, output)
            score_s = sinkhorn(sim)
            print(multi_thread_cal_(score_s.numpy(), 20, [1, 5, 10]))
        model_o.train()
        te = time.time()
        print(te - ts)

        del output
        del sim
        del score_s

        torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()
        
        #save the model.

        torch.save(model_o.state_dict(),\
                        f"{str(folder_path)}/model_refine_{snap_shot[:-1]}.pth")
