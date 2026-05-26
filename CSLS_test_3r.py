import multiprocessing

import gc
import os

import numpy as np
import time
from tqdm import *

from scipy.spatial.distance import cdist

g = 1000000000

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return
    
    
def cal_index(task, score, top_k):
    """get the hits1, hits5, hits10 number for each task"""
#     prediction_array = torch.argsort(score, descending=True, axis=-1)
#     hits1_score = precition_array[:, 0]
#     hits5_score = prediction_array[:, :5]
#     hits10_score = prediction_array[:, :10]
#     hits1_num = rec_num(hits1_score)
#     hits5_num = rec_num(hits5_score)
#     hits10_num = rec_num(hits10_score)
    #hits1_num = 0
    #hits5_num = 0
    #hits10_num = 0
    hits_num = [0, 0, 0]
    hits_wrong = [[], [], []]
    hits_rank = [[], [], []]
    hits_rank_index = [[], [], []]
    hits_score = [[], [], []]
    hits_counter = [[], [], []]
    hits_rank_a = []
    #hits1_wrong = []
    #hits5_wrong = []
    #hits10_wrong = []
    mean = 0
#     count = 0 
    for i in tqdm(range(len(task))):
        #print(i)
        ref = task[i]
        #rint(i)
        #rank = torch.argsort(score[i], descending=True)
        rank = (-score[i]).argsort()
        #top score.
        score_ = score[i][rank][:100]
        #the real rank.
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        hits_rank_a.append(rank[:100])
        for j in range(3):
            if rank_index < top_k[j]:
                hits_num[j] += 1
            else:
                hits_wrong[j].append(ref)
                hits_rank[j].append(rank_index)
                hits_rank_index[j].append(rank[:100])
            hits_score[j].append(score_)
        
    #number hits, the wrong pred, the true hits rank, the top k index.
    #the top score, the overall mean rank for ground truth.
    #here, we add one more dimension for the hits_rank_index to show the
    #true part of the rank.   
    return hits_num, hits_wrong, hits_rank, hits_rank_index, hits_score, hits_rank_a, mean


def multi_thread_cal(score, nums_threads, top_k):
    """
    apply the multi-thread machenism to apply the argsort function in torch
    and return the hits1, hits5 and hits10 number respectively.
    """
    #score = np.matmul(Lvec, Rvec.T)
    hits_ = [0, 0, 0]
    wrong_index = [[], [], []]
    wrong_rank = [[], [], []]
    wrong_rank_index = [[], [], []]
    wrong_scores = [[], [], []]

    rank = []
    #hits1 = 0
    #hits5 = 0
    #hits10 = 0
    total_mean = 0
    test_size = len(score)
    tasks = div_list(np.array(range(test_size)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        #print(1)
        reses.append(pool.apply_async(cal_index, (task, score[task, :], top_k)))
    pool.close()
    pool.join()
    
    for res in reses:
        #hits1_num, hits5_num, hits10_num, mean = res.get()
        hits_num, hits_wrong, hits_rank, hits_rank_index, hits_score, hits_rank_a, mean = res.get()
        hits_ += np.array(hits_num)
        rank += hits_rank_a
        for i in range(len(wrong_index)):
            wrong_index[i] += hits_wrong[i]
            wrong_rank[i] += hits_rank[i]
            wrong_rank_index[i] += hits_rank_index[i]
            wrong_scores[i] += hits_score[i]
        #hits1 += hits1_num
        #hits5 += hits5_num
        #hits10 += hits10_num
        total_mean += mean
    acc = np.array(hits_)/test_size * 100
    total_mean /= test_size
    del score
    gc.collect()
    return hits_, acc, total_mean, wrong_index, wrong_rank, wrong_rank_index, wrong_scores, rank