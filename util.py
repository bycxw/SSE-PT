import sys
import copy
import random
import numpy as np
from collections import defaultdict, Counter


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    ndcg_1 = 0.0
    hit_1 = 0.0
    ndcg_5 = 0.0
    hit_5 = 0.0
    ndcg_10 = 0.0
    hit_10 = 0.0
    ap = 0.0
    # allitems = list(range(itemnum + 1))

    prob = get_pop_distribution(dataset)

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        while len(item_idx) < 101:
            sampled_ids = np.random.choice(range(1, itemnum+1), 101, replace=False, p=prob)
            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        item_idx = item_idx[:101]
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)


        # item_idx += allitems[:test[u][0]]
        # item_idx += allitems[test[u][0] + 1:]

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        #print(predictions)
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 1:
            ndcg_1 += 1
            hit_1 += 1
        if rank < 5:
            ndcg_5 += 1 / np.log2(rank + 2)
            hit_5 += 1
        if rank < 10:
            ndcg_10 += 1 / np.log2(rank + 2)
            hit_10 += 1

        ap += 1.0 / (rank + 1)

        # if rank < args.k:
        #     NDCG += 1 / np.log2(rank + 2)
        #     HT += 1
        if valid_user % 1000 == 0:
            #print '.',
            sys.stdout.flush()

    # return NDCG / valid_user, HT / valid_user
    result = {
        "hr_1": hit_1 / valid_user,
        "hr_5": hit_5 / valid_user,
        "hr_10": hit_10 / valid_user,
        "ndcg_5": ndcg_5 / valid_user,
        "ndcg_10": ndcg_10 / valid_user,
        "mrr": ap / valid_user
    }
    return result

def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    ndcg_1 = 0.0
    hit_1 = 0.0
    ndcg_5 = 0.0
    hit_5 = 0.0
    ndcg_10 = 0.0
    hit_10 = 0.0
    ap = 0.0
    allitems = list(range(itemnum + 1))

    prob = get_pop_distribution(dataset)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        while len(item_idx) < 101:
            sampled_ids = np.random.choice(range(1, itemnum+1), 101, replace=False, p=prob)
            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        item_idx = item_idx[:101]
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)
        # item_idx += allitems[:test[u][0]]
        # item_idx += allitems[test[u][0] + 1:]

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1
        if rank < 1:
            ndcg_1 += 1
            hit_1 += 1
        if rank < 5:
            ndcg_5 += 1 / np.log2(rank + 2)
            hit_5 += 1
        if rank < 10:
            ndcg_10 += 1 / np.log2(rank + 2)
            hit_10 += 1

        ap += 1.0 / (rank + 1)
        # if rank < args.k:
        #     NDCG += 1 / np.log2(rank + 2)
        #     HT += 1
        if valid_user % 100 == 0:
            #print '.',
            sys.stdout.flush()

    # return NDCG / valid_user, HT / valid_user
    result = {
        "hr_1": hit_1 / valid_user,
        "hr_5": hit_5 / valid_user,
        "hr_10": hit_10 / valid_user,
        "ndcg_5": ndcg_5 / valid_user,
        "ndcg_10": ndcg_10 / valid_user,
        "mrr": ap / valid_user
    }
    return result


def print_result(epoch, T, valid, test, f=sys.stdout):
    print("epoch: {}, time: {}".format(epoch, T), file=f)
    print("valid:", file=f)
    for k in valid:
        print("{}:{}".format(k, valid[k]), file=f)
    print("test:", file=f)
    for k in test:
        print("{}:{}".format(k, test[k]), file=f)
    print("-----------------------------", file=f)


def get_pop_distribution(dataset):
    counter = Counter()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    user_data = {
        u:
            [item for item in (train[u] + valid[u] + test[u])]
        for u in train if len(train[u]) > 0 and len(valid[u]) > 0 and len(test[u]) > 0
    }
    for u in user_data:
        counter.update(user_data[u])
    total = 0
    for i in range(1, itemnum+1):
        total += counter[i]
    prob = [counter[i] / total for i in range(1, itemnum+1)]

    return prob


