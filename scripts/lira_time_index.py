import pymongo
from tqdm import tqdm
from collections import defaultdict
import numpy as np

client = pymongo.MongoClient()
coll = client.lira_time.resnet18_cifar100
coll.create_index([('exid', 1)])



def make_summary_docs(base_coll, chunk_size):
    summary_docs = []
    chunks = list(range(0, 50 * 1000 + 1, chunk_size))
    last_chunk = -1
    for chunk in tqdm(chunks[1:]):
        docs = list(base_coll.find({'exid': {'$gt': last_chunk, '$lte': chunk},
                                    'margin': {'$ne': float('nan')}},
                                   {'exid': 1, 'epoch': 1, 'member': 1, 'margin': 1}))
        groups = defaultdict(list)
        for doc in docs:
            if -10000 < doc['margin'] < 10000:
                groups[(doc['epoch'], doc['member'], doc['exid'])].append(doc['margin'])
            
        group_summary = {}
        for k, v in groups.items():
            group_summary[k] = (np.mean(v), np.std(v), len(v))
        summary_docs.extend([{'exid': k[2], 'member': k[1], 'epoch': k[0], 
                             'mean': v[0], 'std': v[1], 'count': v[2]} for k, v in group_summary.items()])
        last_chunk = chunk
    return summary_docs


summary_docs = make_summary_docs(coll, 100)
summary_coll = client.lira_time.summary_stats
summary_coll.insert_many(summary_docs)



