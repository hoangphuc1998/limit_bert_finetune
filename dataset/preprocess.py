import numpy as np
from collections import Counter

def read_hatexplain(data):
    tokens = []
    labels = []
    rationales = []
    for item in data:
        if len(item['post_tokens'])>128:
            continue
        user_indices = [i for i, x in enumerate(item['post_tokens']) if x=='<user>']
        token = [x for i, x in enumerate(item['post_tokens']) if i not in user_indices]
        label_counter = Counter(item['annotators']['label'])
        majority_label = label_counter.most_common(1)[0]
        label = 1
        if majority_label[1]==1:
            continue
        else:
            label = majority_label[0]
        annotator_rationale = item['rationales']
        rationale=None
        if len(annotator_rationale)==0:
            rationale = np.zeros(len(item['post_tokens']), dtype='int')
        else:
            rationale=np.array(annotator_rationale[0])
        rationale = np.delete(rationale, user_indices)
        tokens.append(token)
        labels.append(label)
        rationales.append(rationale)
    return tokens, labels, rationales