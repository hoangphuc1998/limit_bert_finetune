import numpy as np
from collections import Counter

def read_hatexplain(data):
    tokens = []
    labels = []
    rationales = []
    for item in data:
        if len(item['post_tokens'])>128:
            continue
        #invalid_indices = [i for i, x in enumerate(item['post_tokens']) if x=='\u200d' or x=='️' or x=='\u202c' or x=='\u202a']
        invalid_indices = [i for i, x in enumerate(item['post_tokens']) if len((x.encode('ascii', 'ignore')).decode("utf-8"))==0]
        token = [x for i, x in enumerate(item['post_tokens']) if i not in invalid_indices]
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
        rationale = np.delete(rationale, invalid_indices)
        tokens.append(token)
        labels.append(label)
        rationales.append(rationale)
    return tokens, labels, rationales

def encode_tags(labels, encodings):
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return np.array(encoded_labels)