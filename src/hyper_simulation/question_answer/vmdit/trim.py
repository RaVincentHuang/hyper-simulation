import json
from tqdm import tqdm


def read_rel(file_path):
    relation=[]
    with open(file_path, "r") as fin:
        for k,example in enumerate(fin):
            example = json.loads(example)
            for x in example:
                for y in x:
                    relation.append(y)
    #print(len(relation))
    return relation

def read_evi(file_path):
    ids=[]
    ctxs=[]
    query=[]
    with open(file_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            # print(example['ctxs'])
            ctxs.append(example["ctxs"])
            query.append(example["question"])
            ids.append([int(x['id']) for x in example['ctxs']])
    return ids,ctxs,query

def get_context(rel_path,file_path):
    rel=read_rel(rel_path)
    #print(rel)
    all_ids,all_ctxs, query=read_evi(file_path)
    final_evi=[]
    for x in tqdm(range(len(all_ctxs)), desc="Processing", unit="example"):
        ids=all_ids[x]
        del_id=[]
        l=len(ids)
        for i in range(l):
            idx=ids[i]
            for j in range(i+1,l):
                idy=ids[j]
                if [idx,idy] in rel:
                    del_id.append(idy)
        #print("del",del_id)
        evidence=[]
        for ct in all_ctxs[x][:30]:
            if int(ct['id']) not in del_id:
                ct= ct["title"] + "\n" + ct["text"]
                evidence.append(ct)
        # print("l evi:",len(evidence))
        final_evi.append(evidence)
    return final_evi

def calc_trim(rel_path, file_path, trimed_path):
    pro_c = get_context(rel_path, file_path)
    json_data = json.dumps(pro_c)
    with open(trimed_path, "w") as file:
        file.write(json_data)

if __name__=='__main__':
    rel_path = "relation_context/L2/relation_context_popqa_0_50.json"
    file_path= "retr_result/L2/popqa_longtail.jsonl"
    pro_c=get_context(rel_path,file_path)
    json_data=json.dumps(pro_c)
    with open("trimmed_evidences/L2/trimmed_evidences_popqa_0_30.json", "w") as file:
        file.write(json_data)
