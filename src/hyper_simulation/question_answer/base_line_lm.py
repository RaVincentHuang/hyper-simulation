# model_name = 
# mode = 
# top_n = 10
# prompt_name = prompt_no_input_retrieval
# batch_size = 5
# choices = 
# download_dir = .cache
# rank_file
# input_file = "data/eval_data/popqa_longtail.jsonl"
# metric = "match"
# result_fp = data/baseline_result/popqa
# task = qa
# dtype = half
# query_focus = False
# enquery_less = True
# enquery_file_path = 
import scipy.stats as stats
import argparse
from hyper_simulation.question_answer.vmdit.relation import same_sentences_with_llm
from hyper_simulation.question_answer.vmdit.utils import load_file, PROMPT_DICT, save_file_jsonl, preprocess_input_data, postprocess_answers_closed, TASK_INST
from hyper_simulation.question_answer.vmdit.metrics import metric_max_over_ground_truths, exact_match_score, match
import json
import ast
from tqdm import tqdm
from hyper_simulation.llm.chat_completion import get_invoke, get_generate
import numpy as np
# from hyper_simulation.llm.prompt.vmdit import PROMPT_DICT

def get_index_number(l, n, mode):
    result=[]
    if mode == 'average':
        ave=l/n
        i=0
        while i<l:
            result.append(i)
            i+=ave
    elif mode == 'guass':
        lower, upper = 0, l
        mu, sigma = 0,1
        # X表示含有最大最小值约束的正态分布
        # N表示不含最大最小值约束的正态分布
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
        a = X.rvs(n)  # 取其中的b个数，赋值给a；a为array类型
        result=[int(i) for i in a]
        while 1:
            x=result[0]
            result=result[1:]
            if x in result:
                x+=1
                result.append(x)
            else:
                result.append(x)
            set_result=set(result)
            if len(result)==len(set_result):
                break
    elif mode=='top':
        result=range(5)
    print(result)
    return result

def llm_evi_ans(evidences, query):
    result=[]
    #result.append(evidences[0])
    for evi in evidences:
        try:
            rel= same_sentences_with_llm(evi,query).upper()
            print(rel)
            if 'False' in rel:
                continue
            else:
                result.append(evi)
        except:
            result.append(evi)
    return result

from langchain_ollama import ChatOllama
model = ChatOllama(model="qwen2.5:72b", temperature=0.8, top_p=0.95)
def call_model(prompts, max_new_tokens=50):
    preds = get_generate(prompts, model)
    preds = [pred.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds

def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--input_file', type=str, required=False)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="vanilla")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--metric', type=str)
    parser.add_argument('--top_n', type=int, default=1,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--choices",  type=str, default=None,
                        help="space-separated answer candidates")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--query_focus', action="store_true", help='get llm to decide if the query is similar with ctxs.')
    parser.add_argument('--enquery_less', action="store_true", help='reduce the times of getting llm.')
    parser.add_argument('--enquery_file_path', type=str, default=None, help='the file of enquery.')
    parser.add_argument('--rank_file', type=str, default=None, help='the file of rank.')

    args = parser.parse_args()

    #args.model_name='ZhipuAI/chatglm3-6b'
    #args.input_file='../data/eval_data/health_claims_processed.jsonl'
    args.input_file = 'data/eval_data/popqa_longtail.jsonl'
    args.max_new_tokens=100
    args.metric='match'
    args.result_fp='data/baseline_result/triviaqa'
    args.task='qa'
    args.prompt_name="prompt_no_input_retrieval"
    args.mode='retrieval'
    args.top_n=10
    args.dtype='half'
    args.query_focus=False
    # args.enquery_less=True
    # args.enquery_file_path = 'src_vmdit/trimmed_evidences/HNSW/trimmed_evidences_popqa_0.json'
    #args.rank_file='./popqa1_hclu_rank.json'

    input_data = load_file(args.input_file)

    """model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0")
    tokenizer_enquery = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_enquery = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).cuda()
    model_enquery = model_enquery.eval()"""

    # For baseline scripts, we simply load pre-retrieved documents from `retrieval_file` option.
    if args.mode == "retrieval":
        l = 20  # 总共检索范围
        n=args.top_n  #检索个数
        mode = 'guass'  # 正态分布
        index_number = get_index_number(l, n,mode)
        if args.retrieval_file is not None:
            retrieval_data = load_file(args.retrieval_file)
            id2retrieval = {}
            for id, item in enumerate(retrieval_data):
                if not args.query_focus:
                    item["ctxs"]=item["ctxs"][:args.top_n]
                else:
                    item["ctxs"] = [item["ctxs"][i] for i in index_number]
                if "id" not in item:
                    id2retrieval[id] = item["ctxs"]
                else:
                    id2retrieval[item["id"]] = item["ctxs"]
            i=0
            if not args.enquery_less:
                for id, item in enumerate(input_data):
                    retrieval_result = id2retrieval[id if "id" not in item else item["id"]]
                    evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                    if args.query_focus:
                        evidences = llm_evi_ans(evidences,item['question'])
                        print("finish:",i)
                        i+=1
                    else:
                        evidences=evidences[:args.top_n]
                    item["paragraph"] = "\n".join(evidences)

            else:
                fin = open(args.enquery_file_path, "r")
                data = fin.read()
                evidences = json.loads(data)
                for id, item in enumerate(input_data):
                    evidence = evidences[id][:args.top_n]
                    item["paragraph"] = "\n".join(evidence)

        else:
            if args.rank_file is None and not args.enquery_less:
                i=0
                for id, item in enumerate(input_data):
                    if not args.query_focus:
                        item["ctxs"]=item["ctxs"][:args.top_n]
                        retrieval_result = item["ctxs"]
                    else:
                        retrieval_result = [item["ctxs"][i] for i in index_number]
                        #retrieval_result = item["ctxs"][:5]
                        #retrieval_result = [item["ctxs"][0],item["ctxs"][2],item["ctxs"][6],item["ctxs"][12],item["ctxs"][19]]
                    #retrieval_result = item["ctxs"][:args.top_n]
                    evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                    if args.query_focus:
                        evidences = llm_evi_ans(evidences, item['question'])
                        print("finish:",i)
                        i+=1
                    else:
                        evidences=evidences[:args.top_n]
                    item["paragraph"] = "\n".join(evidences)
                    #print(item['paragraph'])
            elif args.rank_file is not None and not args.enquery_less:
                fin=open(args.rank_file, "r")
                data=fin.read()
                ctxs = json.loads(data)
                i=0
                for id, item in enumerate(input_data):
                    #print(evidences[id])
                    #print(len(ctxs[id]))
                    evidences = [ctxs[id][i] for i in index_number]
                    if args.query_focus:
                        evidences_llm = llm_evi_ans(evidences, item['question'])
                        print("finish:", i)
                        i += 1
                    else:
                        evidences_llm=evidences
                    item["paragraph"] = "\n".join(evidences_llm)
            elif args.rank_file is None and args.enquery_less:
                fin = open(args.enquery_file_path, "r")
                data = fin.read()
                evidences = json.loads(data)
                for id, item in enumerate(input_data):
                    evidence = evidences[id][:args.top_n]
                    item["paragraph"] = "\n".join(evidence)
            else:
                print("wrong !")


    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task=='fever' or args.task=='arc_c':
            item["instruction"]=TASK_INST[args.task]+"\n\n### Input:\n" +item["instruction"]
        #else:
         #   item["instruction"] = preprocess_input_data(item, args.task)



    final_results = []
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch
        ]
    
        preds, _ = call_model(processed_batch, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            #item["output"] = pred
            final_results.append(item)

    if len(input_data) % args.batch_size > 0:
        l=len(input_data) // args.batch_size
        batch = input_data[l*args.batch_size:]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]
    
        preds, _ = call_model(processed_batch, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            #item["output"] = pred
            final_results.append(item)

    for item in input_data:
        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(
                exact_match_score, item["output"], item["golds"])
        elif args.metric == "accuracy":
            metric_result = 1.0 if item["golds"][0] in item["output"] else 0.0
        elif args.metric == "match":
            metric_result = match(item["output"], item["golds"])
        else:
            raise NotImplementedError
        item["metric_result"] = metric_result

    print("overall result: {0}".format(
        np.mean([item["metric_result"] for item in input_data])))

    if args.task == "factscore":
        processed_item = []
        for item in input_data:
            processed_item.append(item)
        save_file_jsonl(processed_item, args.result_fp)
    else:
        save_file_jsonl(input_data, args.result_fp)


if __name__ == "__main__":
    main()
