import os
from model.deepresonance import DeepResonanceModel
import torch
import json
from config import *
import argparse
from tqdm import tqdm
from collections import defaultdict


dataset_path = {
        "musicqa": "../data/musicqa_test.json",
        "musiccaps": "../data/musiccaps_test.json",
        "music4way_musiccaps": "../data/music4way_musiccaps_test.json",
        "music4way_mi2t": "../data/music4way_mi2t_test.json",
        "music4way_mv2t": "../data/music4way_mv2t_test.json",
        "music4way_any2t": "../data/music4way_any2t_test.json",
        "musicnet": "../data/musicnet_test.json",
        "gtzan": "../data/gtzan_test.json",
        "mtg": "../data/mtg_test.json",
        "musicinstruct_long": "../data/musicinstruct_long_test.json",
        "musicinstruct_short": "../data/musicinstruct_short_test.json",
        }
mm_root_path = {
        "musicqa": "../data/MusicQA/audios",
        "musiccaps": "../data/MusicCaps/eval_audios",
        "music4way_musiccaps": "../data/music4way/test/audios",
        "music4way_mi2t": "../data/music4way/test",
        "music4way_mv2t": "../data/music4way/test",
        "music4way_any2t": "../data/music4way/test",
        "musicnet": "../data/MusicNet/eval_audios",
        "gtzan": "../data/GTZAN/eval_audios",
        "mtg": "../data/MTG/eval_audios",
        "musicinstruct_long": "../data/MusicInstruct/eval_audios",
        "musicinstruct_short": "../data/MusicInstruct/eval_audios",
        }
 
def parser_args():
    parser = argparse.ArgumentParser(description='eval parameters')
    parser.add_argument('--model', type=str, default='deepresonance')
    parser.add_argument('--ckpt_path', type=str, default='../ckpt/delta_ckpt/deepresonance/7b_tiva_v0')
    parser.add_argument('--server', type=str, default='local')
    parser.add_argument('--mode', type=str, default='test', help='train or test or validation')

    # inputs
    parser.add_argument('--dataset', type=str, default='musiccaps')

    # outputs
    parser.add_argument('--result_file_name', type=str, default='')

    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--max_output_length', type=int, default=512)  # the maximum output sequence length for LLMs
    parser.add_argument('--prellmfusion', action='store_true')  # using preencoding Transformer for the SonicMPT model
    parser.add_argument('--imagebind_embs_seq', action='store_true')  # not using reduced ImageBind embedding
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--topp', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=0.1)
    return parser.parse_args()


class DeepResonancePredict(object):
    def __init__(self, args):
        self.max_len = args['max_length']
        self.max_output_len = args['max_output_length']
        args.update(load_config(args))
        args["max_length"] = self.max_len
        model = DeepResonanceModel(**args)
        delta_ckpt = torch.load(
                os.path.join(args['ckpt_path'], 'pytorch_model.pt'),
                map_location=torch.device('cuda'))
        # print(delta_ckpt)
        model.load_state_dict(delta_ckpt, strict=False)
        self.model = model.eval().half().cuda()
        print(f'[!] LLM initialized.')

    def predict(
            self,
            inputs,
            max_tgt_len=512,
            top_p=1.0,
            temperature=0.1,
            stops_id=None,
    ):
        
        inputs.update({
            'top_p': top_p,
            'temperature': temperature,
            'max_tgt_len': max_tgt_len,
            'stops_id': stops_id,
        })

        response = self.model.generate(inputs)

        return response


def nextgpt_musicqa_predict(nextgpt_engine, inputs, top_p, temperature):
    """Override Chatbot.postprocess"""
    max_tgt_length = nextgpt_engine.max_output_len

    output = nextgpt_engine.predict(
                     inputs=inputs,
                     max_tgt_len=max_tgt_length,
                     top_p=top_p,
                     temperature=temperature,
                     stops_id=[[835]],
                     )

    for i in output:
        answer = i.split("\n###")[0]
    print(answer)
    return answer


if __name__ == '__main__':
    args = parser_args()
    args = vars(args)
    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    nextgpt_engine = DeepResonancePredict(args)
    
    dataset = args["dataset"]
    eval_data = json.load(open(dataset_path[dataset]))
    
    results = defaultdict(lambda: {})
    fileset = set()
    out_file_path = f"../eval_outputs/" + args["result_file_name"] + ".json"
    
    if not os.path.exists(f"../eval_outputs/{dataset}"):
        os.makedirs(f"../eval_outputs/{dataset}")
    
    if os.path.exists(out_file_path):
        results = defaultdict(lambda: {}, json.load(open(out_file_path, 'r')))
        fileset = set(results.keys())
    
    print(f"Already Completed: {sum([len(x) for x in results.values()])}")
    
    count = 0
    for row in tqdm(eval_data):
        if row["id"] in fileset and row["instruction"] in results[row["id"]]:
            continue
        print(row["instruction"])
        inputs = {"inputs": [row["input"]],
                  "instructions": [row["instruction"]],
                  "mm_names": [row["mm_names"]],
                  "mm_paths": [row["mm_paths"]],
                  "mm_root_path": mm_root_path[dataset],
                  "outputs": [""],
                 }
        results[row["id"]][row["instruction"]] = nextgpt_musicqa_predict(
                nextgpt_engine, inputs, args["topp"], args["temp"])
        count += 1
        if count % 10 == 0:
            with open(out_file_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    with open(out_file_path, 'w') as f:
        json.dump(results, f, indent=2)

