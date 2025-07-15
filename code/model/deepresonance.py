import logging
import os
from typing import List
import re
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from .layers import *


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class DeepResonanceModel(nn.Module):
    """LoRA for LLaMa model"""

    def __init__(self, **args):
        super(DeepResonanceModel, self).__init__()
        self.args = args

        self.max_length = args['max_length']
        self.device = torch.cuda.current_device()
        self.stage = args['stage']
        print('args max_length', args['max_length'])
        
        imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt',
                                           self.args['imagebind_version'])
        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = \
            imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

        self.vicuna_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'vicuna_ckpt',
                                             self.args['vicuna_version'])
        print(f'Initializing language decoder from {self.vicuna_ckpt_path} ...')

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path)
        if self.args.get('freeze_lm'):
            print("Freezing the LLaMa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        else:
            print("Instruct tuning the LLaMa ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        print('Language decoder initialized.')

        # use the new trained tokenizer
        tokenizer_path = self.vicuna_ckpt_path
        print(f'Initializing tokenizer from {tokenizer_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        # self.llama_tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self._add_image_token()
        self._add_video_token()
        self._add_audio_token()
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print('Tokenizer initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )
       	if self.args.get('freeze_input_proj'):
            for param in self.llama_proj.parameters():
                param.requires_grad = False

        if self.args.get('prellmfusion'):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.llama_model.config.hidden_size,
                nhead=self.llama_model.config.num_attention_heads,
                dim_feedforward=self.llama_model.config.intermediate_size,
                dropout=self.args['prellmfusion_dropout'],
                batch_first=True
            )
            self.preencoding = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.args['num_prellmfusion_layers']
            )           

        self.input_embeddings = self.llama_model.get_input_embeddings()

    def _add_image_token(self):
        # Add an image token for loss masking (and visualization) purposes.
        self.llama_tokenizer.add_tokens(["<Img>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["</Img>"])  # add special image token to tokenizer

    def _add_video_token(self):
        self.llama_tokenizer.add_tokens({"<Vid>"})  # add special video token to tokenizer
        self.llama_tokenizer.add_tokens({"</Vid>"})  # add special video token to tokenizer

    def _add_audio_token(self):
        self.llama_tokenizer.add_tokens({"<Aud>"})  # add special audio token to tokenizer
        self.llama_tokenizer.add_tokens({"</Aud>"})  # add special audio token to tokenizer

    def encode_video(self, video_paths="", 
            video_inputs=None, do_tokenize=True, enable_emb_seq=False):
        visual_encoder_device = next(self.visual_encoder.parameters()).device
        if do_tokenize:
            inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        else:
            inputs = {ModalityType.VISION: torch.stack(video_inputs, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            if not (self.args['imagebind_embs_seq'] and enable_emb_seq):
                video_embeds = embeddings[ModalityType.VISION]  # bsz x 1024
            else:
                video_embeds = embeddings[f'{ModalityType.VISION}_original']  # bsz x seq x 1024
        inputs_llama = self.llama_proj(video_embeds)  # bsz (x seq) x llama_size
        if not (self.args['imagebind_embs_seq'] and enable_emb_seq):
            inputs_llama = inputs_llama.unsqueeze(1)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths="",
            audio_inputs=None, do_tokenize=True, enable_emb_seq=False):
        
        if do_tokenize:
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(
                audio_paths, self.device)}
        else:
            inputs = {ModalityType.AUDIO: torch.stack(audio_inputs, dim=0).to(self.device)}
        
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            if not (self.args['imagebind_embs_seq'] and enable_emb_seq):
                audio_embeds = embeddings[ModalityType.AUDIO]  # bsz x 1024
            else:
                audio_embeds = embeddings[f'{ModalityType.AUDIO}_original']  # bsz x seq x 1024

        inputs_llama = self.llama_proj(audio_embeds)  # bsz (x seq) x llama_size
        if not (self.args['imagebind_embs_seq'] and enable_emb_seq):
            inputs_llama = inputs_llama.unsqueeze(1)
        atts_llama = torch.ones(inputs_llama.size()[:-1],
                                dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths="",
            image_inputs=None, do_tokenize=True, enable_emb_seq=False):
        visual_encoder_device = next(self.visual_encoder.parameters()).device
        if do_tokenize:
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        else:
            inputs = {ModalityType.VISION: torch.stack(image_inputs, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            if not (self.args['imagebind_embs_seq'] and enable_emb_seq):
                image_embeds = embeddings[ModalityType.VISION]  # bsz x 1024
            else:
                image_embeds = embeddings[f'{ModalityType.VISION}_original']  # bsz x seq x 1024
        inputs_llama = self.llama_proj(image_embeds)  # bsz (x seq) x llama_size
        if not (self.args['imagebind_embs_seq'] and enable_emb_seq):
            inputs_llama = inputs_llama.unsqueeze(1)           
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def _train_with_mode(self, source_embeds=None, targets=None, attention_mask=None):

        outputs = self.llama_model(
            inputs_embeds=source_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        return loss, gen_acc, torch.zeros_like(loss)

    def _enc_align_training_stage_1(self, inputs):
        """
        In the stage 1: training the encoding-side alignment via image/video/audio caption tasks
        modality: the input modality for each caption task, it could be 'image', 'video' or 'audio'.
        """
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'AnyToText':
            source_embeds, targets, attention_mask = self._prepare_batch_mixed_embedding(inputs)
            loss, gen_acc, _ = self._train_with_mode(
                    source_embeds=source_embeds, targets=targets, attention_mask=attention_mask)
            return loss, gen_acc
        else:
            raise NotImplementedError

    def _tokenize_strings(self, text, batch_size, need_bos=False):
        text_tokens = self.llama_tokenizer(
                text, add_special_tokens=False, return_tensors='pt').to(self.device)
        bos_embeds = None
        if need_bos:
            bos = torch.ones([batch_size, 1],
                         dtype=text_tokens.input_ids.dtype,
                         device=text_tokens.input_ids.device
                         ) * self.llama_tokenizer.bos_token_id  # bsz x 1
        if self.args['freeze_lm']:
            text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
            if need_bos:
                bos_embeds = self.llama_model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        else:
            text_embeds = self.llama_model.model.model.embed_tokens(text_tokens.input_ids)
            if need_bos:
                bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        return bos_embeds, text_embeds

    def _prepare_one_mixed_embedding(
            self, input_text, instruction_text, mm_embeds, output_text):
        instruction_text = '### Human: ' + instruction_text + '\n### Assistant: '
        output_text += '\n###'
        input_embeds = []
        instruction_embeds = []
        target_embeds = []
        targets = None
        split_text = re.split(r'(<Audio>|<Image>|<Video>)', input_text)
        mm_embeds_id = 0
        if input_text:
            for i, text in enumerate(split_text):
                if text == '':
                    continue
                if text == '<Image>':
                    input_embeds.append(
                            self.encode_image(
                                image_inputs=[mm_embeds[mm_embeds_id]],
                                do_tokenize=False,
                                enable_emb_seq=True
                            )[0]
                    )
                    mm_embeds_id += 1
                elif text == '<Audio>':
                    input_embeds.append(
                            self.encode_audio(
                                audio_inputs=[mm_embeds[mm_embeds_id]],
                                do_tokenize=False,
                                enable_emb_seq=True
                            )[0]
                    )
                    mm_embeds_id += 1
                elif text == '<Video>':
                    input_embeds.append(
                            self.encode_video(
                                video_inputs=[mm_embeds[mm_embeds_id]],
                                do_tokenize=False,
                                enable_emb_seq=True
                            )[0]
                    )
                    mm_embeds_id += 1
                else:
                    need_bos = True if i == 0 else False
                    bos_embeds, text_embeds = self._tokenize_strings(text, batch_size=1, need_bos=need_bos)
                    if need_bos:
                        input_embeds.append(bos_embeds)
                    input_embeds.append(text_embeds)
            input_embeds = torch.cat(input_embeds, dim=1)  # 1 x (1+s1) x embed_dim

        _, instruction_embed = self._tokenize_strings(instruction_text, batch_size=1)
        instruction_embeds.append(instruction_embed)
        instruction_embeds = torch.cat(instruction_embeds, dim=1)  # 1 x s2 x embed_dim

        if self.training:
            _, target_embed = self._tokenize_strings(output_text, batch_size=1)
            target_embeds.append(target_embed)
            target_embeds = torch.cat(target_embeds, dim=1)  # 1 x s3 x embed_dim

            target_ids = self.llama_tokenizer(output_text, add_special_tokens=False).input_ids
            target_ids = torch.LongTensor(target_ids).expand(1, -1).to(self.device)

            # create targets
            if input_text:
                empty_targets = (
                    torch.ones([1, input_embeds.shape[1] + instruction_embeds.shape[1]],
                               dtype=torch.long).to(self.device).fill_(-100))  # 1 x (1+s1+s2)
            else:
                empty_targets = (
                    torch.ones([1, instruction_embeds.shape[1]],
                               dtype=torch.long).to(self.device).fill_(-100))  # 1 x s2
            targets = torch.cat([empty_targets, target_ids], dim=1)  # 1 x ((1+s1)+s2+s3)
        return input_embeds, instruction_embeds, target_embeds, targets

    def _padding_embeds(self, embeds: List[torch.Tensor], dtype=torch.long):
        """
        input:
        embeds: List[torch.Tensor(1, length, embed_dim)]
        
        output:
        padded_embeds: torch.Tensor(bsz, max_len, embed_dim)
        """
        padded_embeds = []
        padded_masks = []
        max_len = max([x.shape[1] for x in embeds])
        for embed in embeds:
            pad = torch.ones([1, max_len - embed.shape[1]],
                             dtype=dtype,
                             device=self.device
                            ) * self.llama_tokenizer.pad_token_id  # bsz x 1
            if self.args['freeze_lm']:
                pad_embeds = self.llama_model.model.embed_tokens(pad)
            else:
                pad_embeds = self.llama_model.model.model.embed_tokens(pad)
            padded_embed = torch.cat((embed, pad_embeds), dim=1)
            padded_embeds.append(padded_embed)
            padded_mask_prefix = torch.zeros([1, embed.shape[1]],
                                            device=self.device).bool()
            padded_mask_suffix = torch.ones([1, max_len - embed.shape[1]],
                                            device=self.device).bool()
            padded_mask = torch.cat((padded_mask_prefix, padded_mask_suffix), dim=1)
            padded_masks.append(padded_mask)
        if max_len < self.max_length:
            return (torch.cat(padded_embeds, dim=0),
                    torch.cat(padded_masks, dim=0))
        else:
            return (torch.cat(padded_embeds, dim=0)[:, : self.max_length, :],
                    torch.cat(padded_masks, dim=0)[:, : self.max_length])
    
    def _padding_targets(self, targets):
        padded_targets = []
        attention_masks = []
        max_len = max([x.shape[1] for x in targets])
        for target in targets:
            empty_target = (torch.ones(
                [1, max_len - target.shape[1]],
                dtype=target.dtype).to(self.device).fill_(-100))
            atts_prefix = torch.ones(
                    [1, target.shape[1]], dtype=target.dtype).to(self.device)
            atts_suffix = torch.zeros(
                    [1, max_len - target.shape[1]], dtype=target.dtype).to(self.device)
            padded_target = torch.cat((target, empty_target), dim=1)  # 1 x max_len
            attention_mask = torch.cat([atts_prefix, atts_suffix], dim=1).to(self.device)
            assert attention_mask.size() == padded_target.size()
            padded_targets.append(padded_target)
            attention_masks.append(attention_mask)

        if max_len < self.max_length:
            return (torch.cat(padded_targets, dim=0),
                    torch.cat(attention_masks, dim=0))
        else:
            return (torch.cat(padded_targets, dim=0)[:, : self.max_length],
                    torch.cat(attention_masks, dim=0)[:, : self.max_length])

    def _prepare_batch_mixed_embedding(self, inputs):
        input_embeds = []
        instruction_embeds = []
        output_embeds = []
        source_embeds = []
        targets = []
        batch_size = len(inputs["inputs"])
        attention_mask = None
        for i in range(batch_size):
            if inputs["inputs"][i] == "":
                inputs["inputs"][i] = " "
            (single_input_embeds,
             single_instruction_embeds,
             single_output_embeds,
             single_targets
            ) = self._prepare_one_mixed_embedding(
                            inputs["inputs"][i],
                            inputs["instructions"][i],
                            inputs["mm_inputs"][i],
                            inputs["outputs"][i])
            input_embeds.append(single_input_embeds)
            instruction_embeds.append(single_instruction_embeds)
            output_embeds.append(single_output_embeds)
            targets.append(single_targets)
        
        if self.training:
            targets, attention_mask = self._padding_targets(targets)

        dtype = targets[0].dtype if self.training else torch.long
        if self.args["prellmfusion"] and inputs["inputs"][0]:
            input_embeds, input_mask = self._padding_embeds(input_embeds, dtype)
            input_embeds = self.preencoding(input_embeds, src_key_padding_mask=input_mask)
        
        for i in range(batch_size):
            if inputs["inputs"][0]:
                if self.args["prellmfusion"]:
                    seq_len = input_embeds.shape[1] - sum(input_mask[i])
                    single_input_embeds = input_embeds[i][: seq_len, :].view(1, seq_len, -1)
                else:
                    single_input_embeds = input_embeds[i]
            single_instruction_embeds = instruction_embeds[i]
            single_output_embeds = output_embeds[i]
            if not self.training:
                source_embeds.append(torch.cat(
                    (single_input_embeds, single_instruction_embeds), dim=1))
            elif inputs["inputs"][0]:
                source_embeds.append(torch.cat(
                    (single_input_embeds, single_instruction_embeds, single_output_embeds), dim=1))
            else:
                source_embeds.append(torch.cat(
                    (single_instruction_embeds, single_output_embeds), dim=1))

        source_embeds, _ = self._padding_embeds(source_embeds, dtype)
        return source_embeds, targets, attention_mask

    def _instruction_tuning_stage_2(self, inputs):
        """
        In the stage 2: instruction-following training via the instruction dataset.
        """
        loss = 0
        gen_acc = 0
        mse_loss = []

        dataset_type = inputs['dataset_types'][0]
        
        if dataset_type == 'AnyToText':
            source_embeds, targets, attention_mask = self._prepare_batch_mixed_embedding(inputs)
            loss, gen_acc, _ = self._train_with_mode(
                    source_embeds=source_embeds, targets=targets, attention_mask=attention_mask)
        else:
            raise NotImplementedError
        return loss, gen_acc, mse_loss

    def forward(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None

        if self.stage == 1:
            loss, gen_acc = self._enc_align_training_stage_1(inputs)
        elif self.stage == 2:
            loss, gen_acc, mse_loss = self._instruction_tuning_stage_2(inputs)
        else:
            raise NotImplementedError(f"stage {self.stage} is not implemented, now it only support [1, 2]")

        return loss, gen_acc, mse_loss

    def generate_tokens_embeddings(self, inputs, input_embeds):
        """
        inputs: dict
        input_embeds: tensor
        return: the output tokens index
        """
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        return outputs.sequences

    def generate(self, inputs):
        """
            inputs = {
                'inputs': List[str],
                'instructions': List[str],
                'mm_names': List[List[str]],
                'mm_paths': List[List[str]],
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature, Used to modulate logit distribution.
                'stops_id': the default value is [[835], [2277, 29937]] the stop token is '###', which has two types of tokenization ways, [835] and [2277, 29937]
            }
        """

        mm_inputs_all = []
        
        for i in range(len(inputs["mm_names"])):
            mm_inputs = []
            for name, filepath in zip(inputs["mm_names"][i], inputs["mm_paths"][i]):
                path = os.path.join(inputs["mm_root_path"], filepath)
                if name == "video":
                    mm_input = data.load_and_transform_video_data([path], self.device)
                if name == "audio":
                    mm_input = data.load_and_transform_audio_data([path], self.device)
                if name == "image":
                    mm_input = data.load_and_transform_vision_data([path], self.device)
                mm_inputs.append(mm_input[0])
            mm_inputs_all.append(mm_inputs)
 
        inputs["mm_inputs"] = mm_inputs_all

        source_embeds, _, _ = self._prepare_batch_mixed_embedding(inputs)
        generated_ids = self.generate_tokens_embeddings(inputs, source_embeds)

        return_outputs = []

        caption = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return_outputs.append(caption)

        return return_outputs

