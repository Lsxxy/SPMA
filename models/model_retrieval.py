import torch
from models import HarMABase, load_pretrained_harma
import torch.nn.functional as F

import torch

import open_clip
from torch import nn



class SPMA(HarMABase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_affil_loss=False)
        self.config = config
        self.use_affil_loss = config['use_affil_loss']
        self.use_triplet_loss = config['use_triplet_loss']
        self.use_prompt = config['use_prompt']
        self.create_and_load_pretrained(config)
        self.align_before = False

    def create_and_load_pretrained(self, config):
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32")
        if self.use_prompt == True:
            self.pre_prompt_model,_,_ = open_clip.create_model_and_transforms("ViT-B/32",pre_prompt=self.use_prompt)

    def get_vis_emb(self, image, idx=None, label=None):
        if self.use_prompt:
            img_preprompt = self.pre_prompt_model.encode_image(image,normalize=True)
        if self.config['SPMA']:
            if self.align_before:
                img_emb,feas_vis = self.model.encode_image(image,normalize=True)
                return img_emb,feas_vis
            else:
                if self.use_prompt:
                    img_emb = self.model.encode_image(image,img_preprompt=img_preprompt,normalize=True)
                else:
                    img_emb = self.model.encode_image(image,normalize=True)
            return img_emb
        
    def get_txt_emb(self, text_ids,idx=None, label=None):
        if self.use_prompt:
            text_preprompt = self.pre_prompt_model.encode_text(text_ids,normalize=True)
        if self.config['SPMA']:
            if self.align_before:
                txt_emb,feas_txt = self.model.encode_text(text_ids,normalize=True)
                return txt_emb,feas_txt
            else:
                if self.use_prompt:
                    txt_emb = self.model.encode_text(text_ids,text_preprompt=text_preprompt,normalize=True)
                else:
                    txt_emb = self.model.encode_text(text_ids,normalize=True)
            return txt_emb
        



    def forward(self, image, text_ids, idx=None, label=None):
        ## Baseline(Swin-T+Bert-B)
        if self.config['SPMA']:
            if self.align_before:
                img_emb,feas_vis = self.get_vis_emb(image)
                txt_emb,feas_txt = self.get_txt_emb(text_ids)
            else:
                img_emb = self.get_vis_emb(image)
                txt_emb=self.get_txt_emb(text_ids)
        else:
            img_emb= self.get_vision_fusion_embeds(image, self.config)
            txt_emb = self.get_text_fusion_embeds(text_ids, self.config)

        if self.use_affil_loss:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_affil = self.get_affil_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            return loss_contr, loss_affil
        elif self.use_triplet_loss:
            loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
            return loss_triplet
        else:
            loss_before_contr = []
            if self.align_before:
                for i in range(len(feas_vis)):
                    loss_contr = self.get_contr_loss(feas_vis[i],feas_txt[i], idx=idx, label=label, config=self.config)
                    loss_before_contr.append(loss_contr)
                total_loss_before = sum(loss_before_contr)
            loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            if self.align_before:
                return loss_contr,loss_triplet,total_loss_before
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            #TODO new loss
            return loss_contr,loss_triplet,None



        



        