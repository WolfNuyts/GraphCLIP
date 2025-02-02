import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch.utils.data import Dataset
from utils.dataset_utils import add_master_node_with_bidirectional_edges, add_master_node_with_incoming_edges, enc_img_fn, process_adversarial_dataset, unzip_file
import os.path as osp
import json
from tqdm import tqdm
import logging
import open_clip
from PIL import Image
from torch_geometric.data import Data
from pathlib import Path
from scripts import create_adversarial_attributes_dataset
from transformers import AutoProcessor, CLIPVisionModelWithProjection

# Embeds text with CLIP
def dict_to_pyg_graph(d, img_enc, txt_enc, image_id_to_path, metadata, coco_val_ids, use_long_rel_enc, enc_img=True):
    # y: [1, num_img_features]
    # x: [num_nodes, num_txt_features]
    id_to_idx = {}
    # TODO: deal with multiple object names?
    n_obj_nodes = len(d['objects'])
    x = txt_enc([obj['names'][0] for obj in d['objects']])
    attrs = []
    attr_to_x = []
    for idx, o in enumerate(d['objects']):
        for attr in o.get('attributes', []):
            attrs.append(attr)
            attr_to_x.append(idx)
    n_attrs = len(attrs)
    if n_attrs == 0:
        attrs = torch.zeros((0, 2), dtype=torch.int64)
    else:
        attrs = txt_enc(attrs)
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.int64)
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
    attrs_edge_index = torch.zeros((2, n_attrs), dtype=torch.int64)
    for attr_idx, x_idx in enumerate(attr_to_x):
        attrs_edge_index[:, attr_idx] = torch.tensor([attr_idx+n_obj_nodes, x_idx])
    # edge_attr: [num_edges, num_txt_features]
    if len(d['relationships']) == 0:
        edge_attr = torch.zeros((0, 2), dtype=torch.int64)
    else:
        rel_txts = []
        for rel in d['relationships']:
            if use_long_rel_enc:
                subj_txt = d['objects'][id_to_idx[rel['subject_id']]]['names'][0]
                obj_txt = d['objects'][id_to_idx[rel['object_id']]]['names'][0]
                rel_txt = rel['predicate']
                compound_txt = " ".join([subj_txt, rel_txt, obj_txt])
            else:
                compound_txt = rel['predicate']
            rel_txts.append(compound_txt)
        edge_attr = txt_enc(rel_txts)
    attrs_edge_attr = -3*torch.ones((n_attrs, 2), dtype=torch.int64)
    
    coco_id = metadata['coco_id'] if metadata['coco_id'] is not None else -1
    image_id = d['image_id']
    in_coco_val = coco_id in coco_val_ids
    data = Data(x=torch.cat([x, attrs]),
        edge_attr=torch.cat([edge_attr, attrs_edge_attr]),
        edge_index=torch.cat([edge_index, attrs_edge_index], dim=1),
        obj_nodes=torch.arange(0, n_obj_nodes),
        attr_nodes=torch.arange(n_obj_nodes, n_obj_nodes + n_attrs),
        coco_id=torch.tensor([coco_id], dtype=torch.long),
        image_id=torch.tensor([image_id], dtype=torch.long),
        in_coco_val=torch.tensor([in_coco_val], dtype=torch.bool)
    )
    if enc_img:
        # TODO: normalize?
        y = img_enc(image_id_to_path[d['image_id']])
        data.y = y
    return data


# Embeds text with CLIP
def dict_to_pyg_graphs(d, img_enc, txt_enc, image_id_to_path, metadata, coco_val_ids, use_long_rel_enc):
    # y: [1, num_img_features]
    y = img_enc(image_id_to_path[d['image_id']])
    id_to_idx = {}
    # x: [num_nodes, num_txt_features]
    x = txt_enc([obj['names'][0] for obj in d['objects']])
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.int64)
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
    coco_id = metadata['coco_id'] if metadata['coco_id'] is not None else -1
    image_id = d['image_id']
    in_coco_val = coco_id in coco_val_ids
    datas = []
    rel_txts = []
    for rel in d['relationships']:
        if use_long_rel_enc:
            subj_txt = d['objects'][id_to_idx[rel['subject_id']]]['names'][0]
            obj_txt = d['objects'][id_to_idx[rel['object_id']]]['names'][0]
            rel_txt = rel['predicate']
            compound_txt = " ".join([subj_txt, rel_txt, obj_txt])
        else:
            compound_txt = rel['predicate']
        rel_txts.append(compound_txt)
    edge_attr = txt_enc(rel_txts)
    for idx, rel in enumerate(d['relationships']):
        data = Data(x=x[edge_index[:, idx]],
            edge_attr=edge_attr[idx].reshape(1,-1),
            edge_index=torch.tensor([0,1]).reshape(2,1),
            y=y,
            obj_nodes=torch.tensor([0, 1]),
            attr_nodes=torch.tensor([], dtype=torch.int64),
            coco_id=torch.tensor([coco_id], dtype=torch.long),
            image_id=torch.tensor([image_id], dtype=torch.long),
            in_coco_val=torch.tensor([in_coco_val], dtype=torch.bool)
        )
        datas.append(data)
    return datas

class VisualGenome(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, enc_cfg=None, n_samples="all", scene_graphs_filename="scene_graphs.json", use_long_rel_enc=None, one_sample_per_edge = False, dataset=None, enc_img=True):
        self.enc_cfg = enc_cfg
        self.n_samples = n_samples
        self.scene_graphs_filename = scene_graphs_filename
        self.use_long_rel_enc = use_long_rel_enc
        self.enc_img = enc_img
        if not type(transform) == list:
            transform = [transform]
        cached_img_enc_path = None
        def transform_fn(x):
            for t in transform:
                if t == "add_master_node_with_bidirectional_edges":
                    x = add_master_node_with_bidirectional_edges(x)
                elif t == "add_master_node_with_incoming_edges":
                    x = add_master_node_with_incoming_edges(x)
                elif t == "enc_img":
                    x = enc_img_fn(x, cached_img_enc_path)
                elif t is None:
                    x = x
                else:
                    raise Exception(f"Unknown transform {transform}.")
            return x
        if pre_transform == "add_master_node_with_bidirectional_edges":
            pre_transform_fn = add_master_node_with_bidirectional_edges
        elif pre_transform == "add_master_node_with_incoming_edges":
            pre_transform_fn = add_master_node_with_incoming_edges
        elif pre_transform is None:
            pre_transform_fn = lambda x: x
        else:
            raise Exception(f"Unknown pre-transform {pre_transform}.")
        tokenizer = open_clip.get_tokenizer(model_name=self.enc_cfg["model_name"])
        def clip_embedding_txt_enc(txts):
           with torch.no_grad():
                tokens = tokenizer(txts)
                tokens[tokens == 49407] = 0
                tokens = tokens[:, 1:3]
                out = tokens.cpu()
                return out  
        self.clip_embedding_txt_enc = clip_embedding_txt_enc
        self.one_sample_per_edge = one_sample_per_edge
        super().__init__(root, transform_fn, pre_transform_fn, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        cached_img_enc_path = osp.join(self.processed_dir, f"{self.enc_cfg['model_name']}_{self.enc_cfg['pretrained']}_{self.n_samples}_use_sd_img_enc{self.enc_cfg.get('use_sd_img_enc', False)}_img_enc_cache.pt")

    @property
    def raw_file_names(self):
        return ['scene_graphs.json.zip', 'images.zip', 'images2.zip', 'image_data.json.zip', 'annotations_trainval2017.zip', 'realistic_adversarial_samples.json',
        'realistic_adversarial_samples2.json',
        'realistic_adversarial_attributes_gt_1.json',
        'realistic_adversarial_attributes_gt_2.json',
        'realistic_adversarial_attributes_gt_3.json',]

    @property
    def processed_file_names(self):
        return [f"data_{self.scene_graphs_filename}_{self.n_samples}_{self.enc_cfg['model_name']}_{self.enc_cfg['pretrained']}_use_clip_latents={self.enc_cfg['use_clip_latents']}_use_long_rel_enc={self.use_long_rel_enc}_one_sample_per_edge={self.one_sample_per_edge}_enc_img={self.enc_img}_coco_annotated_with_attributes_6.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        def download_and_unzip_if_not_exist(url, idx):
            path = osp.join(self.raw_dir, self.raw_file_names[idx])
            if not osp.isfile(path):
                download_url(url, self.raw_dir)
                unzip_file(self.raw_paths[idx], self.raw_dir)
            else:
                print(f"{path} already exists, skipping download.")
        download_and_unzip_if_not_exist("http://visualgenome.org/static/data/dataset/scene_graphs.json.zip", 0)
        download_and_unzip_if_not_exist("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", 1)
        download_and_unzip_if_not_exist("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", 2)
        download_and_unzip_if_not_exist("http://visualgenome.org/static/data/dataset/image_data.json.zip", 3)
        download_and_unzip_if_not_exist("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 4)
        create_adversarial_attributes_dataset.main(type="1")
        create_adversarial_attributes_dataset.main(type="2")
        create_adversarial_attributes_dataset.main(type="3")

    def process(self):
        logging.info("Processing adversarial dataset...")
        process_adversarial_dataset(in_dir=self.raw_dir, in_fname=self.raw_file_names[5], out_fname_ext="")
        process_adversarial_dataset(in_dir=self.raw_dir, in_fname=self.raw_file_names[6], out_fname_ext="2")
        # Read data into huge `Data` list.
        logging.info("Loading scene graph JSON file...")
        with open(osp.join(self.raw_dir, self.scene_graphs_filename), 'r') as f:
            scene_graphs_dict = json.load(f)
        logging.info("Loading image data JSON file...")
        with open(osp.join(self.raw_dir, "image_data.json"), 'r') as f:
            image_data_dict = json.load(f)
        if not self.n_samples == "all":
            scene_graphs_dict = scene_graphs_dict[:self.n_samples]
            image_data_dict = image_data_dict[:self.n_samples]
        logging.info("Processing scene graphs into PyG graphs...")
        
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=self.enc_cfg["model_name"], pretrained=self.enc_cfg["pretrained"], device=self.enc_cfg["device"])
        tokenizer = open_clip.get_tokenizer(model_name=self.enc_cfg["model_name"])

        image_id_to_path = dict()
        for dir in [Path(self.raw_dir)/"VG_100K", Path(self.raw_dir)/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                image_id_to_path[img_id] = str(path)

        with open(osp.join(self.raw_dir, "annotations", "instances_val2017.json"), 'r') as f:
            mscoco_val_dict = json.load(f)
            coco_val_ids = [int(Path(o['file_name']).stem) for o in mscoco_val_dict['images']]
        cached_img_enc_path = osp.join(self.processed_dir, f"{self.enc_cfg['model_name']}_{self.enc_cfg['pretrained']}_{self.n_samples}_use_sd_img_enc{self.enc_cfg.get('use_sd_img_enc', False)}_img_enc_cache.pt")
        if self.enc_cfg.get('use_sd_img_enc', False):
            model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.enc_cfg['device'])
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            def enc_img(path):
                inputs = processor(images=Image.open(path), return_tensors="pt").to(self.enc_cfg['device'])
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds.cpu()
                return image_embeds
        else:
            def enc_img(img_path):
                 return model.encode_image(preprocess(Image.open(img_path)).unsqueeze(0).to(self.enc_cfg["device"])).cpu()
            
        if not osp.exists(cached_img_enc_path):
            logging.info("Embedding images with CLIP...")
            cached_img_enc = dict()  
            for d in tqdm(scene_graphs_dict):
                img_path = image_id_to_path[d['image_id']]
                with torch.no_grad():
                    img_enc = enc_img(img_path)
                    cached_img_enc[img_path] = img_enc
            torch.save(cached_img_enc, cached_img_enc_path)
        cached_img_enc = torch.load(cached_img_enc_path)
        def img_enc_fn(img_path):
            return cached_img_enc[img_path]
        def clip_latent_txt_enc_fn(txts):
            with torch.no_grad():
                return model.encode_text(tokenizer(txts).to(self.enc_cfg["device"])).cpu()
        txt_enc_fn = clip_latent_txt_enc_fn if self.enc_cfg["use_clip_latents"] else self.clip_embedding_txt_enc
        logging.info("Producing PyG graphs...")
        if self.one_sample_per_edge:
            data_lists = [dict_to_pyg_graphs(d, img_enc_fn, txt_enc_fn, image_id_to_path, metadata, coco_val_ids, self.use_long_rel_enc)
                        for d, metadata in tqdm(zip(scene_graphs_dict, image_data_dict))]
            logging.info("Listing PyG graphs...")
            data_list = [d for dd in data_lists for d in dd]
        else:
            metadatas = {i['image_id']: i for i in image_data_dict}
            data_list = [dict_to_pyg_graph(d, img_enc_fn, txt_enc_fn, image_id_to_path, metadatas[d['image_id']], coco_val_ids, self.use_long_rel_enc, enc_img=self.enc_img)
                        for d in tqdm(scene_graphs_dict)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logging.info("Collating PyG graphs...")
        data, slices = self.collate(data_list)

        logging.info("Saving PyG graphs...")
        torch.save((data, slices), self.processed_paths[0])


class VisualGenomeAdversarial(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset_gt = VisualGenome(*args, **kwargs,  scene_graphs_filename="scene_graphs_gt.json")
        self.dataset_adv = VisualGenome(*args, **kwargs, scene_graphs_filename="scene_graphs_adv.json")

    def __len__(self):
        return len(self.dataset_gt)

    def __getitem__(self, idx):
        return {
            "gt": self.dataset_gt[idx],
            "adv": self.dataset_adv[idx],
        }

class VisualGenomeAdversarial2(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset_gt = VisualGenome(*args, **kwargs,  scene_graphs_filename="scene_graphs_gt2.json")
        self.dataset_adv = VisualGenome(*args, **kwargs, scene_graphs_filename="scene_graphs_adv2.json")

    def __len__(self):
        return len(self.dataset_gt)

    def __getitem__(self, idx):
        return {
            "gt": self.dataset_gt[idx],
            "adv": self.dataset_adv[idx],
        }

class VisualGenomeAdversarialAttr(Dataset):
    def __init__(self, *args, **kwargs):
        self.dataset_gt = VisualGenome(*args, **kwargs,  scene_graphs_filename="realistic_adversarial_attributes_gt_accepted_pruned_and.json")
        self.dataset_adv = VisualGenome(*args, **kwargs, scene_graphs_filename="realistic_adversarial_attributes_adv_accepted_pruned_and.json")

    def __len__(self):
        return len(self.dataset_gt)

    def __getitem__(self, idx):
        return {
            "gt": self.dataset_gt[idx],
            "adv": self.dataset_adv[idx],
        }

class VisualGenomeAdversarialText(Dataset):
    def __init__(self, root):
        self.captions_gt = []
        self.captions_adv = []
        self.img_paths = []
        with open(osp.join(root, 'raw', 'realistic_adversarial_samples.json'), 'r') as f:
            adv_data = json.load(f)
        img_id_to_path = dict()
        for dir in [Path(root)/"raw"/"VG_100K", Path(root)/"raw"/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                img_id_to_path[img_id] = str(path)
        for v in adv_data:
                caption_gt = v['subj_name'] + " " + v['original_predicate'] + " " + v['obj_name']
                caption_adv = v['subj_name'] + " " + v['adv_predicate'] + " " + v['obj_name']
                img_path = img_id_to_path[v['image_id']]
                self.captions_gt.append(caption_gt)
                self.captions_adv.append(caption_adv)
                self.img_paths.append(img_path)

class VisualGenomeAdversarialText2(Dataset):
    def __init__(self, root):
        self.captions_gt = []
        self.captions_adv = []
        self.img_paths = []
        with open(osp.join(root, 'raw', 'realistic_adversarial_samples2.json'), 'r') as f:
            adv_data = json.load(f)
        img_id_to_path = dict()
        for dir in [Path(root)/"raw"/"VG_100K", Path(root)/"raw"/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                img_id_to_path[img_id] = str(path)
        for v in adv_data:
                caption_gt = v['subj_name'] + " " + v['original_predicate'] + " " + v['obj_name']
                caption_adv = v['subj_name'] + " " + v['adv_predicate'] + " " + v['obj_name']
                img_path = img_id_to_path[v['image_id']]
                self.captions_gt.append(caption_gt)
                self.captions_adv.append(caption_adv)
                self.img_paths.append(img_path)

class VisualGenomeAdversarialAttrText(Dataset):
    def __init__(self, root):
        self.captions_gt = []
        self.captions_adv = []
        self.img_paths = []
        with open('datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted_pruned.json', 'r') as f:
            data = json.load(f)
        img_id_to_path = dict()
        for dir in [Path(root)/"raw"/"VG_100K", Path(root)/"raw"/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                img_id_to_path[img_id] = str(path)
        for v in data:
                if len(v['relationships']) > 0:
                    rel = v['relationships'][0]
                else:
                    rel = {
                        'predicate': "and",
                        'subject_id': v['objects'][0]['object_id'],
                        'object_id': v['objects'][1]['object_id'],
                    }
                rel_txt = rel['predicate']

                entity_id_to_txt_gt = {e['object_id']: ','.join(e['attributes']) + " " + e['names'][0] for e in v['objects']}
                subj_txt_gt = entity_id_to_txt_gt[rel['subject_id']]
                obj_txt_gt = entity_id_to_txt_gt[rel['object_id']]
                caption_gt = subj_txt_gt + " " + rel_txt + " " + obj_txt_gt
                
                v['objects'][0]['attributes'], v['objects'][1]['attributes'] = v['objects'][1]['attributes'], v['objects'][0]['attributes']
                entity_id_to_txt_adv = {e['object_id']: ','.join(e['attributes']) + " " + e['names'][0] for e in v['objects']}
                subj_txt_adv = entity_id_to_txt_adv[rel['subject_id']]
                obj_txt_adv = entity_id_to_txt_adv[rel['object_id']]
                caption_adv = subj_txt_adv + " " + rel_txt + " " + obj_txt_adv
                
                img_path = img_id_to_path[v['image_id']]
                self.captions_gt.append(caption_gt)
                self.captions_adv.append(caption_adv)
                self.img_paths.append(img_path)
        print("self.captions_gt", self.captions_gt[-10:])
        print("self.captions_adv", self.captions_adv[-10:])

class VisualGenomeAdversarialAttrText2(Dataset):
    def __init__(self, root):
        self.captions_gt = []
        self.captions_adv = []
        self.img_paths = []
        with open('datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted_pruned.json', 'r') as f:
            data = json.load(f)
        img_id_to_path = dict()
        for dir in [Path(root)/"raw"/"VG_100K", Path(root)/"raw"/"VG_100K_2"]:
            pathlist = dir.glob('*.jpg')
            for path in pathlist:
                img_id = int(path.stem)
                img_id_to_path[img_id] = str(path)
        for v in data:
                if len(v['relationships']) > 0:
                    rel = v['relationships'][0]
                else:
                    rel = {
                        'predicate': "and",
                        'subject_id': v['objects'][0]['object_id'],
                        'object_id': v['objects'][1]['object_id'],
                    }

                entity_id_to_txt_gt = {e['object_id']: ','.join(e['attributes']) + " " + e['names'][0] for e in v['objects']}
                subj_txt_gt = entity_id_to_txt_gt[rel['subject_id']]
                obj_txt_gt = entity_id_to_txt_gt[rel['object_id']]
                caption_gt = [subj_txt_gt, obj_txt_gt]
                
                v['objects'][0]['attributes'], v['objects'][1]['attributes'] = v['objects'][1]['attributes'], v['objects'][0]['attributes']
                entity_id_to_txt_adv = {e['object_id']: ','.join(e['attributes']) + " " + e['names'][0] for e in v['objects']}
                subj_txt_adv = entity_id_to_txt_adv[rel['subject_id']]
                obj_txt_adv = entity_id_to_txt_adv[rel['object_id']]
                caption_adv = [subj_txt_adv, obj_txt_adv]
                
                img_path = img_id_to_path[v['image_id']]
                self.captions_gt.append(caption_gt)
                self.captions_adv.append(caption_adv)
                self.img_paths.append(img_path)
        print("self.captions_gt", self.captions_gt[-10:])
        print("self.captions_adv", self.captions_adv[-10:])