import os
import glob
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from psbody.mesh import Mesh
import librosa    
from pathlib import Path

class VocaDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    count = 0
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                count += 1
                if count == 10:
                    break
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

                        

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if args.base_model_path is not None:
            v['vertice'] = v['vertice'][:,:3931*3]
            v['template'] = v['template'][:3931*3]
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args) # dataset | dict of three array: indicate 3 parts of subjects
    train_data = VocaDataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = VocaDataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = VocaDataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset

def load_vertices(path, scale = 1):
    return Mesh(filename=path).v * scale

def load_base_model(path, scale = 1):
    import glob

    if path == "":
        return None

    meshes = []
    for filename in sorted(glob.glob(os.path.join(path, "*.obj"))):
        meshes.append(load_vertices(filename, scale=scale))
    return meshes

class NPFADataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, opt):
        self.max_len = opt.max_len
        self.phase_data_root = os.path.join(opt.dataroot, opt.phase)
        self.vertice_dim = opt.vertice_dim
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        # Check if data path exists
        if not os.path.exists(self.phase_data_root):
            raise Exception("Data path does not exist: {}".format(self.phase_data_root))
        
        # Find all data files
        self.data_dirs = list(sorted(glob.glob(os.path.join(self.phase_data_root, "*/*.wav"))))

        # Check if data exists
        if len(self.data_dirs) == 0:
            raise Exception("No data found in: {}".format(self.phase_data_root))

        # Get the directory name of each data file
        self.data_dirs = list(set(map(lambda filepath: os.path.dirname(filepath), self.data_dirs)))

        # Load identities
        self.identity_dict = {}
        self.identity_dirs = list(sorted(glob.glob(os.path.join(opt.dataroot, 'identity', '*.npy'))))  
        if len(self.identity_dirs) == 0:
            raise Exception("No identities found in: {}".format(opt.dara_root))
        if opt.phase == 'valid':
            self.condition_index = opt.train_subjects.index(opt.condition_subject)
        identity_count = 0
        for identity_dir in self.identity_dirs:
            identity_name = Path(os.path.basename(identity_dir)).stem
            if NPFADataset.use_identity(opt, identity_name):
                self.identity_dict[identity_name] = (
                    identity_count if opt.phase != 'valid' else self.condition_index,
                    torch.FloatTensor(np.load(identity_dir)[0]).flatten(0), # (14062 * 3,)
                )
                identity_count += 1
        if identity_count == 0:
            raise Exception("No identities found for {} mode in: {}".format(opt.phase, opt.dara_root))

        if opt.phase == "train" or opt.phase == 'debug':
            if identity_count != len(opt.train_subjects):
                raise Exception(f"Number of identities found for {opt.phase} mode in {opt.dataroot} is not equal to the number of subjects specified in the option. Expect {opt.train_subjects}, but found identities: {self.identity_dict.keys()}")
        self.one_hot_labels = torch.eye(len(opt.train_subjects), dtype=torch.float) # (num_identities, num_identities)

        # Load data to memory
        self.data = []
        for data_dir in tqdm(self.data_dirs, desc='Loading data'):
            # Load audio
            audio_dir = os.path.join(data_dir, "audio.wav")
            speech_array, sampling_rate = librosa.load(audio_dir, sr=16000)
            wav_data = torch.FloatTensor(np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values))
            
            vertices_30fps_dirs = list(sorted(glob.glob(os.path.join(data_dir, '*/mesh_pred_all_vs_30fps.npy'))))
            vertices_60fps_dirs = list(sorted(glob.glob(os.path.join(data_dir, '*/mesh_pred_all_vs_60fps.npy'))))
            if len(vertices_30fps_dirs) + len(vertices_60fps_dirs) == 0:
                raise Exception("Find empty data directory at {}".format(data_dir))

            for vertices_dir in vertices_30fps_dirs + vertices_60fps_dirs:
                # Load template
                identity_name = os.path.basename(os.path.dirname(vertices_dir))
                if self.identity_dict.get(identity_name) is None:
                    continue
                identity_index, identity_neutral = self.identity_dict[identity_name]

                # Load vertices
                vertices_data = torch.FloatTensor(np.load(vertices_dir)).flatten(1) # (frame_num, 14062 * 3)
                # Skip every other frame if the fps is 60
                if "60fps" in os.path.basename(vertices_dir):
                    vertices_data = vertices_data[::2] 
                # Check if the number of vertices is correct
                if vertices_data.shape[1] != self.vertice_dim:
                    raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
                
                # vertices_data = (vertices_data - identity_neutral).clone.detach()
                data_item = {
                    'audio'   : wav_data.unsqueeze(0),
                    'vertice' : vertices_data.unsqueeze(0),
                    'template': identity_neutral.unsqueeze(0),
                    'one_hot' : self.one_hot_labels[identity_index].unsqueeze(0),
                    'identity_name': identity_name,
                    'audio_dir': audio_dir,
                    'vertices_dir': vertices_dir
                }
                self.data.append(data_item)
    
    def __getitem__(self, index):
        # Clip long audio and vertice
        frame_num = self.data[index]['vertice'].shape[1]
        if frame_num > self.max_len:
            start = random.randint(0, frame_num - self.max_len)
            clipped_vertice, clipped_audio = NPFADataset.clip(
                self.data[index]['vertice'],
                start,
                start + self.max_len,
                self.data[index]['audio']
            )
            return {
                **self.data[index],
                'audio'   : clipped_audio,
                'vertice' : clipped_vertice
            }
        return self.data[index]
    
    @property
    def identity_num(self):
        return len(self.identity_dict)

    def use_identity(opt, identity_name):
        if (opt.phase == 'train' or opt.phase == 'debug') and (identity_name in opt.train_subjects):
            return True
        if opt.phase == 'valid' and identity_name in opt.valid_subjects:
            return True
        return False

    def clip(vertice, start, end, audio):
        vertice_frame_num = vertice.shape[1]
        audio_frame_num = audio.shape[1]
        audio_start = start * audio_frame_num // vertice_frame_num
        audio_end = end * audio_frame_num // vertice_frame_num
        clipped_vertice = vertice[:,start:end]
        clipped_audio = audio[:,audio_start:audio_end]
        return clipped_vertice, clipped_audio
    
    def __len__(self):
        return len(self.data)