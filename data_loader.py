import os
import glob
import torch
from collections import defaultdict
from torch.utils import data
from util import util
import copy
import json
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from psbody.mesh import Mesh
import librosa    
from pathlib import Path

class OriginVocaDataset(data.Dataset):
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
    processor = Wav2Vec2Processor.from_pretrained("./facebook/wav2vec_processor")

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
                    data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit

                        

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

def get_voca_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args) # dataset | dict of three array: indicate 3 parts of subjects
    train_data = OriginVocaDataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = OriginVocaDataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = OriginVocaDataset(test_data,subjects_dict,"test")
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

class NPFABaseDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, opt):
        self.max_len = opt.max_len
        self.isTrain = opt.isTrain
        self.vertice_dim = opt.vertice_dim
        self.phase = opt.phase
        self.dataroot = opt.dataroot
        self.phase_data_root = os.path.join(self.dataroot, self.phase)
        self.speaker_info_dir = os.path.join(self.dataroot, "speaker.json")
        self.train_subjects:list = opt.train_subjects
        self.valid_subjects:list = opt.valid_subjects
        if self.phase == 'valid':
            self.condition_subject = opt.condition_subject
        self.processor = Wav2Vec2Processor.from_pretrained("./facebook/wav2vec_processor")
        self.data = []
        if self.isTrain:
            self.train_data = []
            self.test_data = []

        # Check if data path exists
        if not os.path.exists(self.phase_data_root):
            raise Exception("Data path does not exist: {}".format(self.phase_data_root))

        if os.path.exists(self.speaker_info_dir):
            with open(self.speaker_info_dir, "r") as speaker_info_file:
                self.speaker_info = json.load(speaker_info_file)
        else:
            self.speaker_info = None
    
    def initialize(self):
        raise NotImplementedError

    def find_speaker(self, source_name):
        if self.speaker_info is None:
            raise Exception("Speaker info does not exist: {}".format(self.speaker_info_dir))
        target_list = [item for item in self.speaker_info if item['name'] == source_name]
        # print(self.speaker_info)
        # print(target_list)
        # print(source_name)
        assert len(target_list) == 1
        return target_list[0]['speaker']
    
    def get_identity(self):
        identity_dict = {}
        identity_dirs = list(sorted(glob.glob(os.path.join(self.dataroot, 'identity', '*.npy'))))  
        if len(identity_dirs) == 0:
            raise Exception("No identities found in: {}".format(self.dataroot))
        if self.phase == 'valid':
            self.condition_index = self.train_subjects.index(self.condition_subject)
        identity_count = 0
        for identity_dir in identity_dirs:
            identity_name = Path(os.path.basename(identity_dir)).stem
            if self.use_identity(identity_name):
                identity_dict[identity_name] = (
                    identity_count if self.phase != 'valid' else self.condition_index,
                    torch.FloatTensor(np.load(identity_dir)[0]).flatten(0) / 100, # (14062 * 3,)
                )
                identity_count += 1
        if identity_count == 0:
            raise Exception("No identities found for {} mode in: {}".format(self.phase, self.dataroot))
        if self.phase == "train" or self.phase == 'debug':
            if identity_count != len(self.train_subjects):
                raise Exception(f"Number of identities found for {self.phase} mode in {self.dataroot} is not equal to the number of subjects specified in the option. Expect {self.train_subjects}, but found identities: {identity_dict.keys()}")
        return identity_dict
    
    def clip(vertice, start, end, audio):
        vertice_frame_num = vertice.shape[0]
        audio_frame_num = audio.shape[0]
        audio_start = start * audio_frame_num // vertice_frame_num
        audio_end = end * audio_frame_num // vertice_frame_num
        clipped_vertice = vertice[start:end]
        clipped_audio = audio[audio_start:audio_end]
        return clipped_vertice, clipped_audio
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Clip long audio and vertice
        frame_num = self.data[index]['vertice'].shape[0]
        if frame_num > self.max_len:
            if self.isTrain:
                start = random.randint(0, frame_num - self.max_len)
            else:
                start = 0
            clipped_vertice, clipped_audio = NPFAVerticeDataset.clip(
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
    
    def use_identity(self, identity_name):
        if (self.phase == 'train' or self.phase == 'debug') and (identity_name in self.train_subjects):
            return True
        if self.phase == 'valid' and identity_name in self.valid_subjects:
            return True
        return False
    

class NPFAVerticeDataset(NPFABaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.data_pattern = os.path.join(self.phase_data_root, "*/*.wav")
    
    def initialize(self):
        
        # Find all data files
        self.data_dirs = list(sorted(glob.glob(self.data_pattern)))

        # Check if data exists
        if len(self.data_dirs) == 0:
            raise Exception("No data found in: {}".format(self.phase_data_root))
        
        # Load identities
        self.identity_dict = self.get_identity()

        # Get the directory name of each data file
        self.data_dirs = list(set(map(lambda filepath: os.path.dirname(filepath), self.data_dirs)))
        self.one_hot_labels = torch.eye(len(self.train_subjects), dtype=torch.float) # (num_identities, num_identities)

        # Load data to memory
        self.data = []
        for data_dir in tqdm(self.data_dirs, desc='Loading data'):
            # Get the directory name of each data file
            audio_dir = os.path.join(data_dir, "audio.wav")
            source_name = os.path.basename(data_dir)
            # speaker_name = self.find_speaker(source_name)
            # if speaker_name is None:
            #     raise Exception("No speaker info found for {}".format(source_name))
            # else:
            #     if speaker_name != "chn_female_1":
            #         continue
            
            # Load audio
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
                vertices_data = torch.FloatTensor(np.load(vertices_dir)).flatten(1) / 100 # (frame_num, 14062 * 3)
                # Skip every other frame if the fps is 60
                if "60fps" in os.path.basename(vertices_dir):
                    vertices_data = vertices_data[::2] 
                # Check if the number of vertices is correct
                if vertices_data.shape[1] != self.vertice_dim:
                    raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
                
                # vertices_data = (vertices_data - identity_neutral).clone.detach()
                data_item = {
                    'audio'   : wav_data,
                    'vertice' : vertices_data,
                    'template': identity_neutral,
                    'one_hot' : self.one_hot_labels[identity_index],
                    'identity_name': identity_name,
                    'audio_dir': audio_dir,
                    'vertices_dir': vertices_dir,
                    'data_dir': data_dir,
                    'source_name': source_name
                }
                self.data.append(data_item)
        
        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class NPFABSDataset(NPFABaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.data_pattern = os.path.join(self.phase_data_root, "*NEU*/*.wav")
        # self.data_pattern = os.path.join(self.phase_data_root, "*neutral*/*.wav")
    
    def initialize(self):
        
        # Find all data files
        self.data_dirs = list(sorted(glob.glob(self.data_pattern)))

        # Check if data exists
        if len(self.data_dirs) == 0:
            raise Exception("No data found in: {}".format(self.phase_data_root))
        
        # Load identities
        self.identity_dict = self.get_identity()

        # Get the directory name of each data file
        self.data_dirs = list(set(map(lambda filepath: os.path.dirname(filepath), self.data_dirs)))
        self.one_hot_labels = torch.eye(len(self.train_subjects), dtype=torch.float) # (num_identities, num_identities)

        # Load data to memory
        self.data = []
        for data_dir in tqdm(self.data_dirs, desc='Loading data'):
            # Get the directory name of each data file
            audio_dir = os.path.join(data_dir, "audio.wav")
            source_name = os.path.basename(data_dir)
            # speaker_name = self.find_speaker(source_name)
            # if speaker_name is None:
            #     raise Exception("No speaker info found for {}".format(source_name))
            # else:
            #     if speaker_name != "chn_female_1":
            #         continue
            
            # Load audio
            speech_array, sampling_rate = librosa.load(audio_dir, sr=16000)
            wav_data = torch.FloatTensor(np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values))
            
            vertices_30fps_dirs = list(sorted(glob.glob(os.path.join(data_dir, '*/bs_30fps.npy'))))
            vertices_60fps_dirs = list(sorted(glob.glob(os.path.join(data_dir, '*/bs_60fps.npy'))))
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
                    'audio'   : wav_data,
                    'vertice' : vertices_data,
                    # 'template': identity_neutral,
                    'one_hot' : self.one_hot_labels[identity_index],
                    'identity_name': identity_name,
                    'audio_dir': audio_dir,
                    'vertices_dir': vertices_dir,
                    'data_dir': data_dir,
                    'source_name': source_name
                }
                self.data.append(data_item)
        
        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class NPFARavdessDataset(NPFAVerticeDataset):
    
    def __init__(self, opt):
        super().__init__(opt)
        self.data_pattern = os.path.join(self.phase_data_root, "*/*/*.wav")

class NoStyleDataset(NPFABaseDataset):
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        return {
            **data,
            "one_hot": self.one_hot_labels[0]
        }

class FilterFaceDataset(NPFABaseDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        return {
            **data,
            'vertice': data['vertice'][:,:28224],
            'template': data['template'][:28224]
        }


class NPFAVerticeNoStyleDataset(NoStyleDataset, NPFAVerticeDataset):
    pass

class NPFAEmbeddingDataset(NPFABaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
    
    def initialize(self):
        
        # Find all data files
        self.data_dirs = list(sorted(glob.glob(os.path.join(self.phase_data_root, "*/*.wav"))))

        # Check if data exists
        if len(self.data_dirs) == 0:
            raise Exception("No data found in: {}".format(self.phase_data_root))

        # Get the directory name of each data file
        self.data_dirs = list(set(map(lambda filepath: os.path.dirname(filepath), self.data_dirs)))

        # Load data to memory
        self.data = []
        for data_dir in tqdm(self.data_dirs, desc='Loading data'):
            # Load audio
            source_name = os.path.basename(data_dir)
            audio_dir = os.path.join(data_dir, "audio.wav")
            speech_array, sampling_rate = librosa.load(audio_dir, sr=16000)
            wav_data = torch.FloatTensor(np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values))
            
            vertices_30fps_dir = os.path.join(data_dir, 'exp_code_30fps.npy')
            vertices_60fps_dir = os.path.join(data_dir, 'exp_code_60fps.npy')
            
            vertices_dir = list(filter(lambda x: os.path.exists(x), [vertices_30fps_dir, vertices_60fps_dir]))
            if len(vertices_dir) != 1:
                raise Exception("Vertice data error at {}".format(data_dir))
            vertices_dir = vertices_dir[0]
            vertices_data = torch.FloatTensor(np.load(vertices_dir)).flatten(1) # (frame_num, 14062 * 3)
            # Load vertices
            # Skip every other frame if the fps is 60
            if "60fps" in os.path.basename(vertices_dir):
                vertices_data = vertices_data[::2] 
            # Check if the number of vertices is correct
            if vertices_data.shape[1] != self.vertice_dim:
                raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
            # vertices_data = (vertices_data - identity_neutral).clone.detach()
            data_item = {
                'audio'   : wav_data,
                'vertice' : vertices_data,
                'one_hot' : torch.ones((1), dtype=torch.float),
                'identity_name': 'Embedding',
                'audio_dir': audio_dir,
                'vertices_dir': vertices_dir,
                'data_dir': data_dir,
                'source_name': source_name
            }
            self.data.append(data_item)
        
        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class DecaDataset(NPFABaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.audio_dir = os.path.join(self.phase_data_root, "wav")
        self.template_dir = os.path.join(self.phase_data_root, "template")
    
    def initialize(self):
        
        # Find all data files
        self.data_dirs = list(sorted(glob.glob(os.path.join(self.phase_data_root, "npy/*.npy"))))

        # Check if data exists
        if len(self.data_dirs) == 0:
            raise Exception("No data found in: {}".format(self.phase_data_root))

        # Load data to memory
        self.data = []
        for vertices_dir in tqdm(self.data_dirs, desc='Loading data'):
            # Load audio
            source_name = os.path.basename(vertices_dir)
            audio_dir = os.path.join(self.audio_dir, source_name.replace("npy", "wav"))
            template_dir = os.path.join(self.template_dir, source_name)

            # Load audio
            speech_array, sampling_rate = librosa.load(audio_dir, sr=16000)
            wav_data = torch.FloatTensor(np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values))

            # Load template
            identity_neutral = torch.FloatTensor(np.load(template_dir)).flatten(0) # (15069)
            
            # Load vertices
            vertices_data = torch.FloatTensor(np.load(vertices_dir)).flatten(1) # (frame_num, 15069)

            # Check if the number of vertices is correct
            if vertices_data.shape[1] != self.vertice_dim:
                raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
            # vertices_data = (vertices_data - identity_neutral).clone.detach()
            data_item = {
                'audio'   : wav_data,
                'vertice' : vertices_data,
                'template': identity_neutral,
                'one_hot' : torch.ones((1), dtype=torch.float), # No styling currently
                'identity_name': 'Deca',
                'audio_dir': audio_dir,
                'vertices_dir': vertices_dir,
                'data_dir': vertices_dir,
                'source_name': source_name
            }
            self.data.append(data_item)
        
        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class VocaDataset(NPFABaseDataset):
    def __init__(self, opt):
        self.dataroot = opt.dataroot
        self.max_len = opt.max_len
        self.isTrain = opt.isTrain
        self.vertice_dim = opt.vertice_dim
        self.train_subjects:list = opt.train_subjects
        self.audio_path = os.path.join(self.dataroot, "wav")
        self.vertices_path = os.path.join(self.dataroot, "vertices_npy")
        self.processor = Wav2Vec2Processor.from_pretrained("./facebook/wav2vec_processor")
        self.template_file = os.path.join(self.dataroot, "templates.pkl")
        self.one_hot_labels = torch.eye(len(self.train_subjects), dtype=torch.float) # (num_identities, num_identities)
        self.data = []
        if self.isTrain:
            self.train_data = []
            self.test_data = []
    
    def initialize(self):

        # Load templates
        with open(self.template_file, 'rb') as fin:
            template_dict = pickle.load(fin,encoding='latin1')

        # Load audio
        for root, dirs, files in os.walk(self.audio_path):
            for audio_filename in tqdm(files):
                if audio_filename.endswith("wav"):
                    # Prepare files to load
                    wav_path = os.path.join(root, audio_filename)
                    source_name = Path(os.path.basename(audio_filename)).stem
                    subject_id = "_".join(source_name.split("_")[:-1])
                    vertices_dir = os.path.join(self.vertices_path, source_name + '.npy')
                    if subject_id not in self.train_subjects:
                        continue
                    if not os.path.exists(vertices_dir):
                        continue

                    # Load audio
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values)

                    # Load template
                    template = template_dict[subject_id]
                    # Load vertices
                    vertices_data = np.load(vertices_dir,allow_pickle=True)[::2,:]
                    if vertices_data.shape[1] != self.vertice_dim:
                        raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
                    
                    one_hot = self.one_hot_labels[self.train_subjects.index(subject_id)]

                    input_values, vertices_data, template, one_hot = util.to_FloatTensor(
                        input_values, vertices_data, template, one_hot
                    )

                    template = template.flatten(0)

                    data_item = {
                        'audio'   : input_values,
                        'vertice' : vertices_data,
                        'template': template,
                        'one_hot' : one_hot,
                        'identity_name': subject_id,
                        'audio_dir': wav_path,
                        'vertices_dir': vertices_dir,
                        'data_dir': wav_path,
                        'source_name': source_name
                    }
                    self.data.append(data_item)

        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class VocaBSDataset(VocaDataset):
    
    def __init__(self, opt):
        super().__init__(opt)
        self.vertices_path = os.path.join(self.dataroot, "bs_npy")

    def initialize(self):

        # Load audio
        for root, dirs, files in os.walk(self.audio_path):
            for audio_filename in tqdm(files):
                # if len(self.data) > 10:
                #     break
                if audio_filename.endswith("wav"):
                    # Prepare files to load
                    wav_path = os.path.join(root, audio_filename)
                    source_name = Path(os.path.basename(audio_filename)).stem
                    subject_id = "_".join(source_name.split("_")[:-1])
                    vertices_dir = os.path.join(self.vertices_path, source_name + '.npy')
                    if subject_id not in self.train_subjects:
                        continue
                    if not os.path.exists(vertices_dir):
                        continue

                    # Load audio
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values)

                    # Load template
                    # template = template_dict[subject_id]

                    # Load vertices
                    vertices_data = np.load(vertices_dir,allow_pickle=True)[::2,:]
                    if vertices_data.shape[1] != self.vertice_dim:
                        raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
                    
                    one_hot = self.one_hot_labels[self.train_subjects.index(subject_id)]

                    input_values, vertices_data, one_hot = util.to_FloatTensor(
                        input_values, vertices_data, one_hot
                    )

                    # template = template.flatten(0)

                    data_item = {
                        'audio'   : input_values,
                        'vertice' : vertices_data,
                        # 'template': template,
                        'one_hot' : one_hot,
                        'identity_name': subject_id,
                        'audio_dir': wav_path,
                        'vertices_dir': vertices_dir,
                        'data_dir': wav_path,
                        'source_name': source_name,
                    }
                    self.data.append(data_item)

        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class VocaPCADataset(VocaBSDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.vertices_path = os.path.join(self.dataroot, "pca_npy")
        

class USCVocaDataset(FilterFaceDataset, VocaDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.vertice_dim = 42186 # For checking USC vertice_dim
        self.vertices_path = os.path.join(self.dataroot, "vertices_npy_usc")
        self.template_file = os.path.join(self.dataroot, "usc_templates.pkl")


# By C3 MRO, __get_item__ in VocaDataset will be override by NoStyleDataset
class VocaNoStyleDataset(NoStyleDataset, VocaDataset):
    pass


class VocaDataset2(NPFABaseDataset):
    def __init__(self, opt):
        self.dataroot = opt.dataroot
        self.max_len = opt.max_len
        self.isTrain = opt.isTrain
        self.vertice_dim = opt.vertice_dim
        self.train_subjects:list = opt.train_subjects
        self.test_subjects:list = opt.test_subjects
        self.audio_path = os.path.join(self.dataroot, "wav")
        self.vertices_path = os.path.join(self.dataroot, "vertices_npy")
        self.processor = Wav2Vec2Processor.from_pretrained("./facebook/wav2vec_processor")
        self.template_file = os.path.join(self.dataroot, "templates.pkl")
        self.one_hot_labels = torch.eye(len(self.train_subjects), dtype=torch.float) # (num_identities, num_identities)
        self.data = []
        if self.isTrain:
            self.train_data = []
            self.test_data = []
    
    def initialize(self):

        # Load templates
        with open(self.template_file, 'rb') as fin:
            template_dict = pickle.load(fin,encoding='latin1')
        # Load audio
        for root, dirs, files in os.walk(self.audio_path):
            for audio_filename in tqdm(files):
                if audio_filename.endswith("wav"):
                    # Prepare files to load
                    wav_path = os.path.join(root, audio_filename)
                    source_name = Path(os.path.basename(audio_filename)).stem
                    subject_id = "_".join(source_name.split("_")[:-1])
                    vertices_dir = os.path.join(self.vertices_path, source_name + '.npy')
                    
                    if (subject_id not in self.train_subjects) and (subject_id not in self.test_subjects):
                        continue
                    if not os.path.exists(vertices_dir):
                        continue

                    # Load audio
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values)

                    # Load template
                    template = template_dict[subject_id]
                    # Load vertices
                    vertices_data = np.load(vertices_dir,allow_pickle=True)[::2,:]
                    if vertices_data.shape[1] != self.vertice_dim:
                        raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
                    
                    one_hot = self.one_hot_labels[self.train_subjects.index(subject_id)] if subject_id in self.train_subjects else self.one_hot_labels[self.test_subjects.index(subject_id)]

                    input_values, vertices_data, template, one_hot = util.to_FloatTensor(
                        input_values, vertices_data, template, one_hot
                    )

                    template = template.flatten(0)

                    data_item = {
                        'audio'   : input_values,
                        'vertice' : vertices_data,
                        'template': template,
                        'one_hot' : one_hot,
                        'identity_name': subject_id,
                        'audio_dir': wav_path,
                        'vertices_dir': vertices_dir,
                        'data_dir': wav_path,
                        'source_name': source_name
                    }
                    # self.data.append(data_item)
                    if self.isTrain:
                        if subject_id in self.train_subjects:
                            self.train_data.append(data_item)
                        else:
                            self.test_data.append(data_item)
                    else:
                        self.data.append(data_item)
        # print(len(self.train_data))
        # print(len(self.test_data))
        # print(len(self.data))
        # Split to two set when training
        # if self.isTrain:
        #     self.train_data, self.test_data = NPFABaseDataset.split(self.data)


class MixDataset(NPFABaseDataset):
    
    def __init__(self, opt):
        super().__init__(opt)
        with open(opt.mix_config, 'r') as f:
            self.subdatasets:list[dict] = json.load(f)
    
    def initialize(self):
        # Load identities
        self.identity_dict = self.get_identity()

        self.one_hot_labels = torch.eye(len(self.train_subjects), dtype=torch.float) # (num_identities, num_identities)

        # Load data to memory
        for subdataset in self.subdatasets:
            subdataset_name = subdataset['name']
            subdataset_glob = subdataset['glob']
            subdataset_count = 0
            subdataset_dir = os.path.join(self.phase_data_root, subdataset_glob)
            print(f"Loading {subdataset_name} at path {subdataset_dir}")
            
            data_dirs = sorted(list(glob.glob(subdataset_dir)))
            data_dirs = list(set(map(lambda filepath: os.path.dirname(filepath), data_dirs)))

            for data_dir in tqdm(data_dirs, desc='Loading data'):
                # Get the directory name of each data file
                audio_dir = os.path.join(data_dir, "audio.wav")
                source_name = os.path.basename(data_dir)
                # Load audio
                speech_array, sampling_rate = librosa.load(audio_dir, sr=16000)
                # Normalize
                max_peak = np.max(np.abs(speech_array))
                # -6db
                ratio = 0.5 / max_peak
                normalized_speech = speech_array * ratio

                wav_data = torch.FloatTensor(np.squeeze(self.processor(normalized_speech,sampling_rate=16000).input_values))
                
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
                    vertices_data = torch.FloatTensor(np.load(vertices_dir)).flatten(1) / 100 # (frame_num, 14062 * 3)
                    vertices_data = vertices_data[:,:28224]
                    # Skip every other frame if the fps is 60
                    if "60fps" in os.path.basename(vertices_dir):
                        vertices_data = vertices_data[::2] 
                    # Check if the number of vertices is correct
                    if vertices_data.shape[1] != self.vertice_dim:
                        raise Exception(f"Number of vertices in {vertices_dir} is not correct, expect {self.vertice_dim}, but read {vertices_data.shape[1]}")
                    
                    # vertices_data = (vertices_data - identity_neutral).clone.detach()
                    data_item = {
                        'audio'   : wav_data,
                        'vertice' : vertices_data,
                        'template': identity_neutral,
                        'one_hot' : self.one_hot_labels[identity_index],
                        'identity_name': identity_name,
                        'audio_dir': audio_dir,
                        'vertices_dir': vertices_dir,
                        'data_dir': data_dir,
                        'source_name': source_name
                    }
                    self.data.append(data_item)
                    subdataset_count += 1
            print(f"Loaded {subdataset_count} items in {subdataset_name} ")
        
        # Split to two set when training
        if self.isTrain:
            self.train_data, self.test_data = util.split_dataset(self)

class MixVocaDataset(NPFABaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        opt.dataroot = opt.mix_dataroot
        opt.train_subjects = opt.mix_train_subjects
        # opt.vertice_dim=28224
        self.mixdataset = MixDataset(opt)
        opt.dataroot = opt.voca_dataroot
        opt.train_subjects = opt.voca_train_subjects
        self.uscvocadataset = USCVocaDataset(opt)
        opt.train_subjects = opt.voca_train_subjects + opt.mix_train_subjects
        opt.vertice_dim=28224

    def initialize(self):
        self.mixdataset.initialize()
        self.uscvocadataset.initialize()
        for data_item in self.mixdataset.data:
            data_item['one_hot'] = torch.nn.functional.pad(data_item['one_hot'],pad=(0,len(self.uscvocadataset.data[0]['one_hot'])))
            data_item['vertice'] = data_item['vertice'][:,:28224]
            data_item['template'] = data_item['template'][:28224]
        for data_item in self.uscvocadataset.data:
            data_item['one_hot'] = torch.nn.functional.pad(data_item['one_hot'],pad=(1,0))
            
        self.data = self.mixdataset.data + self.uscvocadataset.data
        
        if self.isTrain:
            mix_train_data, mix_test_data = util.split_dataset(self.mixdataset)
            uscvoca_train_data, uscvoca_test_data = util.split_dataset(self.uscvocadataset)
            self.train_data = mix_train_data + uscvoca_train_data
            self.test_data = mix_test_data + uscvoca_test_data

    def get_identity(self):
        identity_dict = {}
        mix_identity_dict = self.mixdataset.get_identity()
        uscvoca_identity_dict = self.uscvocadataset.get_identity()
        identity_dict = mix_identity_dict + uscvoca_identity_dict

def get_dataset(opt):
    # Load dataset by option
    Dataset = globals()[opt.dataset]
    # Check if it is NPFADataset
    assert issubclass(Dataset, NPFABaseDataset)
    # Instance of NPFADataset
    dataset = Dataset(opt)
    dataset.initialize()
    if opt.isTrain:
        return (
            data.DataLoader(
                dataset=dataset.train_data,
                batch_size=1,
                shuffle=True,
                generator=util.global_generator()
            ),
            data.DataLoader(
                dataset=dataset.test_data,
                batch_size=1,
                shuffle=False
            )
        )
    return data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )