import sys,csv,random,io
sys.path.append('.')
sys.path.append('..')
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Normalize
from open_clip_train.data import *

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
DATA_ROOT_PATH = '/scratch/data/um_mri/'  # Change this to where you stored your MRI data
CT_ROOT_PATH = '/scratch/shared_data/data/um_ct/' # Change this to where you stored your CT data
JSON_ROOT_PATH = '/scratch/localjson' # Change this to where you stored your MRI JSON files
CT_JSON_ROOT_PATH = '/nfs/turbo/CTjsons' # Change this to where you stored your CT JSON files


# Information about a single study.
class MrStudyInfo(object):
    def __init__(self, root, key, value):
        self.series = np.array([os.path.join(root, key, series, 'img.pt') for series in value['series'].keys()])
        self.report = np.array(value['report'])
        if len(self.report) == 0:
            print(key)
        self.label = 0 if self.report[0] == "no significant abnormalities." or ('study is unremarkable' in self.report[0].lower()) else 1 # normalcy label
        self.key = key
        self.serienames = [s for s in value['series']]
        if 'subid' in value:
            self.subid = value['subid']

    def is_normal(self):
        return self.label == 0
    
    def get_report(self, shuffle):
        if shuffle:
            if hasattr(self,'altreport'):
                if random.random() < 0.5:
                    return 'This study shows: ' + ' '.join(np.random.permutation(self.altreport).tolist())
            return 'This study shows: ' + ' '.join(np.random.permutation(self.report).tolist())
        else:
            return 'This study shows: ' + ' '.join(self.report.tolist())

    def get_series(self, shuffle):
        if shuffle:
            return np.random.permutation(self.series).tolist()
        else:
            return self.series.tolist()


# process listed report into individual text items
def processreport(s):
    lines = s.split('\n')
    ret = []
    for line in lines:
        if ('. ' not in line) or len(line) < 5:
            continue
        idx = line.index('. ')
        ret.append(line[idx+2:])
    return ret





# this function converts the serie name text to LongTensor
def chartovec(s):
    if len(s) > 109:
        s = s[0:109]
    ret = [chartovechelp(c) for c in s.lower()]
    ret.append(46)
    ret.extend([0]*(110-len(ret)))
    return torch.LongTensor(ret)
char_to_index = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz_+-0123456789.*(),")}
def chartovechelp(c):
    return char_to_index[c]+1 if c in char_to_index else 45

# Dataset class for MRI/CT studies for ItemizedCLIP training
class ChunkedDataset(Dataset):
    def __init__(
        self, 
        input_filename, # the input json data
        transform=None, # visual transforms
        tokenizer=None, # text tokenizer
        num_series=None, # limits number of sequences to use per study during training
        report_replacer = None, # path to csv file for report replacement file
        use_seriename = False, # whether to use serie name encodings
        alt_report = False, # whether to use alternative report when shuffling
        is_ct = False,  # whether this is CT data or MRI data
        report_cap = 7  # maximum number of report items to use
    ):
    
        logging.debug(f'Loading json data from {input_filename}.')
        if is_ct:
            jrp = CT_JSON_ROOT_PATH
        else:
            jrp = JSON_ROOT_PATH
        with open(os.path.join(jrp, input_filename + '.json'), 'r') as file:
            studies = json.load(file)
        if is_ct:
            drp = CT_ROOT_PATH
        else:
            drp = DATA_ROOT_PATH
        self.studies = [MrStudyInfo(root=os.path.join(drp, input_filename), key=key, value=value) for key, value in studies.items()]
        if len(self.studies) == 991: # do this so that the val set is a multiple of 8 and prevents bugs from distributed training with 8 GPUs
            print('extending to 992')
            self.studies.append(self.studies[-1])
        if report_replacer is not None: # load in report replacement file and replace reports accordingly
            repdict = {}
            reader = csv.reader(open(report_replacer))
            nocount = 0
            for row in reader:
                if len(row) == 2:
                    repdict[row[0]] = row[1]
            for study in self.studies:
                if study.key in repdict:
                    if alt_report:
                        study.altreport = study.report
                    study.report = np.array(processreport(repdict[study.key]))
                else:
                    nocount += 1
            logging.info('replacer no-count: '+str(nocount))
        
        self.num_series = num_series
        self.is_train = num_series is not None
        self.report_cap = report_cap
        
        logging.debug('Done loading data.')

        self.transform = transform
        self.tokenizer = tokenizer
        self.use_seriename = use_seriename
        self.alt_report = alt_report
        self.rng = random.Random() # unseeded rng
    
    def __len__(self):
        return len(self.studies)
    
    def getrandom(self):
        return self.__getitem__(self.rng.randint(0,len(self)-1))
    
    def __getitem__(self, idx):
        study = self.studies[idx]

        # get is_normal
        is_normal = torch.as_tensor([study.is_normal()]).long()

        origreport = study.report

        # get report
        if self.is_train:
            thereport = np.random.permutation(origreport)
        else:
            thereport = origreport

        # pad or truncate report items to self.report_cap
        if len(thereport) > self.report_cap:
            textmask = torch.ones(self.report_cap)
            thereport = thereport[0:self.report_cap]
        elif len(thereport) < self.report_cap:  
            thereport = thereport.tolist() + ["Placeholder"] * (self.report_cap - len(thereport))
            textmask = torch.zeros(self.report_cap)
            textmask[0:len(origreport)] = 1
        else:
            textmask = torch.ones(self.report_cap)

        assert len(thereport) == self.report_cap

        report = self.tokenizer(thereport)
        

        # get sequences
        series = study.get_series(shuffle=self.is_train)
        repeats = -(-self.num_series // len(series)) if self.num_series else -(-10 // len(series)) # [hard code]: replace 10 to # series during training.
        series *= repeats
        series = series[:self.num_series] if self.num_series else series
        
        # load-in sequences
        imgs = []
        sn = []
        for s in series:
            try:
                img = torch.load(s, weights_only=True)
            except:
                print('Missing: '+s)
                continue
            img = img[None, ...].float() / 255.0 # [1, d, h, w]
            sn.append(s.split('/')[-2])

            if self.transform:
                img = self.transform(img)
                img = torch.as_tensor(img).float()

            normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
            img = normalizer(img)
            imgs.append(img)

        while len(imgs) < (self.num_series if self.num_series else 10):
            print('expanding imgs')
            imgs.append(imgs[-1])
            sn.append(sn[-1])

        serienames = torch.zeros(1)
        if self.use_seriename:
            flist = [chartovec(s) for s in sn]
            serienames = torch.stack(flist,dim=0)

        return torch.stack(imgs, dim=0), report, is_normal, serienames, textmask

# helper collator function for collating ChunkedDataset
# the final output should be (images, (reports, textmasks), is_normals, serienames)
def chunked_collator(datas):
    if len(datas[0]) == 2:
        return chunked_collator([data[0] for data in datas]),chunked_collator([data[1] for data in datas])
    if len(datas[0]) == 5:
        datas = [[d[i] for d in datas] for i in range(5)]
        return default_collate(datas[0]),(torch.cat(datas[1]),default_collate(datas[4])),default_collate(datas[2]),default_collate(datas[3])
    else:
        return default_collate(datas)

# get dataloaders for itemizedclip
def get_itemizedclip_dataset(args, preprocess_fn, is_train, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if (not is_train) and input_filename == 'valid' and args.is_ct:
        input_filename = 'val'
    dataset = ChunkedDataset(
        input_filename,
        preprocess_fn,
        tokenizer,
        num_series=args.num_series if is_train else None,
        report_replacer = args.report_replacer,
        use_seriename = args.use_serienames,
        alt_report = args.alt_report,
        is_ct = args.is_ct,
        report_cap = args.num_sampled_captions
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=chunked_collator
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

# Main function for getting data
def get_data(args, tokenizer=None):
    data = {}

    if args.is_ct_rate:
        from open_ct_rate.data_chest_ct import get_ctrate_data
        return get_ctrate_data(args,tokenizer)

    if args.itemizedclip_text or args.use_itemizedclip_loss:
        data_fn = get_itemizedclip_dataset
    else:
        raise NotImplementedError

    if args.train_data:
        data["train"] = data_fn(args, None, is_train=True, tokenizer=tokenizer)
    if args.val_data:
        data["val"] = data_fn(args, None, is_train=False, tokenizer=tokenizer)
    if args.test_data:
        data["test"] = data_fn(args, None, is_train=False, tokenizer=tokenizer)
    return data
