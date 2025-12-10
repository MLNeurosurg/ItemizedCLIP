import sys
sys.path.append('.')
sys.path.append('..')

from open_mri_train.data import *


class PublicDataset(Dataset):
    def __init__(self,data_dir,transform=None,tokenizer=None,num_series=None,use_seriename=False, dataname = None, bloodonly=False):
        
        self.dataname = dataname
        if dataname == 'rsna':
            jrp = data_dir
            self.dataphrase = 'data_250309'
            self.serienamemap = {"img_bone":"MPR-AX_HEAD-protocol_BoneWindow",
                "img_brain":"MPR-AX_HEAD-protocol_BrainWindow",
                "img_blood":"MPR-AX_HEAD-protocol_BloodWindow"}
        elif dataname == 'cq500':
            jrp = data_dir
            self.dataphrase = 'data_250309'
            self.serienamemap = {"img_bone":"AX_BoneWindow",
                "img_brain":"AX_BrainWindow",
                "img_blood":"AX_BloodWindow"}
        else:
            raise NotImplementedError

        keys = os.listdir(jrp)
        if dataname == 'cq500':
            keys.remove('CQ500CT163')
        
        self.bloodonly = bloodonly
        self.studies = [(key,MrStudyInfo(root=jrp,key=key,value = self.getvalue(jrp,key))) for key in keys]
        self.num_series = num_series
        self.transform = transform
        self.tokenizer = tokenizer
        self.mykeys = keys
        
        self.use_seriename = use_seriename

        
    
    def getvalue(self,jrp,key):
        subid = os.listdir(jrp+'/'+key)
        if self.bloodonly:
            d = {"report": ["no report included"],"series":{"img_blood":{}},"subid":subid}
        else:
            if self.dataname == 'cq500':
                series = {}
                for sid in subid:
                    series[sid+'/'+self.dataphrase+"/img_blood"] = {}
                    series[sid+'/'+self.dataphrase+"/img_bone"] = {}
                    series[sid+'/'+self.dataphrase+"/img_brain"] = {}
                d = {"report": ["no report included"],"series":series,"subid":subid}
            elif self.dataname == 'rsna':
                d = {"report": ["no report included"],"series":{"img_blood":{},"img_bone":{},"img_brain":{}},"subid":subid}
        return d
        
    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        uid, study = self.studies[idx]

        # get is_normal
        is_normal = torch.as_tensor([study.is_normal()]).long()

        # get report
        report = self.tokenizer([str(study.get_report(shuffle=False))])[0]

        # get series
        series = study.get_series(shuffle=False)
        
        # load in series
        imgs = []
        sn = []
        for s in series:
            if self.dataname == 'rsna':
                try:
                    img = torch.load(s.replace(uid,uid+'/'+study.subid[0]).replace('/img.pt','.pt'), weights_only=True)
                except:
                    print('missing: '+s.replace(uid,uid+'/'+study.subid[0]).replace('/img.pt','.pt'))
                    continue
                sn.append(self.serienamemap[s.split('/')[-2]])
                
            elif self.dataname == 'cq500':
                try:
                    img = torch.load(s.replace('/img.pt','.pt').replace(self.dataphrase+'/',''), weights_only=True)
                    sn.append(s.split('/')[-4]+'_'+self.serienamemap[s.split('/')[-2]])
                except:
                    print(s.replace('/img.pt','.pt').replace(self.dataphrase+'/',''))
                    continue

            
            img = img[None, ...].float() / 255.0

            if self.transform:
                img = self.transform(img)
                img = torch.as_tensor(img).float()

            normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
            img = normalizer(img)
            imgs.append(img)
        if len(imgs) == 0:
            print(uid)
            return None

        serienames = torch.zeros(1)
        if self.use_seriename:
            flist = [chartovec(s) for s in sn]
            serienames = torch.stack(flist,dim=0)

        return uid, torch.stack(imgs, dim=0), report, is_normal, serienames



def get_public_dataset(args, preprocess_fn, tokenizer=None):
    data_dir = args.data_dir
    assert data_dir
    dataset = PublicDataset(
        data_dir,
        preprocess_fn,
        tokenizer,
        args.num_series,
        use_seriename=args.use_serienames,
        dataname = args.dataset,
        bloodonly = args.bloodonly
    )
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    dataloader.imagefolder=None
    return dataloader
