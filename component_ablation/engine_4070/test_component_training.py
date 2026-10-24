"""Synthetic infrastructure tests. No real checkpoint or dataset is accessed."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import component_runner as r


class TinyImages(Dataset):
    def __init__(self,prefix,count):
        self.prefix=prefix
        self.count=count
        gen=torch.Generator().manual_seed(70)
        self.x=torch.rand(count,4,32,32,generator=gen)
        self.y=(torch.rand(count,1,32,32,generator=gen)>.85).float()
    def __len__(self):return self.count
    def __getitem__(self,i):return self.x[i],self.y[i],f"{self.prefix}_{i}"


def make_config(root,arm):
    return {"arm":arm,"seed":42,"workers":0,"output_root":str(root),"protocol_hash":"a"*64,
            "split_sha256":"b"*64,"initial_checkpoint_sha256":"c"*64,
            "recipe":{"epochs":2,"warmup_epochs":1,"module_learning_rate":1e-4,
                      "tail_learning_rate":1e-5,"optimizer_betas":[.9,.999],
                      "optimizer_epsilon":1e-8,"weight_decay":0.0}}


class TrainingInfrastructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory(prefix="carer_component_synthetic_")
        cls.root=Path(cls.temp.name)
        cls.train=TinyImages("train",2)
        cls.val=TinyImages("val",1)
        cls.report={"synthetic_only":True,"real_data_accessed":False,"tests":[]}
    @classmethod
    def tearDownClass(cls):
        (r.PACKAGE/"component_training_test_report.json").write_text(json.dumps(cls.report,indent=2),encoding="utf-8")
        cls.temp.cleanup()

    def execute(self,name,arm,interrupt=False):
        root=self.root/name
        cfg=make_config(root,arm)
        r.set_seed(42)
        model=r.build_model(arm)
        initial=r.cpu_state(model)
        cfg["frozen_backbone_state_sha256"]=r.state_hash({name:value for name,value in initial.items()
            if not name.startswith(("color_detail_stem.","d1.","out_conv."))})
        run=root/"runs"/arm/"seed_42"
        checkpoints=root/"checkpoints"/arm/"seed_42"
        if interrupt:
            real=r.save_torch
            def interrupted_save(path,value):
                real(path,value)
                if Path(path).name=="last.pth" and value["completed_epoch"]==1:
                    raise RuntimeError("SYNTHETIC interruption after atomic epoch-one checkpoint")
            with patch.object(r,"save_torch",interrupted_save):
                with self.assertRaisesRegex(RuntimeError,"SYNTHETIC interruption"):
                    r.train_engine(model,self.train,self.val,cfg,run,checkpoints,torch.device("cpu"),test_only=True)
            del model
            r.set_seed(42)
            model=r.build_model(arm)
        result=r.train_engine(model,self.train,self.val,cfg,run,checkpoints,torch.device("cpu"),test_only=True)
        last=r.load_torch_checkpoint(checkpoints/"last.pth")
        self.assertEqual(result["completed_epochs"],2)
        self.assertEqual([row["epoch"] for row in last["rows"]],[0,1,2])
        for key,value in initial.items():
            if not key.startswith(("color_detail_stem.","d1.","out_conv.")):
                self.assertTrue(torch.equal(value,last["state_dict"][key]),key)
        return last,result,cfg,run,checkpoints

    def test_atomic_resume_and_frozen_weights(self):
        full,_,_,_,_=self.execute("uninterrupted","core")
        resumed,result,cfg,run,checkpoints=self.execute("resumed","core",True)
        self.assertEqual(full["best_epoch"],resumed["best_epoch"])
        for key in full["state_dict"]:
            self.assertTrue(torch.equal(full["state_dict"][key],resumed["state_dict"][key]),key)
            self.assertTrue(torch.equal(full["best_state_dict"][key],resumed["best_state_dict"][key]),key)
        self.assertEqual([row["augmentation_stream_sha256"] for row in full["rows"]],
                         [row["augmentation_stream_sha256"] for row in resumed["rows"]])
        model=r.build_model("core")
        again=r.train_engine(model,self.train,self.val,cfg,run,checkpoints,torch.device("cpu"),test_only=True)
        self.assertEqual(again,result)
        changed=dict(cfg)
        changed["workers"]=1
        with self.assertRaisesRegex(ValueError,"config differs"):
            r.train_engine(model,self.train,self.val,changed,run,checkpoints,torch.device("cpu"),test_only=True)
        self.report["tests"].append("real 32px Core: interrupted/resumed optimizer and weights equal uninterrupted training; frozen tensors unchanged; config change rejected")

    def test_other_arms_have_finite_losses(self):
        streams=[]
        for arm in ("no_bound","random_init"):
            last,result,_,_,_=self.execute(arm,arm)
            self.assertTrue(all(torch.isfinite(v).all() for v in last["state_dict"].values()))
            streams.append([row["augmentation_stream_sha256"] for row in last["rows"]])
            self.assertTrue(result["synthetic_test_only"])
        self.assertEqual(streams[0],streams[1])
        self.report["tests"].append("real 32px no_bound/random_init: finite two-epoch training; shared data-stream hashes; frozen tensors unchanged")

    def test_seal_gate_precedes_data_discovery(self):
        from types import SimpleNamespace
        args=SimpleNamespace(output_root=self.root/"unsealed",arm="core",seed=42)
        with patch.object(r,"check_package",return_value="d"*64),patch.object(r,"discover_pairs",side_effect=AssertionError("test discovery occurred")) as discover:
            with self.assertRaises(FileNotFoundError):
                r.test(args)
            discover.assert_not_called()
        self.report["tests"].append("official-test discovery is blocked before a valid all-nine seal")

    def test_content_identity_not_filename(self):
        root=self.root/"filename_collisions"
        pairs=[]
        for role,color in [("train",30),("test",170)]:
            folder=root/role
            folder.mkdir(parents=True)
            image=folder/"11_A.png"
            mask=folder/"mask.png"
            r.Image.fromarray(r.np.full((16,16,3),color,dtype=r.np.uint8)).save(image)
            r.Image.fromarray(r.np.zeros((16,16),dtype=r.np.uint8)).save(mask)
            pairs.append((image,mask,"11_a"))
        train_record=r.dataset_manifest(pairs[:1])
        test_record=r.dataset_manifest(pairs[1:],"synthetic test")
        r.verify_disjoint_image_contents(train_record,test_record)
        duplicate=r.copy.deepcopy(test_record)
        duplicate["files"][0]["image"]["pixel_sha256"]=train_record["files"][0]["image"]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError,"duplicates training"):
            r.verify_disjoint_image_contents(train_record,duplicate)
        self.report["tests"].append("same split-local filenames with distinct images are accepted; matching canonical image content is rejected")


if __name__=="__main__":
    unittest.main(verbosity=2)
