"""Small CUDA operator smoke test, using synthetic tensors and no real data."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import json
from pathlib import Path
import torch
import component_runner as r


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This optional smoke test needs CUDA")
    device=torch.device("cuda")
    records=[]
    recipe=r.protocol()[0]["shared"]
    for arm in r.ARMS:
        r.set_seed(42)
        model=r.build_model(arm).to(device)
        images=torch.rand(1,4,128,128,device=device)
        masks=(torch.rand(1,1,128,128,device=device)>.8).float()
        opt=r.optimizer_for(model,recipe)
        criterion=r.BCEDiceLoss()
        r.configure_parameters(model,True)
        r.set_seed(42)
        model.train()
        for warmup in [True,False]:
            r.configure_parameters(model,warmup)
            opt.zero_grad(set_to_none=True)
            logits,aux=model(images,return_aux=True)
            loss=criterion(logits,masks)
            if not torch.isfinite(loss):raise AssertionError("Nonfinite CUDA loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise AssertionError("Nonfinite CUDA gradients")
            opt.step()
        records.append({"arm":arm,"finite":True,"image_size":128,"warmup_and_tail_update":True})
        del model,opt,images,masks,logits,aux
        torch.cuda.empty_cache()
    result={"synthetic_only":True,"real_data_accessed":False,"gpu":torch.cuda.get_device_name(0),
            "torch":torch.__version__,"arms":records,
            "note":"128px operator test is not a 512px 4070 memory benchmark and not an experimental result"}
    (Path(__file__).parent/"component_cuda_test_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
