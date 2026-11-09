"""Build fitting-only detail confidence from frozen coarse geometry."""
from pathlib import Path
import json, hashlib, time
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from scene.scene_1st_stage import Scene
from gaussian_renderer import GaussianModel
from utils.graphics_utils import fov2focal
from custom_utils.visible_prior import camera_rays, visible_sample
from custom_utils.frequency_prior import lowpass, depth_interior, texture_agreement


def prepare(dataset, output, iteration=5000):
    start=time.monotonic(); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        scene=Scene(dataset,GaussianModel(dataset.sh_degree),load_iteration=iteration,shuffle=False)
        views=scene.getTrainCameras(); device='cuda'; depths=[]; poses=[]; focals=[]; hp=[]; interiors=[]
        root=Path(dataset.model_path)/'train'/f'ours_{iteration}'
        for view in views:
            d=torch.load(root/'Depth_pth'/(Path(view.image_name).stem+'.pth'),map_location=device,weights_only=True)
            assert d.shape==(1,view.image_height,view.image_width)
            depths.append(d); interiors.append(depth_interior(d))
            rgb=view.original_image[:3].to(device); hp.append(rgb-lowpass(rgb))
            pose=torch.eye(4,device=device)
            pose[:3,:3]=torch.as_tensor(view.R.T,device=device);pose[:3,3]=torch.as_tensor(view.T,device=device)
            poses.append(pose);focals.append((fov2focal(view.FoVx,view.image_width),fov2focal(view.FoVy,view.image_height)))
        inv=[torch.linalg.inv(p) for p in poses]; stats={}
        for i,view in enumerate(views):
            h,w=hp[i].shape[-2:]
            points=camera_rays(h,w,*focals[i],device)*depths[i]
            target_valid=interiors[i].clone();target_valid[:8]=False;target_valid[-8:]=False;target_valid[:,:8]=False;target_valid[:,-8:]=False
            neighbors=sorted([j for j in range(len(views)) if i!=j],key=lambda j:float(torch.linalg.norm(inv[i]-inv[j])))[:6]
            support=torch.zeros((h,w),device=device); scores=torch.zeros_like(support)
            for j in neighbors:
                packed=torch.cat((hp[j],interiors[j][None].float()),0)
                sampled,valid=visible_sample(points,poses[j]@inv[i],packed,depths[j],*focals[j],.03,8)
                vote=texture_agreement(hp[i],sampled[:3],valid & target_valid & (sampled[3]>.999))
                support+=(vote>0);scores+=vote
            confidence=torch.where(support>=2,scores/support.clamp_min(1)*(support/3).clamp_max(1),0)
            assert torch.isfinite(confidence).all()
            record={'image_name':view.image_name,'confidence':confidence.cpu().half()}
            torch.save(record,output/(Path(view.image_name).stem+'.pt'))
            Image.fromarray((confidence.cpu().numpy()*255).round().astype(np.uint8)).save(output/(Path(view.image_name).stem+'.png'))
            stats[view.image_name]={'supported_fraction':float((confidence>0).float().mean()),'mean_confidence':float(confidence.mean()),'neighbors':[views[j].image_name for j in neighbors]}
            print(view.image_name,stats[view.image_name],flush=True)
        result={'source':dataset.source_path,'coarse':dataset.model_path,'iteration':iteration,'sigma':2.0,'neighbors':6,'minimum_support':2,'depth_tolerance':.03,'border':8,'statistics':stats,'preparation_seconds':time.monotonic()-start,'clean_targets_used':False}
        (output/'manifest.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=ArgumentParser();model=ModelParams(parser,sentinel=True)
    parser.add_argument('--output',required=True);parser.add_argument('--iteration',type=int,default=5000)
    args=get_combined_args(parser);prepare(model.extract(args),args.output,args.iteration)
