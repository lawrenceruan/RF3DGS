"""Bounded depth search selected on four views, validated on two unused views."""
from pathlib import Path
import json, time
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
from custom_utils.correspondence_prior import patch_vote


def prepare(dataset, output):
    start = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        scene = Scene(dataset, GaussianModel(dataset.sh_degree), load_iteration=5000, shuffle=False)
        views = scene.getTrainCameras(); device = 'cuda'
        depths=[]; poses=[]; focals=[]; hp=[]; band=[]; interiors=[]
        for view in views:
            path=Path(dataset.model_path)/'train/ours_5000/Depth_pth'/(Path(view.image_name).stem+'.pth')
            d=torch.load(path, map_location=device, weights_only=True)
            depths.append(d); interiors.append(depth_interior(d))
            rgb=view.original_image[:3].to(device)
            hp.append(rgb-lowpass(rgb)); band.append(lowpass(rgb, 1)-lowpass(rgb, 4))
            pose=torch.eye(4, device=device)
            pose[:3,:3]=torch.as_tensor(view.R.T, device=device)
            pose[:3,3]=torch.as_tensor(view.T, device=device)
            poses.append(pose)
            focals.append((fov2focal(view.FoVx,view.image_width),fov2focal(view.FoVy,view.image_height)))
        inv=[torch.linalg.inv(p) for p in poses]; stats={}
        factors=[1., .98, 1.02, .96, 1.04, .94, 1.06]
        for i, view in enumerate(views):
            h,w=hp[i].shape[-2:]; points=camera_rays(h,w,*focals[i],device)*depths[i]
            target_valid=interiors[i].clone()
            target_valid[:10]=False;target_valid[-10:]=False;target_valid[:,:10]=False;target_valid[:,-10:]=False
            neighbors=sorted([j for j in range(len(views)) if j!=i],key=lambda j:float(torch.linalg.norm(inv[i]-inv[j])))[:6]
            # Interleave held-out correspondence validators across neighbor distances.
            selectors=[neighbors[k] for k in [0,2,3,5]]
            validators=[neighbors[k] for k in [1,4]]
            def warp(j, factor):
                packed=torch.cat((band[j], hp[j], interiors[j][None].float()))
                pose=poses[j]@inv[i]
                sampled,valid=visible_sample(points*factor,pose,packed,depths[j],*focals[j],.03,10)
                # Search cannot move a match more than two source pixels.
                xyz=(pose[:3,:3]@points.flatten(1)+pose[:3,3:4]).reshape_as(points)
                altered=(pose[:3,:3]@(points*factor).flatten(1)+pose[:3,3:4]).reshape_as(points)
                dx=focals[j][0]*(xyz[0]/xyz[2]-altered[0]/altered[2])
                dy=focals[j][1]*(xyz[1]/xyz[2]-altered[1]/altered[2])
                valid &= target_valid & (sampled[6]>.999) & (dx.square()+dy.square() <= 4)
                return sampled, valid
            best=torch.zeros((h,w),device=device); chosen=torch.ones_like(best)
            for factor in factors:
                scores=[]
                for j in selectors:
                    sampled,valid=warp(j,factor)
                    scores.append(patch_vote(band[i],sampled[:3],valid))
                scores=torch.stack(scores);count=(scores>0).sum(0)
                score=torch.where(count>=2,scores.sum(0)/4,0)
                use=score>best+1e-6
                best=torch.where(use,score,best);chosen=torch.where(use,factor,chosen)
            # Validate selected geometry without choosing a new depth on these views.
            votes=[]; validation=[]; original=[]
            for j in neighbors:
                sampled,valid=warp(j,chosen)
                coarse=patch_vote(band[i],sampled[:3],valid)
                fine=patch_vote(hp[i],sampled[3:6],valid,window=7,threshold=.7)
                vote=torch.minimum(coarse,fine)
                votes.append(vote)
                if j in validators:validation.append(vote)
                fixed,fixed_valid=warp(j,1.)
                original.append(texture_agreement(hp[i],fixed[3:6],fixed_valid))
            votes=torch.stack(votes); count=(votes>0).sum(0)
            valid=(best>0)&(count>=3)&(torch.stack(validation)>0).any(0)
            confidence=torch.where(valid,votes.sum(0)/count.clamp_min(1)*(count/4).clamp_max(1),0)
            original=torch.stack(original); old=(original>0).sum(0)>=2
            assert torch.isfinite(confidence).all()
            torch.save(dict(image_name=view.image_name,confidence=confidence.cpu().half()),output/(Path(view.image_name).stem+'.pt'))
            for suffix,value in [('',confidence),('_shift',((chosen-1).abs()/.06)*valid),('_added',valid&~old)]:
                Image.fromarray((value.float().cpu().numpy()*255).round().astype(np.uint8)).save(output/(Path(view.image_name).stem+suffix+'.png'))
            stats[view.image_name]=dict(supported_fraction=float(valid.float().mean()),mean_confidence=float(confidence.mean()),old_gate_fraction=float(old.float().mean()),newly_supported=float((valid&~old).float().mean()),shifted_supported=float((valid&(chosen!=1)).float().mean()),selectors=[views[j].image_name for j in selectors],validators=[views[j].image_name for j in validators])
            print(view.image_name, json.dumps(stats[view.image_name]),flush=True)
        (output/'manifest.json').write_text(json.dumps(dict(statistics=stats,preparation_seconds=time.monotonic()-start,clean_targets_used=False,depth_tolerance=.03,factors=factors,max_pixel_shift=2,training_views=len(views)),indent=2))

if __name__=='__main__':
    parser=ArgumentParser();model=ModelParams(parser,sentinel=True);parser.add_argument('--output',required=True)
    args=get_combined_args(parser);prepare(model.extract(args),args.output)
