"""Record the complete CLI configuration without changing training behavior."""
from pathlib import Path
import json,os,hashlib,warnings


def record_training_args(args, trainer, argv):
    inactive={
        'lambda_T':'The transmission prior uses the trainer schedule; this flag is not wired.',
        'lambda_R':'The reflection reconstruction coefficient is fixed in the trainer.',
        'lambda_exc':'Exclusion uses exc_weight(iteration); this flag is not wired.',
        'lambda_depth':'Use --depth_smooth_weight for the active bilateral depth coefficient.',
        'lambda_beta':'The beta TV coefficient is fixed in the trainer.',
        'T_ply':'Stage two initializes transmission from SfM; this argument is not wired.'}
    supplied={token[2:].split('=',1)[0] for token in argv if token.startswith('--')}
    ignored={name:dict(value=getattr(args,name,None),reason=reason)
             for name,reason in inactive.items() if name in supplied}
    if ignored:
        warnings.warn('Ignored legacy controls: '+ '; '.join('--'+name+': '+v['reason'] for name,v in ignored.items()),RuntimeWarning,stacklevel=2)
    path=Path(os.path.join('./output/',args.scene_name))/'training_args.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    record={'format_version':1,'arguments':vars(args).copy(),'argv':list(argv),
            'RF_SEED_environment':os.environ.get('RF_SEED'),
            'trainer_sha256':hashlib.sha256(Path(trainer).read_bytes()).hexdigest(),
            'explicitly_supplied_but_inactive_controls':ignored}
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
    return path
