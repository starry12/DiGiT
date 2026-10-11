"""The GIDS/DiGiT v4 baseline remains frozen; only DiGiT variants run here."""
from candidates.pa_sage_512b_pair_v4.common import *
from candidates.pa_sage_512b_pair_v4 import common as baseline

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/pa_sage_512b_latency_20260926_v1'
MODULE='candidates.pa_sage_512b_latency_v1'
VARIANTS=('baseline','frontier','functional','combined')


def switches(variant):
    require(variant in VARIANTS,'Unknown latency variant')
    return dict(frontier=variant in ('frontier','combined'),
                functional=variant in ('functional','combined'))


def source_files():
    return sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json')


def verify():
    m=read(HERE/'manifest.json')
    require(baseline.verify()==m['baseline_sha256'],'Frozen v4 changed')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Candidate changed: '+rel)
    for name,digest in m['evidence_sha256'].items():require(sha(OUT/name)==digest,'Evidence changed: '+name)
    for name,digest in m['runtime_sha256'].items():require(sha(name)==digest,'Runtime changed: '+name)
    require(read(HERE/'protocol.json')==read(baseline.HERE/'protocol.json'),'Training protocol changed')
    require(sha(OUT/'inputs.json')==sha(baseline.OUT/'inputs.json'),'Inputs changed')
    return sha(HERE/'manifest.json')


def validate_completion(report):
    baseline.validate_completion(report)
    require(report['arm']=='digit' and report['variant'] in VARIANTS,'This candidate only runs DiGiT')
    require(report['changes']==switches(report['variant']),'Unexpected optimization switches')
    require(report['model_implementation']==('functional_mean_sage' if report['changes']['functional'] else 'original_sage'),
            'Wrong model implementation')
    require(report['frontier_implementation']==('known_extent_coo' if report['changes']['frontier'] else 'dgl_graph'),
            'Wrong frontier implementation')
    require(report['additional_sampling_profile'] is False,'Performance must be uninstrumented')
    return True


def full_schedule():
    return ['baseline','frontier','functional','combined','combined','functional','frontier','baseline']
