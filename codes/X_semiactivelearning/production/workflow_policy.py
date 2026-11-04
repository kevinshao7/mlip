"""Pure policy checks for the active dispersion-free pressure campaign."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def policy():
    return json.loads((ROOT / 'autonomous_policy.json').read_text())


def partition(frame):
    cfg = policy()
    residue = (int(frame) // cfg['frame_block_size']) % cfg['frame_partition_modulus']
    if residue == cfg['evaluation_block_residue']:
        return 'evaluation'
    if residue == cfg['validation_block_residue']:
        return 'validation'
    return 'training'


def partition_frames(count, start=0, stride=1, evaluation=False):
    if count<0 or start<0 or stride<1:
        raise ValueError('Invalid analyzed frame window')
    return [start+i*stride for i in range(count)
            if (partition(start+i*stride)=='evaluation') == evaluation]


def passes_force_gate(report, cfg=None):
    cfg = cfg or policy()
    value = report.get(cfg['force_metric'])
    return (report.get('status') == 'complete'
            and report.get('n_clusters', 0) >= cfg['minimum_evaluation_clusters']
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value < cfg['force_threshold_mev_per_angstrom'])


def validate_nodisp_output(text, input_text):
    """Reject changed inputs, multiple jobs and any dispersion evaluation.

    The LibXC component name contains -V even without a nonlocal correction;
    the name alone is not evidence that dispersion was evaluated.
    """
    echoed = re.findall(r'^\s*\|\s*\d+> ?(.*)$', text, re.M)
    normalize = lambda lines: [' '.join(line.split()).lower() for line in lines if line.strip()]
    if normalize(echoed) != normalize(input_text.splitlines()):
        raise ValueError('no_dispersion_echoed_input_mismatch')
    if len(re.findall(r'FINAL SINGLE POINT ENERGY', text)) != 1:
        raise ValueError('no_dispersion_requires_single_job')
    if re.search(r'DFTD[234]\s+V|DISPERSION GRADIENT|Dispersion correction\s+[-+.\d]|Non-local correlation energy|NL Correlation|VV10.*(?:energy|correction)', text, re.I):
        raise ValueError('dispersion_evaluation_in_no_dispersion_job')
    if not re.search(r'Functional name\s+\.+\s+wB97M-V exchange-correlation functional', text, re.I):
        raise ValueError('missing_wb97m_electronic_functional')


def require_method_verified():
    import campaign
    path = campaign.OUT / 'method_verified.json'
    if not path.exists():
        raise ValueError('Direct wB97M route needs the single-geometry method pilot before production DFT')
    record = json.loads(path.read_text())
    for name, digest in record['validator_sha256'].items():
        if campaign.sha(ROOT / name) != digest:
            raise ValueError('Method validator changed; revalidate the preserved pilot')
    if record['method'] != campaign.config()['dft']['functional_basis']:
        raise ValueError('Verified pilot method disagrees with current method')
