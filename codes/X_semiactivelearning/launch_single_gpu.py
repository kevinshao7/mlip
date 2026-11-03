#!/usr/bin/env python3
"""Launch the next prepared wB97M campaign batch with no required flags."""
import json
import os
os.environ.setdefault('SAL_RUN_ID', 'wb97m_step5')
import campaign
from workflow_policy import policy


def main():
    from atomic_references import prepare
    prepare()
    cfg = policy()
    path = campaign.OUT / 'pressure_progress.json'
    passed = json.loads(path.read_text()).get('passed', {}) if path.exists() else {}
    pressure = next((p for p in campaign.config()['pressures_gpa'] if campaign.tag(p) not in passed), None)
    if pressure is None:
        raise ValueError('All configured pressures have passed the force gate')
    if not (campaign.OUT / 'model_lineage.json').exists():
        campaign.command_init(None)
    args = campaign.parser().parse_args([
        'finetune', '--pressure', str(pressure), '--run', '--epochs', str(cfg['finetune_batch_epochs']),
        '--changed-parameter', 'acquisition_dataset',
        '--change-note', 'Add validated wB97M-only acquisition labels; retain temporal partitions, parent, lr=1e-4 and training recipe.'])
    args.func(args)


if __name__ == '__main__':
    main()
