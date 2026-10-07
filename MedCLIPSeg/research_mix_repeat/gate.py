"""Prospective new-seed screen for background false positives in positive images."""
from statistics import mean,median
KEYS=('DSC_percent','small_DSC_percent','micro_foreground_precision_percent',
      'micro_foreground_recall_percent','zero_overlap_positive')

def replication_gate(summaries,cases,suite):
    per_seed={};g=suite['screening_gate']
    for seed in suite['seeds']:
        br,tr=f'baseline_seed{seed}',f'global_mix_seed{seed}'
        b,t=summaries[br],summaries[tr];delta={k:t[k]-b[k] for k in KEYS}
        before={r['name']:r for r in cases[br]};after={r['name']:r for r in cases[tr]}
        if not before or before.keys()!=after.keys():raise ValueError('Cases must be paired')
        if any(int(before[n]['gt_pixels'])!=int(after[n]['gt_pixels']) for n in before):raise ValueError('GT differs')
        fp_before=sum(int(r['fp_pixels']) for r in before.values());fp_after=sum(int(r['fp_pixels']) for r in after.values())
        paired_fp=[int(after[n]['fp_pixels'])-int(before[n]['fp_pixels']) for n in before]
        fraction=sum(v<0 for v in paired_fp)/len(paired_fp)
        dsc=[100*(float(after[n]['DSC'])-float(before[n]['DSC'])) for n in before]
        constraints=dict(precision_direction=delta['micro_foreground_precision_percent']>g['precision_gain_each_seed_min_pp'],
                         actual_false_positive_decrease=fp_after<fp_before,
                         fp_case_fraction=fraction>=g['fp_improved_fraction_min'],
                         overall=delta['DSC_percent']>=-g['overall_DSC_max_drop_pp'],
                         small=delta['small_DSC_percent']>=-g['small_DSC_max_drop_pp'],
                         recall=delta['micro_foreground_recall_percent']>=-g['recall_max_drop_pp'],
                         zero_overlap=delta['zero_overlap_positive']<=g['zero_overlap_max_increase'])
        per_seed[str(seed)]=dict(delta=delta,fp_baseline=fp_before,fp_candidate=fp_after,
                                fp_relative_reduction_percent=100*(fp_before-fp_after)/fp_before if fp_before else None,
                                fp_improved_fraction=fraction,median_DSC_delta_pp=median(dsc),
                                constraints=constraints,passed=all(constraints.values()))
    average=mean(v['delta']['micro_foreground_precision_percent'] for v in per_seed.values())
    return dict(passed=all(v['passed'] for v in per_seed.values()) and average>=g['mean_precision_min_gain_pp'],
                per_seed=per_seed,mean_precision_delta_pp=average,seed666_excluded_from_primary=True,
                engineering_screen_only=True,independent_data_confirmation=False)
