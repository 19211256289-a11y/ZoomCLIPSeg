"""Problem-specific engineering screen, locked before any scientific training."""
from statistics import median

KEYS=('DSC_percent','small_DSC_percent','micro_foreground_precision_percent',
      'micro_foreground_recall_percent','zero_overlap_positive')

def screening_gate(summaries,cases,settings):
    candidate=summaries['spatial_mix'];pairs={}
    for reference,minimum in (('baseline',settings['small_DSC_min_gain_pp']),
                               ('global_mix',settings['over_global_mix_small_DSC_min_gain_pp'])):
        base=summaries[reference];delta={k:candidate[k]-base[k] for k in KEYS}
        left={r['name']:r for r in cases[reference]};right={r['name']:r for r in cases['spatial_mix']}
        if left.keys()!=right.keys():raise ValueError('Unpaired validation cases')
        threshold=base['source_train_area_threshold']
        if candidate['source_train_area_threshold']!=threshold:raise ValueError('Small-lesion threshold changed')
        if any(float(left[n]['gt_fraction'])!=float(right[n]['gt_fraction']) for n in left):raise ValueError('GT area mismatch')
        names=[n for n in left if 0<float(left[n]['gt_fraction'])<=threshold]
        if not names:raise ValueError('No small positive cases')
        differences=[100*(float(right[n]['DSC'])-float(left[n]['DSC'])) for n in names]
        center=median(differences);fraction=sum(v>1e-8 for v in differences)/len(differences)
        constraints=dict(small_mean=delta['small_DSC_percent']>=minimum,
                         small_median=center>=settings['small_median_delta_min_pp'],
                         small_improved_fraction=fraction>=settings['small_improved_fraction_min'],
                         overall=delta['DSC_percent']>=-settings['overall_DSC_max_drop_pp'],
                         precision=delta['micro_foreground_precision_percent']>=-settings['precision_max_drop_pp'],
                         zero_overlap=delta['zero_overlap_positive']<=settings['zero_overlap_max_increase'])
        pairs[reference]=dict(delta=delta,small_n=len(names),small_median_delta_pp=center,
                              small_improved_fraction=fraction,constraints=constraints,passed=all(constraints.values()))
    return dict(passed=all(v['passed'] for v in pairs.values()),primary_endpoint='source-defined small-lesion DSC',
                delta=pairs['baseline']['delta'],over_capacity_control=pairs['global_mix']['delta'],
                paired_checks=pairs,engineering_screen_only=True)
