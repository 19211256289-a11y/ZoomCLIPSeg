"""Prospective investment criteria, distinct from descriptive four-arm effects."""
from statistics import median
KEYS=('DSC_percent','NSD_percent','small_DSC_percent','micro_foreground_precision_percent',
      'micro_foreground_recall_percent','positive_DSC_below_50_percent','zero_overlap_positive')

def combination_gate(summaries,cases,suite):
    modes=('baseline','global_mix','detail','detail_global_mix')
    if set(summaries)!=set(modes) or set(cases)!=set(modes):raise ValueError('All four arms required')
    paired={m:{r['name']:r for r in cases[m]} for m in modes};names=set(paired['baseline'])
    if not names or any(set(paired[m])!=names for m in modes):raise ValueError('Matched cases required')
    if any(len(paired[m])!=len(cases[m]) for m in modes):raise ValueError('Duplicate cases')
    if any(float(paired[m][n]['gt_fraction'])!=float(paired['baseline'][n]['gt_fraction']) for m in modes for n in names):raise ValueError('GT differs')
    combo=summaries['detail_global_mix']
    differences={m:{k:combo[k]-summaries[m][k] for k in KEYS} for m in modes[:-1]}
    threshold=summaries['baseline']['source_train_area_threshold']
    small=[n for n in sorted(names) if 0<float(paired['baseline'][n]['gt_fraction'])<=threshold]
    if not small:raise ValueError('Small cohort missing')
    d=[100*(float(paired['detail_global_mix'][n]['DSC'])-float(paired['global_mix'][n]['DSC'])) for n in small]
    fraction=sum(v>0 for v in d)/len(d);g=suite['screening_gate'];a=differences['global_mix'];b=differences['detail']
    constraints=dict(small_gain=a['small_DSC_percent']>=g['small_gain_over_global_min_pp'],
        overall_protected=a['DSC_percent']>=-g['overall_max_drop_pp'],
        precision_protected=a['micro_foreground_precision_percent']>=-g['precision_max_drop_pp'],
        recall_protected=a['micro_foreground_recall_percent']>=-g['recall_max_drop_pp'],
        small_case_fraction=fraction>=g['small_improved_fraction_min'],
        no_more_severe_failures=a['positive_DSC_below_50_percent']<=0,
        competitive_with_detail_overall=b['DSC_percent']>=-g['overall_max_drop_vs_detail_pp'],
        competitive_with_detail_small=b['small_DSC_percent']>=-g['small_max_drop_vs_detail_pp'])
    individual={m:{k:summaries[m][k]-summaries['baseline'][k] for k in KEYS} for m in ('detail','global_mix')}
    interaction={k:combo[k]-summaries['detail'][k]-summaries['global_mix'][k]+summaries['baseline'][k] for k in KEYS}
    return dict(passed=all(constraints.values()),constraints=constraints,combo_minus=differences,
        individual_minus_baseline=individual,interaction=interaction,small_improved_fraction_over_global=fraction,
        small_median_delta_over_global_pp=median(d),engineering_screen_only=True,independent_confirmation=False)
