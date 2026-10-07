"""Fixed validation-only V6 screening; historical controls are descriptive."""
import json
from pathlib import Path
from .common import suite_output,now,sha256,write_json


def gate(candidate,reference,minimum_small_gain):
    if candidate is None or reference is None:return dict(passed=False,reason='Missing completed source validation')
    d={k:candidate[k]-reference[k] for k in ('DSC_percent','small_DSC_percent','micro_foreground_precision_percent')}
    return dict(passed=bool(d['small_DSC_percent']>=minimum_small_gain and d['DSC_percent']>=-.5 and d['micro_foreground_precision_percent']>=-1),
        small_DSC_gain_pp=d['small_DSC_percent'],overall_DSC_change_pp=d['DSC_percent'],micro_precision_change_pp=d['micro_foreground_precision_percent'])


def write_report(suite):
    root=suite_output(suite);rows={}
    for variant in suite['variants']:
        path=root/'evaluations'/variant/'Val'/suite['dataset']/'summary.json'
        if path.exists():rows[variant]=json.loads(path.read_text())
    plan=json.loads((root/'experiment_plan.json').read_text())
    meta=plan['signatures'][suite['variants'][0]]['config']['RESEARCH']
    for label in ('ADAM20','ADAM40'):
        path=Path(meta[f'REFERENCE_{label}_SUMMARY']);assert sha256(path)==meta[f'REFERENCE_{label}_SHA256']
        rows[label.lower()+'_reference']=json.loads(path.read_text())
    screening=dict(generated_at=now(),selection_uses='source Val only',automatic_expansion=False,statistical_significance_established=False)
    for candidate in suite['variants']:
        for ref in ('adam20_reference','adam40_reference'):
            screening[f'{candidate}_vs_{ref}']=gate(rows.get(candidate),rows[ref],2)
    screening['consistency_vs_augmentation']=gate(rows.get('paired_consistency'),rows.get('paired_aug'),1)
    screening['augmentation_passed']=all(screening['paired_aug_vs_'+r]['passed'] for r in ('adam20_reference','adam40_reference'))
    screening['consistency_passed']=screening['consistency_vs_augmentation']['passed'] and all(screening['paired_consistency_vs_'+r]['passed'] for r in ('adam20_reference','adam40_reference'))
    write_json(root/'screening.json',screening)
    lines=['# 第六轮同步镜像与跨视图约束结果','',
        '两组均为20轮、每步两个batch24视图、680次Adam更新、1360次训练前向/反向。单种子warm-start探索，不自动扩展。',
        'Adam20/40为已验收历史参照。1360次是逻辑视图计数，不含梯度检查点重算；不称与Adam40严格等算力。两组新实验的重算设置相同。',
        '正式评估使用原始单视图MC30，不采用镜像TTA，不以测试域调参。','',
        '|实验|DSC %|小目标 DSC %|NSD %|像素精确率 %|零重叠|DSC<50图数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v,s in rows.items():lines.append(f"|{v}|{s['DSC_percent']:.4f}|{s['small_DSC_percent']:.4f}|{s['NSD_percent']:.4f}|{s['micro_foreground_precision_percent']:.4f}|{s['zero_overlap_positive']}|{s['positive_DSC_below_50_percent']}|")
    lines+=['','```json',json.dumps(screening,ensure_ascii=False,indent=2),'```','',
        '一致性减少不能替代分割改善；通过工程门槛也不等于统计显著或方法新颖。']
    path=root/'research_report.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');return path
