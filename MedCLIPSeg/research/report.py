"""Descriptive reports and predeclared validation-only screening gates."""
import json
from pathlib import Path

from .common import suite_output, now, write_json


def gate(candidate, reference, minimum_small_gain):
    needed = ('small_DSC_percent', 'DSC_percent', 'micro_foreground_precision_percent')
    if not candidate or not reference or any(candidate.get(k) is None or reference.get(k) is None for k in needed):
        return {'passed': False, 'reason': 'Missing complete validation measurements'}
    small_gain = candidate['small_DSC_percent']-reference['small_DSC_percent']
    overall_change = candidate['DSC_percent']-reference['DSC_percent']
    precision_change = candidate['micro_foreground_precision_percent']-reference['micro_foreground_precision_percent']
    omission_reduction = reference['empty_positive']-candidate['empty_positive']
    benefit = small_gain >= minimum_small_gain
    if minimum_small_gain == 1:
        benefit = benefit or omission_reduction >= 1
    return dict(passed=bool(benefit and overall_change >= -.5 and precision_change >= -1),
                small_DSC_gain_pp=small_gain, overall_DSC_change_pp=overall_change,
                micro_precision_change_pp=precision_change, omitted_images_recovered=omission_reduction)


def write_report(suite):
    root = suite_output(suite)
    summaries, validation = [], {}
    for path in sorted((root / 'evaluations').glob('*/*/*/summary.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        if result.get('subset_check') or result.get('diagnostic'):
            continue
        summaries.append(result)
        if result['split'] == 'Val' and result['target'] == suite['dataset']:
            validation[result['variant']] = result
    screening = {'generated_at': now(), 'selection_uses': 'source validation only',
                 'detail': gate(validation.get('detail'), validation.get('dice_corrected'), 2),
                 'detail_beta': gate(validation.get('detail_beta'), validation.get('detail'), 1),
                 'statistical_significance_established': False, 'automatic_expansion': False}
    write_json(root / 'screening.json', screening)
    lines = ['# 研究实验结果', '', '这是从已有 100% 训练检查点继续训练的探索性比较，不能当作从基础权重训练的公平最终结论。',
             '筛选仅使用源域验证集；外域结果不参与模型选择。小目标阈值来自源域训练集。', '',
             '|Variant|Split|Target|N|DSC %|NSD %|Small DSC %|Empty positive|Micro precision %|ms/image|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.2f}'
    for r in summaries:
        lines.append(f"|{r['variant']}|{r['split']}|{r['target']}|{r['n']}|{fmt(r['DSC_percent'])}|{fmt(r['NSD_percent'])}|{fmt(r['small_DSC_percent'])}|{r['empty_positive']}/{r['n_positive']}|{fmt(r['micro_foreground_precision_percent'])}|{fmt(r['milliseconds_per_image'])}|")
    lines += ['', '## 预先规定的筛选门槛', '', '```json', json.dumps(screening, ensure_ascii=False, indent=2), '```', '',
              '通过门槛仅表示值得进一步验证，不表示统计显著或方法新颖。未通过不会自动追加实验。',
              '原始概率、方差、熵、PNG 预测及文件对应清单保存在各评估目录。NSD 沿用作者的 2 像素 surface Dice。']
    path = root / 'research_report.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return path
