"""All seeds and cohorts; descriptive comparison without test-driven selection."""
import json
from statistics import mean, stdev
from .common import suite_output, write_json, now, split_run, sha256

METRICS = ('DSC_percent', 'small_DSC_percent', 'NSD_percent',
           'micro_foreground_precision_percent', 'micro_foreground_recall_percent',
           'positive_DSC_below_50_percent', 'zero_overlap_positive')


def paired_comparison(rows, candidate, reference, seeds):
    pairs = []
    for seed in seeds:
        a, b = rows.get(f'{candidate}_seed{seed}'), rows.get(f'{reference}_seed{seed}')
        if a is not None and b is not None:
            pairs.append(dict(seed=seed, **{m: a[m]-b[m] if a[m] is not None and b[m] is not None else None for m in METRICS}))
    if len(pairs) != len(seeds):
        return dict(complete=False, completed_pairs=len(pairs), per_seed=pairs)
    valid = [m for m in METRICS if all(p[m] is not None for p in pairs)]
    return dict(complete=True, completed_pairs=len(pairs), per_seed=pairs,
                mean_changes={m: mean(p[m] for p in pairs) for m in valid},
                std_changes={m: stdev(p[m] for p in pairs) for m in valid})


def write_report(suite):
    root = suite_output(suite)
    plan = json.loads((root/'experiment_plan.json').read_text(encoding='utf-8'))
    cohorts = [('Val', suite['dataset'])] + [('Test', target) for target in suite['test_targets']]
    results, completed = {}, 0
    lines = ['# Kvasir 正式训练结果', '',
             '三组各三个种子，从 UniMedCLIP 基础权重开始，每次100轮。分割头和适配器按种子重新初始化；同种子三组初始参数一致。',
             '所有组采用整图 Dice 修正和两个训练视图的相同预算，属于修正后的预算匹配对照；不能将 paired_repeat 直接标成未经改动的官方结果。',
             '只比较第100轮最终权重，MC=30，阈值0.5，单视图推理。完整测试在九组全部训练结束后运行。',
             '训练策略的主要对比为 paired_aug 对 paired_repeat，文本同步的必要消融为 paired_aug 对 paired_image。未增加模型结构模块。',
             '源域验证集已经用于方法开发。测试集曾用于原始复现，本轮不用于调参、挑种子或选择检查点。', '']
    for split, target in cohorts:
        rows = {}
        for run in suite['runs']:
            dest = root/'evaluations'/run/split/target
            if not (dest/'manifest.json').exists():
                continue
            manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
            if not manifest.get('complete'):
                continue
            s = json.loads((dest/'summary.json').read_text(encoding='utf-8'))
            variant, seed = split_run(run)
            expected = 100 if split == 'Val' else plan['test_cohorts'][target]['n']
            assert s['run_id'] == run and s['variant'] == variant and s['seed'] == seed
            assert s['n'] == expected and s['split'] == split and s['target'] == target and s['mc_samples'] == 30
            assert not s['diagnostic'] and not s['subset_check']
            assert manifest['signature']['training_signature'] == plan['signatures'][run]['digest']
            assert sha256(dest/'cases.csv') == manifest['cases_sha256']
            rows[run] = s
        aggregates = {}
        for variant in suite['variants']:
            values = [rows[f'{variant}_seed{seed}'] for seed in suite['seeds'] if f'{variant}_seed{seed}' in rows]
            if len(values) == len(suite['seeds']):
                aggregates[variant] = {m: dict(mean=mean(v[m] for v in values), sample_std=stdev(v[m] for v in values))
                                       for m in METRICS if all(v[m] is not None for v in values)}
        comparisons = {a+'_vs_'+b: paired_comparison(rows, a, b, suite['seeds']) for a, b in (
            ('paired_aug', 'paired_repeat'), ('paired_image', 'paired_repeat'), ('paired_aug', 'paired_image'))}
        results[f'{split}/{target}'] = dict(summaries=rows, aggregates=aggregates, paired_comparisons=comparisons)
        completed += len(rows)
        lines += [f'## {split} {target}', '', f'已完成 {len(rows)}/9 次评估。', '',
                  '|实验|seed|DSC %|小目标 DSC %|NSD %|严重失败|零重叠|', '|---|---:|---:|---:|---:|---:|---:|']
        for s in rows.values():
            small = 'N/A' if s['small_DSC_percent'] is None else f"{s['small_DSC_percent']:.4f}"
            lines.append(f"|{s['variant']}|{s['seed']}|{s['DSC_percent']:.4f}|{small}|{s['NSD_percent']:.4f}|{s['positive_DSC_below_50_percent']}|{s['zero_overlap_positive']}|")
        if aggregates:
            lines += ['', '|三种子汇总|DSC 均值 ± 样本标准差 %|小目标 DSC %|', '|---|---:|---:|']
            for v, values in aggregates.items():
                a, b = values['DSC_percent'], values.get('small_DSC_percent')
                small = f"{b['mean']:.4f} ± {b['sample_std']:.4f}" if b else 'N/A'
                lines.append(f"|{v}|{a['mean']:.4f} ± {a['sample_std']:.4f}|{small}|")
        if split == 'Test':
            cohort = plan['test_cohorts'][target]
            lines += ['', f"当前划分 {cohort['n']} 条记录，按图像文件 SHA256 共 {cohort['n_unique_images']} 张独立内容。主表保留现有划分。",
                      '跨测试集共享图像数量：'+json.dumps(cohort['shared_unique_images_with_other_tests'], ensure_ascii=False)+'。存在重叠的数据集不能作为完全独立证据。']
            if rows and cohort['n_unique_images'] < cohort['n']:
                lines += ['', '|去重敏感性统计|seed|按唯一图像等权 DSC %|', '|---|---:|---:|']
                for s in rows.values():
                    lines.append(f"|{s['variant']}|{s['seed']}|{s['unique_image_sensitivity']['DSC_percent']:.4f}|")
                lines.append('同一图像对应多条记录时先平均这些记录的 DSC，再对不同图像等权平均；不据此更改主要评价或挑选模型。')
        lines.append('')
    write_json(root/'formal_results.json', dict(generated_at=now(), completed_evaluations=completed,
               total_evaluations=54, complete=completed == 54, plan_digest=plan['digest'], cohorts=results,
               statistical_significance_established=False, automatic_expansion=False))
    lines += ['这些结果不自动触发追加训练。均值和三个种子本身不构成统计显著性证据；待全部结果完成后统一验收。', '']
    destination = root/'research_report.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    return destination
