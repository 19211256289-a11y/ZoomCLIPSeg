"""Render the verified evidence as a local acceptance report and scientific figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE/'audit.json').read_text(encoding='utf-8'))
assert d['passed']
variants = ['paired_repeat', 'paired_image', 'paired_aug']
seeds = [666, 667, 668]
labels = {'paired_repeat':'预算匹配基线', 'paired_image':'仅图像镜像增强', 'paired_aug':'同步图文镜像增强'}
english = {'paired_repeat':'Matched baseline', 'paired_image':'Image reflection', 'paired_aug':'Image + text reflection'}
colors = ['#667085', '#2563eb', '#059669']

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])

def ms(variant, key):
    v = d['aggregates'][variant][key]
    return f"{v['mean']:.4f} ± {v['sd']:.4f}"

def signed(v):
    return f'{v:+.4f}'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
for ax, metric, title in zip(axes, ['DSC_percent','small_DSC_percent'], ['Overall DSC', 'Small-lesion DSC']):
    for variant, color in zip(variants, colors):
        ax.plot(seeds, [d['summaries'][f'{variant}_seed{s}'][metric] for s in seeds],
                marker='o', linewidth=2, markersize=5, label=english[variant], color=color)
    ax.set_xticks(seeds)
    ax.set_xlabel('Training seed')
    ax.set_ylabel('DSC (%)')
    ax.set_title(title)
    ax.grid(axis='y', color='#e5e7eb')
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('Kvasir source validation | final epoch 100 | MC = 30', fontsize=12)
fig.savefig(HERE/'three_seed_validation.png', dpi=180)
fig.savefig(HERE/'three_seed_validation.svg')
plt.close(fig)

parts = ['# MedCLIPSeg 九组正式训练与源域验证验收',
         f"验收时间：{d['checked_at']}（香港时间）。实验套件：`formal_kvasir_v1`。",
         '**结论：九组正式训练及九组最终验证均通过本次完整性与数值验收。同步图文增强在源域验证集上取得稳定的总体增益，但相对仅图像增强的额外收益较小，尚不能据此确认测试集泛化或论文创新性。**',
         '本报告的“通过”指训练按锁定协议完成、检查点可读取、预测与指标可复核，不代表已通过统计显著性检验或已达到投稿标准。当前独立测试评估继续运行。',
         '## 1. 九组训练是否真正完成',
         '- 3 个方法 × 3 个种子（666、667、668），每组 100 个 epoch，共 900 个 epoch；每组均完成 3,400 次 Adam 更新、100 次调度器更新，日志覆盖 epoch 1–100。',
         '- 九个最终检查点均为 epoch 100，使用 `latest.pth`，并且其 SHA256 与对应验证结果一致。模型和优化器浮点状态均有限，没有 NaN/Inf。',
         '- 同一种子的三个方案具有相同初始参数指纹；三个种子的适配器与分割头初始参数各不相同。全部从锁定的 UniMedCLIP 基础权重独立开始，没有使用早期预实验训练完成的权重作为初始化。',
         '- 九组冻结参数的最终指纹与初始化逐字节一致；可训练参数的指纹均发生变化。三个方案结构一致：总参数 214,580,757，可训练参数 18,677,268。',
         f"- 复核 {d['checked_input_hashes']:,} 个锁定文件哈希（代码、Train/Val 文件、文本资源和基础权重），并重新读取 900 份概率/方差/熵数组及对应 PNG，按原始 GT 重新计算 DSC、NSD、召回、精确率等指标，均与已保存结果一致。",
         f"- 队列累计记录的训练活动时间为 **{d['active_training_hours']:.2f} 小时**，含训练进程中的逐轮验证、保存和启动开销，不含用户暂停等待及独立 MC=30 最终评估；不能当作纯 GPU kernel 时间。最后一组训练于 2026-10-03 09:43:19 完成，最终源域验证于 09:44:13 完成。",
         '验收脚本仅使用 CPU，在本报告目录写入证据；未调用训练/推理队列操作，也未修改锁定代码、配置或既有结果。',
         '## 2. 三组对照具体比较什么',
         table(['方案','两个训练视图','作用'], [
             ['`paired_repeat`','同一图像、掩膜和文本，两次独立随机前向','控制两次前向和额外训练预算'],
             ['`paired_image`','第二视图镜像图像和掩膜，文本不变','衡量图像镜像增强收益'],
             ['`paired_aug`','相同镜像，同时同步文本中的方向词','检验图文空间对应的额外贡献']]),
         '三组统一使用整图 Dice 修正、同一损失配方和学习率、batch=24、FP32、梯度检查点、相同数据划分；每个视图分别计算批内对比损失。最终评估均为单视图、MC=30、阈值 0.5。`paired_repeat` 是本轮公平对照，不等同于未经修改的官方 MedCLIPSeg。当前方案属于训练策略改进，没有新增模型结构，也没有启用此前失败的细节支路或自适应 β。',
         '## 3. 完整源域验证结果',
         '下表均为同一 Kvasir Val 的 100 张图像，所有图像均含病灶。小目标阈值固定为训练集中阳性图像面积占比的第 25 百分位，即 **5.89897%**，验证集小目标共 **25 张**。均值 ± 样本标准差反映三个训练种子的波动，不是置信区间。',
         table(['方案','总体 DSC (%)','小目标 DSC (%)','NSD (%)','像素召回 (%)','像素精确率 (%)'],
               [[labels[v]]+[ms(v,k) for k in ['DSC_percent','small_DSC_percent','NSD_percent','micro_foreground_recall_percent','micro_foreground_precision_percent']] for v in variants]),
         f"![三种子验证结果]({(HERE/'three_seed_validation.png').as_posix()})",
         '原始九组结果如下；“严重失败”为 DSC < 50% 的阳性图像数，“零重叠”为预测与 GT 没有任何前景交集的图像数，二者并非同一个定义。',
         table(['种子','方案','DSC (%)','小目标 DSC (%)','NSD (%)','严重失败','零重叠'],
               [[s,labels[v]]+[f"{d['summaries'][f'{v}_seed{s}'][k]:.4f}" for k in ['DSC_percent','small_DSC_percent','NSD_percent']]+
                [d['summaries'][f'{v}_seed{s}']['positive_DSC_below_50_percent'],d['summaries'][f'{v}_seed{s}']['zero_overlap_positive']]
                for s in seeds for v in variants]),
         '## 4. 收益来自哪里',
         '下表差值单位均为**百分点**，先按同种子配对，再等权平均。',
         table(['比较','总体 DSC','小目标 DSC','NSD','像素召回','像素精确率'],
               [[labels[a]+' − '+labels[b]]+[signed(d['comparisons'][a+'_vs_'+b]['mean_delta'][k])
                 for k in ['DSC_percent','small_DSC_percent','NSD_percent','micro_foreground_recall_percent','micro_foreground_precision_percent']]
                for a,b in [('paired_image','paired_repeat'),('paired_aug','paired_repeat'),('paired_aug','paired_image')]]),
         '总体 DSC 的主要改善来自加入图像镜像增强；文本同步在此基础上带来较小的附加增益。相对仅图像增强，同步文本的总体 DSC 在三个种子均为正，小目标 DSC 在两个种子为正、seed666 略降。不能把整个完整方案相对基线的提升全部归因于文本同步。',
         table(['种子','同步文本额外总体 DSC','同步文本额外小目标 DSC','逐图改善 / 下降 / 持平','逐图差值中位数'],
               [[s,signed(d['comparisons']['paired_aug_vs_paired_image']['per_seed'][str(s)]['delta']['DSC_percent']),
                 signed(d['comparisons']['paired_aug_vs_paired_image']['per_seed'][str(s)]['delta']['small_DSC_percent']),
                 ' / '.join(str(d['comparisons']['paired_aug_vs_paired_image']['per_seed'][str(s)]['all_cases'][k]) for k in ['improved','declined','tied']),
                 signed(d['comparisons']['paired_aug_vs_paired_image']['per_seed'][str(s)]['all_cases']['median_pp'])] for s in seeds]),
         '为避免把同一图像的三个种子当成三个独立样本，下面先对每张图的种子差值求平均，再检查 100 张图像的分布。该分析用于检查收益集中程度，不改变完整主指标，也不用于删除困难样本。']

c = d['comparisons']['paired_aug_vs_paired_image']['seed_averaged_cases']
small = d['comparisons']['paired_aug_vs_paired_image']['seed_averaged_small_cases']
largest_small = small['best'][0]['delta_pp']
parts += [f"- 同步文本对比仅图像增强：{c['improved']} 张改善、{c['declined']} 张下降、{c['tied']} 张持平；逐图平均差值的中位数为 {c['median_pp']:+.4f} 个百分点。",
          f"- 收益最大的两张图贡献了总体均值差的 {c['top_two_contribution_pp']:+.4f} 个百分点；其余 98 张的平均差值为 {c['remaining_mean_pp']:+.4f} 个百分点。因此需要同时关注平均分和困难病例，不能直接表述为普遍改善了所有病灶。",
          f"- 小目标中，`{small['best'][0]['name']}` 一张图贡献 {largest_small/25:+.4f} 个百分点，占文本同步额外小目标净增益的 {largest_small/25/small['mean_pp']*100:.1f}%；其余 24 张平均变化 {(small['mean_pp']*25-largest_small)/24:+.4f} 个百分点。小目标附加收益尤其集中，尚缺少普遍改善小目标的证据。",
          table(['平均提升最多的图像','DSC 变化（百分点）'], [[r['name'],signed(r['delta_pp'])] for r in c['best']]),
          table(['平均下降最多的图像','DSC 变化（百分点）'], [[r['name'],signed(r['delta_pp'])] for r in c['worst']]),
          '## 5. 残留薄弱点与结论边界',
          '以下为先前验收已经持续追踪的困难图像，每格为三个种子的 DSC（666 / 667 / 668，单位 %）。这些图像用于错误分析，不能凭三个样本代表整个测试集。',
          table(['困难图像','预算匹配基线','仅图像增强','同步图文增强'],
                [[n]+[' / '.join(f"{values[f'{v}_seed{s}']:.2f}" for s in seeds) for v in variants] for n,values in d['hard_cases'].items()]),
          '- 同步图文增强仍存在严重失败，尚不能宣称消除了小病灶漏检。零重叠减少或消失也可能仅代表获得了少量重叠，需结合 DSC 和召回判断。',
          '- 验证集中没有无病灶图像；`negative_false_positive=0` 是空分母下的计数，不能证明正常图像假阳性得到控制。',
          '- Val 已多次用于研究路线开发。这里的三种子稳定性是源域验证证据，不能替代预先锁定的 Test 主指标。三种子未进行显著性认定；不能将重复种子或存在内容重复的数据当作独立临床病例。',
          '- 当前保留原始文本提示协议。部分提示属性与标注信息有关，论文必须说明信息来源及推理时的可获得性，不能无条件宣称与只用图像的模型处于相同输入条件。',
          '- 外域原始清单含重复图像：ColonDB 360 条记录对应 329 个唯一文件内容，CVC300 60 条对应 58 个；两者共享 52 个唯一图像。最终报告应同时保留原清单和唯一图像等权统计，不能称为五个相互独立的外部队列。',
          '## 6. 正在进行的 Test 评估快照',
          f"本次审计取证时，完成 **{len(d['partial_tests'])}/45** 个 Test 任务。以下仅复核已完成任务的清单、CSV 哈希、检查点对应关系和汇总值；本次未逐一重算 Test 的 NPZ/PNG。全部 Test 的最终验收需另行进行。"]

targets = ['Kvasir','ColonDB','ClinicDB','CVC300','BKAI']
complete_seeds = [s for s in seeds if all(f'test/{v}_seed{s}/{t}' in d['partial_tests'] for v in variants for t in targets)]
if complete_seeds:
    parts += ['为避免不平衡比较，下表只汇总三个方法在五个目标上均已完成的种子：**'+', '.join(map(str,complete_seeds))+'**。仅展示阶段性 DSC，不用于调参、选择权重或改变后续评估。']
    rows = []
    for target in targets:
        means = {v:float(np.mean([d['partial_tests'][f'test/{v}_seed{s}/{target}']['DSC_percent'] for s in complete_seeds])) for v in variants}
        rows.append([target]+[f'{means[v]:.4f}' for v in variants]+[signed(means['paired_aug']-means['paired_repeat']),signed(means['paired_aug']-means['paired_image'])])
    parts += [table(['目标','预算匹配基线','仅图像增强','同步图文增强','完整方案 − 基线','完整方案 − 图像增强'],rows),
              '阶段性测试结果存在正负不同的变化，不能仅凭源域验证的正增益就宣称跨域优越。若最终 Test 未证明文本同步的额外价值，应如实缩小贡献表述；本轮最终 checkpoint 和协议继续保持锁定。']
parts += ['## 7. 下一步建议',
          '先完成既定 45 个 Test 任务，统一汇总三个种子的主指标、外域指标、漏检和重复图像敏感性分析，再决定是否保留文本同步作为独立贡献。现在可以整理方法、实验设置和消融表的论文草稿；结果结论应等待完整 Test 验收。本次不新增模块、不追加训练。',
          '## 8. 可复核材料',
          '- `audit.py`：CPU 审计脚本；重跑会重新读取九个最终检查点和 900 份验证预测，在本目录更新证据，不控制队列。',
          '- `audit.json`：检查点哈希、各组指标、参数指纹、逐种子/逐图比较及 Test 状态快照。',
          '- `audit.log`：实际执行日志；`paired_cases.json`：按图像汇总的三种子配对差值。',
          '- `three_seed_validation.png` / `.svg`：三个种子的源域验证曲线；`build_report.py`：报告与图的生成脚本。',
          f"锁定计划 digest：`{d['plan_digest']}`。DSC/NSD 沿用当前项目的作者兼容度量（逐图四位小数后求均值）；NSD 采用原 GT 分辨率下 2 像素容差。"]
(HERE/'九组正式训练验收报告.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
print('REPORT:',HERE/'九组正式训练验收报告.md')
print('FULLY COMPLETED TEST SEEDS AT SNAPSHOT:',complete_seeds)
