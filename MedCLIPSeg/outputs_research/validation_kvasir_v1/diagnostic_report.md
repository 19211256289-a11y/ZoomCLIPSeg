# 固定权重提示与空间诊断

已完成 54/54。仅源域验证集；属于事后机制诊断，不是独立测试或新方法性能。

generic 仅改变推理输入，权重仍曾使用属性文本训练；不能把它当作重新训练过的公平无属性基线。no_location 仍含其他属性。镜像输出已反变换，不与原预测做 TTA。

|条件|训练方案|总体 DSC % 均值 ± 标准差|小目标 DSC %|
|---|---|---:|---:|
|original|paired_repeat|89.0512 ± 0.2759|85.6881 ± 1.0507|
|original|paired_image|89.9870 ± 0.2881|86.7895 ± 0.5302|
|original|paired_aug|90.3593 ± 0.2520|87.7005 ± 0.6193|
|no_location|paired_repeat|88.4609 ± 0.3756|85.3601 ± 0.8502|
|no_location|paired_image|89.4675 ± 0.0741|86.8099 ± 0.8466|
|no_location|paired_aug|89.3778 ± 0.1340|86.6848 ± 0.1566|
|generic|paired_repeat|84.3286 ± 0.5048|77.7988 ± 4.6468|
|generic|paired_image|85.6257 ± 0.3329|80.8208 ± 4.1533|
|generic|paired_aug|83.7709 ± 1.5393|81.1089 ± 5.6999|
|opposite_location|paired_repeat|88.5273 ± 0.6221|85.0615 ± 0.6411|
|opposite_location|paired_image|90.1754 ± 0.0322|87.0109 ± 0.7803|
|opposite_location|paired_aug|89.4058 ± 0.1414|86.7331 ± 0.4981|
|hflip_sync|paired_repeat|87.9291 ± 0.2634|83.9613 ± 0.5550|
|hflip_sync|paired_image|89.4562 ± 0.3734|85.6292 ± 0.3586|
|hflip_sync|paired_aug|89.3921 ± 0.1965|86.3249 ± 0.9114|
|vflip_sync|paired_repeat|88.4781 ± 0.0888|84.6717 ± 0.4064|
|vflip_sync|paired_image|89.0022 ± 0.1317|86.0620 ± 0.3223|
|vflip_sync|paired_aug|89.2867 ± 0.4901|86.0763 ± 0.8130|

|条件|同步文本相对图像增强的 DSC 变化 pp|该优势相对原提示变化 pp|
|---|---:|---:|
|original|+0.3723|+0.0000|
|no_location|-0.0897|-0.4620|
|generic|-1.8548|-2.2271|
|opposite_location|-0.7696|-1.1419|
|hflip_sync|-0.0641|-0.4364|
|vflip_sync|+0.2845|-0.0878|

|条件|方案|有效改写数|全体概率 MAE|全体二值不一致率|方向词子集 DSC 变化 pp|
|---|---|---:|---:|---:|---:|
|original|paired_repeat|0|0.000000|0.000000|+0.0000|
|original|paired_image|0|0.000000|0.000000|+0.0000|
|original|paired_aug|0|0.000000|0.000000|+0.0000|
|no_location|paired_repeat|100|0.010061|0.010078|-0.9403|
|no_location|paired_image|100|0.010099|0.010088|-0.8412|
|no_location|paired_aug|100|0.010937|0.010946|-1.6071|
|generic|paired_repeat|100|0.033674|0.033601|-5.0789|
|generic|paired_image|100|0.033881|0.033702|-4.6106|
|generic|paired_aug|100|0.042678|0.042610|-7.4048|
|opposite_location|paired_repeat|66|0.008181|0.008222|-0.7938|
|opposite_location|paired_image|66|0.006814|0.006748|+0.2855|
|opposite_location|paired_aug|66|0.008985|0.008930|-1.4447|
|hflip_sync|paired_repeat|56|0.027996|0.027971|-1.1693|
|hflip_sync|paired_image|56|0.017740|0.017545|-0.7603|
|hflip_sync|paired_aug|56|0.016896|0.016777|-1.0410|
|vflip_sync|paired_repeat|37|0.023882|0.023829|-0.0710|
|vflip_sync|paired_image|37|0.018698|0.018463|-1.1849|
|vflip_sync|paired_aug|37|0.017866|0.017554|-1.1171|

逐图结果包含有效文本改写比例、含方向词子集、概率 MAE 和二值预测不一致率。opposite_location 对 center-only 文本无改写，其输出必须与原提示一致。
本诊断不会自动触发新训练，不会修改正式 Test 主表，也不从这些条件中挑选测试提示。
