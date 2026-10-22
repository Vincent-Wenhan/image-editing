# 真人盲评使用准备

工具准备不等于收到评分。当前所有真实评分仍 pending；页面中 rater_a/b/c 仅为
待分配代号。组织者须实际邀请至少三位独立真人，不能由程序或代理代填、互相讨论，
也不能将单人改代号充当多人。此工作流不自动发送任何邀请。

将 `panels/` 整体复制给评审者（含 images/ 和各自页面），只让其打开自己的
`rater_*/index.html`；**不分发 bundle/private、方法映射、训练预算、研究结论或
代理联系图**。离线页面不联网，CSV 下载到评审者本机，不能自动写回服务器。
请勿在评审前介绍某种条件预期更好或更差。

分别评价编辑完成度与无关内容保留；身份指令应与 source 保持一致。target 是
编辑意图参考，不要求像素复刻。旧位置留有物体且新增一个记 duplicate_object；
只出现一个但未到目标处记 wrong_target_position；不相关色块可记 role_color_copy；
无需编辑却破坏画面记 identity_break。名称只是分类，不能据此透露具体方法。
主类别选最严重项，其余写说明；不清楚选 NA/unjudgeable，不能猜测。

评审完成后由组织者把真实 CSV 放本地不可覆盖目录，再执行：

```bash
image-edit review import --bundle <bundle> --ratings <real-a.csv> \
  --reviewer rater_a --independent --output <new-import-a>
# 对真实b/c分别导入；修订必须另建目录，不覆盖以前的原始CSV。
image-edit review merge --bundle <bundle> --import <import-a> \
  --import <import-b> --import <import-c> --annotations <task-types.csv> \
  --output <new-human-summary>
```

三位完整独立评审之前保持 pending、人工区间为 null。组织者负责真人身份和独立性，
工具只验证声明、任务覆盖、数值和原始字节SHA；NA不当0分。

本轮分成两个独立包：旧60-step checkpoint 的新Dev诊断比较，及新20-step matched
Geometry/Dynamic Binary比较。不同预算不混成“公平方法收益”。干预图不混入正式
方法评分。seed分列，场景是统计单位，失败分组只作描述；完整真人结果另行报告。

本轮实际目录（在role-failure worktree的 `outputs/role_failure_20261022/` 下）：

|包|公开入口|私有导入基准|
|---|---|---|
|旧60-step、48任务/人|`human_existing_01/panels/rater_a/index.html`，b/c同理|`human_existing_01/bundle/`|
|新20-step、48任务/人|`human_pilot_01/panels/rater_a/index.html`，b/c同理|`human_pilot_01/bundle/`|

两包均已生成三套空白表，实际评分0条。将上面命令的bundle替换为对应私有基准，
annotations使用同包的tasks.csv。原始真实CSV可放该实验根的新 `human_imports/`
子目录，每次导入/修订/汇总使用新名字；尚未收到CSV时不要执行填分或假导入。
