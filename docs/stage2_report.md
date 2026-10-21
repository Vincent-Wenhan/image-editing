# Stage 2 最小数据管线实际状态

设计见 [stage2_design.md](stage2_design.md)。当前完成生成代码及18项CPU/安装CLI测试
（18.23s），Ruff通过；真实Blender渲染和Oracle条件验收尚待运行。CPU fixture中的
EXR是明确标注的测试占位，不是渲染证据，不计为真实Blender验证。

官方Blender二进制下载入口返回403，改为固定官方PyPI bpy4.5.3软件包。新建下载
链路较慢，保留已接收前缀并通过官方wheel URL续传；软件SHA256锁定为
`8badbbeacce3317ca45c0963aa660f2587e2b3b707f618f9ea9af217e4049ec7`。
本次仅调整自己刚启动的软件传输，保留尝试日志；模型/数据没有重下载或移动，
原有任务和第三方源码未改动。软件安装与传输记录在新输出根下。

数据/模型的Oracle接口尚未实现；未训练Role Predictor、Oracle LoRA或Qwen。

## Oracle 接口CPU验证

已实现独立监督/Oracle loader与明确的FLUX上界入口。32项接口/场景/旧推理回归通过
（20.29s），另13项共享tracing/编码/后端测试通过（15.14s）。删除target和原始
反事实RGB后仍能构建source/Oracle，默认拒绝Oracle及非法数值/几何/未消费strength。
GPU角色消费与真实Blender仍待验收；没有训练角色adapter，不能声称Oracle质量上界。

## 软件实际探测

官方wheel校验及可选依赖安装完成。首次导入缺libXrender；已按固定Ubuntu包SHA
本地提取X/GL运行库。使用子进程局部环境后，真实bpy导入和场景构建通过：
4.5.3 LTS，build `67807e1800cc`，background=True，Cycles/device=CPU。
真实显示字符串含LTS，修正版本比较为数值tuple而非放宽pin；增加回归。
20项运行库/场景/CLI相关回归通过（15.26s）。完整渲染仍待运行。
