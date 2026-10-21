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
