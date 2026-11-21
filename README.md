# video-deinterlace

## 题面

老素材做去隔行后，运动画面出现梳齿、静止镜头又发虚：① 拆场按行号从 1 开始编号，场偏移一行；② order 被忽略，底场当成顶场；③ weave 织回时行序颠倒，画面上下翻转；④ bob 用重复行代替插值，边缘出现锯齿；⑤ 运动检测阈值被写死，暗部被误判；⑥ 自适应在运动区也使用 weave，梳齿复现；⑦ 奇数高度不报错，最后一行被丢失；⑧ NaN 像素不校验，扩散到全帧；⑨ 批量处理共用场缓冲，帧与帧之间串数据；⑩ threshold 超出 0..1 不校验。deint/fields.py 是核心，tests/test_fields.py 15个红。要修的是：零基奇偶拆场与场序、weave 行序、bob 边缘插值、按传入阈值检测运动、静止/运动区自适应混合、帧结构校验与 DeintError、NaN 拦截、批量原子回滚与帧间缓冲隔离、threshold 范围校验。

## 运行方式

```powershell
cd D:\gsb-20261119\video-deinterlace
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败全部来自 `deint/fields.py` 中 8 个桩的 `NotImplementedError`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_fields.py` 是评测基准，**不可修改**；只允许修改 `deint/fields.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数；帧是二维灰度 `list[list[float]]`，数值比较容差为 `1e-6`。
