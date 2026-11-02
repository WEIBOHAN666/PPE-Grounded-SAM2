# 数据与关联约定

## 输入：每张图片一个 JSON 对象

串联入口 `scripts/check_prediction_association.py` 接收一个或多个预测 JSON。以下是仅包含必填字段的示例：

```json
{
  "image": "example.jpg",
  "width": 640,
  "height": 480,
  "instances": [
    {"instance_id": 42, "label": "person", "detection_score": 0.92, "box_xyxy": [0, 0, 100, 200]},
    {"instance_id": 7, "label": "head", "detection_score": 0.87, "box_xyxy": [20, 10, 60, 50]},
    {"instance_id": 99, "label": "helmet", "detection_score": 0.91, "box_xyxy": [25, 10, 55, 30]}
  ]
}
```

| 字段 | 约定 |
| --- | --- |
| `image` | 非空图片路径；绝对路径直接使用，相对路径先相对运行目录查找，再相对 JSON 所在目录查找 |
| `width` / `height` | 正整数，必须与原图实际尺寸一致 |
| `instances` | 检测实例列表；允许 `[]` |
| `instance_id` | 在单张图片中唯一的整数；不要求连续，不是列表下标；`0` 有效 |
| `label` | 精确为 `person`、`head`、`helmet` 之一 |
| `detection_score` | `[0, 1]` 内的有限数值；拒绝布尔值、NaN 和无穷大 |
| `box_xyxy` | 四个有限数值组成的列表 `[x1, y1, x2, y2]`，使用原图像素坐标；`0 <= x1 < x2 <= width`，`0 <= y1 < y2 <= height` |

坐标原点在图片左上角，x 向右增加，y 向下增加。COCO 的 `[x, y, width, height]` 不能直接当作 `box_xyxy`；需先转换为 `[x, y, x + width, y + height]`。

模型导出的 JSON 还包含 `prompt`、`instance_count`，实例中还包含 `raw_phrase`、`mask_score`、`mask_area_pixels`、`mask_file`。后处理保留这些字段，但当前不使用掩码匹配。`mask_file` 原本相对模型导出目录；串联脚本只复制原图和 JSON，不搬运掩码，复查目录内保留的该字段不是新的掩码路径。

## 中间结果：实例 ID 关联

`associate_prediction(payload, min_score=0.6)` 返回输入的深拷贝，保留全部原始实例，重新计算并写入：

```json
{
  "association_min_score": 0.6,
  "associations": [
    {"person_id": 42, "head_id": 7, "helmet_id": 99}
  ]
}
```

上面仅展示新增字段。实际对象还保留 `image`、尺寸和 `instances` 等字段。输入中已有的 `associations` 会被重新计算，不作为手工关联直接采用。

- 仅分数达到阈值的实例参与关联，防止低分目标占用配对。
- 每个达标 `person` 对应一条关联，顺序与输入中的达标人员顺序一致。
- 找不到头部时，`head_id` 和 `helmet_id` 都为 `null`；找到头部但找不到安全帽时，仅 `helmet_id` 为 `null`。
- 没有达标人员时，`associations` 为 `[]`。孤立的头部和安全帽仍可计入类别数量，但不会单独生成人员复查记录。
- 底层关联算法返回类别列表中的索引，适配器将索引映射为原始 `instance_id`，避免不同类别、不同排序造成 ID 错配。

## 输出：整批图片一个 report.json

```json
{
  "counts": {"person": 2, "head": 1, "helmet": 0},
  "review_images": ["a.jpg", "b.jpg"],
  "reviews": [
    {"person_id": 42, "reason": "helmet_unmatched", "review_status": "pending", "image": "a.jpg"},
    {"person_id": 8, "reason": "head_unmatched", "review_status": "pending", "image": "b.jpg"}
  ],
  "errors": []
}
```

该示例表示：两张成功处理的图片中，共有两个人员目标、一个头部目标、没有达标安全帽目标；两个人员各产生一条复查记录。

| 字段 | 含义 |
| --- | --- |
| `counts` | 成功处理图片中达标检测实例的数量，包含未匹配实例；不是去重后的现实世界人数 |
| `review_images` | 至少有一条复查记录的图片文件名，按处理顺序去重 |
| `reviews` | 人员级复查记录，通过 `(image, person_id)` 定位到具体人员 |
| `errors` | 可记录的处理错误，每项包含 `image`、`reason`、`message` |

复查原因：

- `head_unmatched`：该人员没有达标的关联头部，优先于安全帽检查。
- `helmet_unmatched`：已有达标的关联头部，但没有达标的关联安全帽。

关联完整的人员不产生复查记录。没有达标目标时仍保存一份原尺寸图片，计数为零，复查列表为空。

## 错误行为

串联入口会先校验全部预测与原图。JSON 解析失败、重复 ID、未知类别、无效坐标、找不到图片、图片解码失败、尺寸不一致或图片重名时，命令失败；不会保证生成汇总报告。这是入口的快速失败行为。

复查脚本内部可记录以下错误（如直接调用该脚本，或文件在预检后发生变化）：

| `reason` | `message` |
| --- | --- |
| `invalid_instance` | 检测目标的数据不合法 |
| `duplicate_instance_id` | 检测目标的instance_id重复 |
| `invalid_association` | 关联记录引用的目标不存在或类别不匹配 |
| `image_not_found` | 图片文件不存在 |
| `image_read_failed` | 图片文件存在，但无法读取 |

图片处理返回失败时，该图只合并错误，不合并计数和复查记录。保存图片失败、无法创建目录等 I/O 异常会中止处理，当前不提供事务回滚；排错后使用新的输出目录重试。

## 输出目录

假设指定 `--output-dir outputs/review/example`：

```text
outputs/review/example/
├── input/
│   ├── images/                      # 原图副本
│   └── predictions/                 # image 改为文件名、增加 associations 的 JSON
├── output/
│   ├── images/                      # 普通框、标签及复查标记；保留原文件名与尺寸
│   └── report.json                  # 整批最终报告
└── integration_summary.json         # 输入来源、关联结果与产物路径
```

普通目标按类别绘制边界框；需要复查的人员绘制红色框及 `REVIEW:head_unmatched` 或 `REVIEW:helmet_unmatched` 文本。两类标记绘制在同一张内存图像上，最后统一保存。生成的来源摘要可能包含本机绝对路径，默认位于忽略的 `outputs/` 中。
