# Ground Truth 标注指南

## 文件说明

- `ground_truth_template.json`: 标注模板文件，包含所有20张vlm_attempt图片的空白模板

## 标注步骤

### 1. 填写 object_counts（对象计数）

对于每个对象类型，填写**实例数量**（即有多少个独立的物体）：

- **car**: 汽车数量
- **truck_bus**: 卡车/公交车数量
- **pedestrian**: 行人数量
- **bicyclist**: 骑自行车的人数量
- **motorcyclescooter**: 摩托车/滑板车数量
- **suvpickuptruck**: SUV/皮卡数量
- **building**: 建筑物数量（通常按栋计算）
- **tree**: 树木数量
- **trafficlight**: 交通灯数量
- **trafficcone**: 交通锥数量
- **signsymbol**: 标志/符号数量
- **fence**: 围栏数量（按段计算）
- **column_pole**: 柱子/杆子数量
- **road**: 道路（通常为1，表示道路存在）
- **sidewalk**: 人行道（通常为1，表示人行道存在）
- **roadshoulder**: 路肩（通常为1）
- **parkingblock**: 停车块数量
- **lanemkgsdriv**: 驾驶车道标线（通常为1）
- **lanemkgsnondriv**: 非驾驶车道标线（通常为1）
- **sky**: 天空（通常为1）
- **vegetationmisc**: 其他植被数量
- **wall**: 墙壁数量
- **bridge**: 桥梁数量
- **tunnel**: 隧道数量
- **archway**: 拱门数量
- **train**: 火车数量
- **animal**: 动物数量
- **child**: 儿童数量
- **cartluggagepram**: 手推车/行李车/婴儿车数量
- **othermoving**: 其他移动物体数量
- **misc_text**: 文本/文字数量

**注意**：
- 如果某个类别不存在，填写 `0`
- 计数要准确，这是评估计数准确度的关键
- 对于背景类（如road, sky），通常为1表示存在

### 2. 填写 present_classes（存在的类别列表）

列出**所有在图片中出现的类别**（即使数量为0，只要有任何像素属于该类，就列出）：

例如：
```json
"present_classes": [
  "car",
  "truck_bus",
  "pedestrian",
  "building",
  "tree",
  "road",
  "sidewalk",
  "sky",
  "signsymbol"
]
```

**注意**：
- 使用小写字母
- 类别名称与 object_counts 中的键名一致
- 这是用于幻觉检查的：如果VLM描述中提到了不在这个列表中的类别，就是幻觉

### 3. 填写 notes（可选）

可以添加一些备注，例如：
- 模糊不清的情况
- 部分遮挡的物体
- 特殊情况说明

## 示例

```json
{
  "image_name": "0001TP_006840.png",
  "object_counts": {
    "car": 1,
    "truck_bus": 1,
    "pedestrian": 2,
    "bicyclist": 0,
    "motorcyclescooter": 0,
    "suvpickuptruck": 0,
    "building": 3,
    "tree": 1,
    "trafficlight": 1,
    "trafficcone": 0,
    "signsymbol": 5,
    "fence": 0,
    "column_pole": 1,
    "road": 1,
    "sidewalk": 1,
    "roadshoulder": 1,
    "parkingblock": 0,
    "lanemkgsdriv": 1,
    "lanemkgsnondriv": 1,
    "sky": 1,
    "vegetationmisc": 0,
    "wall": 0,
    "bridge": 0,
    "tunnel": 0,
    "archway": 0,
    "train": 0,
    "animal": 0,
    "child": 0,
    "cartluggagepram": 0,
    "othermoving": 0,
    "misc_text": 0
  },
  "present_classes": [
    "car",
    "truck_bus",
    "pedestrian",
    "building",
    "tree",
    "trafficlight",
    "signsymbol",
    "column_pole",
    "road",
    "sidewalk",
    "roadshoulder",
    "lanemkgsdriv",
    "lanemkgsnondriv",
    "sky"
  ],
  "notes": "有一个行人部分被遮挡，但可以清楚看到，计入pedestrian。signsymbol包括多个标志牌。"
}
```

## 使用标注文件

填写完成后，保存为 `ground_truth_annotations.json`，然后在评估脚本中使用：

```bash
python evaluate_vlm_descriptions.py \
    --baseline_dir ./vlm_baseline_results \
    --mask_results_dir ./vlm_mask_results \
    --mask_dir ../UNet_baseline/test_results \
    --images_dir ../data/vlm_attempt \
    --ground_truth_file ./ground_truth_annotations.json \
    --class_dict ../data/class_dict.csv \
    --output ./vlm_evaluation_results.json
```

## 提示

1. **计数准确性最重要**：这是评估的核心指标
2. **一致性**：确保 object_counts 和 present_classes 一致
3. **仔细检查**：可以多次检查，确保没有遗漏或错误计数
4. **参考原图**：建议同时查看原图和分割掩码来确认
