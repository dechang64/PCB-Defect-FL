# PCB 缺陷分割数据集 & SOTA 调研（2026-06-26）

## 一、公开数据集对比

### 1. DeepPCB（我们现有）
- **来源**: tangsanli5201/DeepPCB, GitHub
- **图片**: 1500 对（模板+测试），640×640
- **标注**: bbox（缺角坐标）
- **缺陷**: 6类（open, short, mousebite, copper, holes, spur）
- **采集**: 线性 CCD 扫描，~48px/mm
- **问题**: 合成数据（模板对比生成缺陷），低分辨率，无像素级标注

### 2. PKU-Market-PCB
- **来源**: 北京大学 Open Lab on Human Robot Interaction
- **图片**: 1386
- **标注**: bbox
- **缺陷**: 6类（missing hole, mouse bite, open, short, spur, spurious copper）
- **采集**: Photoshop 合成，不是真实拍摄
- **问题**: 同样是合成数据

### 3. PCB-Defect (2025, Data in Brief) ⭐
- **DOI**: 10.1016/j.dib.2025.112296
- **来源**: Islamic University of Technology, Bangladesh
- **图片**: 230 张
- **分辨率**: 800×600 ~ 6000×4000（平均 6.61MP，最大 31MP）
- **标注**: COCO JSON bbox（1704 个标注，~7.4/image），**segmentation 字段留空**（polygon 未使用）
- **缺陷**: 6类（missing pad, mouse bite, open circuit, short circuit, spur, spurious copper）
- **采集**: **真实化学刻蚀** FR4 基板，设计阶段嵌入缺陷，物理刻蚀实现
- **数据可用**: Mendeley Data 公开
- **特点**: 真实缺陷 + 高分辨率，但只有 bbox 标注

### 4. SolDef_AI (2024, MDPI) ⭐⭐
- **DOI**: MDPI 2504-4494/8/3/117
- **来源**: University of Salento, Italy
- **图片**: 1150 张（含正常+缺陷），**228 张有 polygon 标注**
- **分辨率**: 2560×1440
- **标注**: **polygon → instance mask**（JSON 格式，xy 像素坐标）
- **缺陷**: 2大类（位置缺陷 + 焊料缺陷），含 spike/insufficient/excessive
- **采集**: **Cainda USB 显微镜**，HD CMOS，**1000× 放大**，30fps
- **视角**: 3个视角（俯视 + 45° + 侧视）
- **数据可用**: Kaggle 公开 (https://kaggle.com/datasets/f899d21ce26435a9aa74d...)
- **训练**: Mask R-CNN (Detectron2)，两个子数据集分别训练
- **特点**: **唯一的真实显微 + instance mask 标注数据集**

### 5. MeiweiPCB
- **来源**: github.com/youtang1993/MeiweiPCB
- **标注**: semantic segmentation
- **特点**: PCB 表面缺陷语义分割

### 6. PCB-AoI (KubeEdge)
- **来源**: KubeEdge-Ianvs
- **采集**: 真实 AOI 设备
- **标注**: bbox
- **特点**: 异常检测范式（非固定缺陷类型）

---

## 二、SOTA 方法对比

| 方法 | 类型 | 数据集 | 标注 | 性能 | 文献 |
|------|------|--------|------|------|------|
| Mask R-CNN (Detectron2) | 实例分割 | SolDef_AI | polygon mask | baseline | MDPI 2024 |
| Y-MaskNet | 实例分割 | PCB | mask | mAP@0.5:0.95 高 | 2025 |
| SME-YOLO | 检测 | PKU-PCB | bbox | SOTA small defect | arXiv 2026 |
| CM-UNetv2 | 语义分割 | MeiweiPCB | semantic | — | MDPI 2025 |
| SAID (SAM-based) | 分割 | 工业 | mask | — | PMC 2025 |
| **SAM2 fine-tune** | 分割 | 工业 | mask | — | **空白** |

---

## 三、关键结论

### 3.1 数据集选择

**SolDef_AI 最匹配"显微级 PCB"方向**：
- ✅ 真实显微镜拍摄（1000× Cainda）
- ✅ instance mask 标注（polygon → mask）
- ✅ 2560×1440 高分辨率
- ✅ Kaggle 公开下载
- ⚠️ 只有 228 张有标注（少，但可做 few-shot）
- ⚠️ 缺陷类型少（2大类 vs 6类）

**PCB-Defect (2025) 是次选**：
- ✅ 真实化学刻蚀缺陷
- ✅ 高分辨率（最大 31MP）
- ✅ 230 张 + 1704 标注
- ❌ 只有 bbox，没有 polygon/mask（但可自行标注或用 SAM2 辅助标）
- ❌ 不是显微级

### 3.2 SAM2 微调的空白

- **没有论文**在 PCB 缺陷上做过 SAM2 mask_decoder 微调
- SolDef_AI 用的是 Mask R-CNN，不是 SAM2
- SAID 用 SAM 但不是 SAM2，且有场景提示约束
- **这是 PCB-Defect-FL 的机会**：FL + SAM2 微调 + 显微 PCB = 空白领域

### 3.3 实验设计建议

**方案 A: SolDef_AI（推荐）**
1. 下载 SolDef_AI 228张标注图片
2. SAM2 mask_decoder 微调（和 organoid 脚本复用）
3. 对比 zero-shot SAM2 vs 微调 SAM2 vs Mask R-CNN
4. FL 框架：多工厂联邦学习焊点缺陷分割

**方案 B: PCB-Defect (2025)**
1. 下载 230 张高分辨率图片
2. 用 SAM2 zero-shot 生成初始 mask，人工校正
3. SAM2 微调 + FL

**方案 C: 两者结合**
- SolDef_AI 做显微分割（应用价值）
- PCB-Defect 做高分辨率检测（对比 baseline）
- FL 跨数据集验证（泛化能力）

---

## 四、与现有 PCB-Defect-FL 论文的关系

现有论文核心：TrustFL-Defect 七模块 + bbox-pixel gap 11.3pp
现有数据：DeepPCB（合成 640px bbox）

**升级路径**：
1. DeepPCB → SolDef_AI/PCB-Defect（合成 → 真实显微）
2. bbox 评估 → pixel-level 评估（bbox → instance mask）
3. zero-shot SAM2 → 微调 SAM2（无先验 → 有形态先验）
4. 单数据集 → FL 跨数据集（DeepPCB + SolDef_AI + PCB-Defect）

**论文贡献升级**：
- 现有："bbox 高估 11.3pp"（在合成数据上）
- 升级："FL + SAM2 微调在真实显微 PCB 上的像素级缺陷分割"（应用价值）
- 新增贡献：(1) 真实显微数据集验证 (2) SAM2 微调 vs zero-shot 对比 (3) FL 跨数据集泛化
