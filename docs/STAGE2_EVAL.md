# 兩階段模型評估結果與優化方向

`savemodel_two_stage`(stage 1 MobileNetV3-Large class-agnostic proposal detector +
stage 2 ConvNeXt-Tiny crop classifier)在本地驗證集與比賽平台上的量測結果，以及依此排序的優化清單。

評估日期 2026-08-16，對應權重 `savemodel_two_stage/proposal/model.pth`（epoch 119，
`backbone: MobileNet`）與 `savemodel_two_stage/classifier/best.pth`（epoch 29）。
**換權重或換 backbone 後（例如正在進行的 swin_t 版 stage-1）務必重跑本文件底部的指令並更新數字**——
底下的優先序判斷全部綁在這組特定的量測值上。

---

## 1. 平台分數與本地驗證集一致，不是 submission 流程的問題

比賽平台對 `data/test_dateset`（過濾 confidence ≥ 0.4 的 submission）打回 **mAP 0.36**。

本地驗證集（`stratified_split(valid_ratio=0.2, seed=42)`，未做任何 confidence 過濾，
`evaluate_stage2_records` 內部門檻是 0.001）算出來是：

```
two_stage_map50 = 0.3700
```

兩者幾乎一致，代表：

* 沒有座標轉換一類的 silent bug——`utils/tool.py: _scale_to_original()` 用的是跟
  `predict.py` 一樣正確的逐軸 stretch 反變換，不是會讓框整個跑掉的
  `decode_bbox(remove_pad=True)` letterbox 誤用。
* 本地驗證集分數能代表平台測試集分數，泛化沒有明顯落差。
* 剛好也代表 submission 時做的「confidence < 0.4 全部丟掉」這個過濾**沒有明顯拖累分數**——
  平台 0.36 vs 本地無過濾 0.370，差距在雜訊範圍內。這件事優先度低，不是本文件的重點。

0.36～0.37 是**模型目前真實的水準**，不是流程 bug。以下拆解它從哪裡來。

---

## 2. Stage-1（找東西）：recall 是目前最大的瓶頸

`proposal_recall`（class-agnostic 偵測器，IoU / topK 各種組合）：

| IoU | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|
| 0.5 | 64.1% | 72.1% | 78.2% | **80.7%** |
| 0.75 | 44.5% | 47.0% | 48.3% | 48.7% |

**19.3% 的 GT 框在 IoU 0.5 這麼寬鬆的標準下都沒有任何 proposal 命中**，而且 k 從 50 加到
100 只再多拿 2.5 個百分點——不是「候選框給太少」，是偵測器真的沒看到那些物件。IoU 0.75
的 recall 更是卡在 48% 左右，代表就算框到了，位置也普遍偏鬆。

這個數字直接對應下一節誤差拆解裡「完全沒被框到」的 21.2%，是全部誤差來源裡最大的一塊，
比所有分類錯誤加起來（16.2%）還多。

---

## 3. Stage-2（認東西）：oracle 準確率

`evaluate_crop_classifier`，餵完美 GT 裁切框（跳過 stage-1 品質，測分類器本身的上限）：

| 指標 | 數值 |
|---|---|
| accuracy | 62.0% |
| macro recall | 64.0% |

34 類隨機猜基準約 2.9%，62% 是隨機猜的 21 倍，但離「好」還有距離。這是分類器自己的天花板，
不會因為換 stage-1 backbone而改善，要動的話得從分類器本身（架構、資料、loss）下手。

---

## 4. 誤差拆解（`DetectionEval.report()`，2,730 張驗證圖）

| 狀況 | 數量 | 佔比 |
|---|---|---|
| 框對、類別也對 | 5,291 | 62.6% |
| 框對，類別錯（同語意群，見 §5） | 357 | 4.2% |
| 框對，類別錯（跨語意群） | 1,017 | 12.0% |
| **完全沒被框到** | **1,789** | **21.2%** |
| background 上的 false positive | 245,787 | — |

background FP 數字看起來很誇張，但這是評估時刻意把 score_threshold 壓到 0.001（跟
`predict.py` 的理由一樣：AP 要積分整條 PR 曲線）的預期結果，這些幾乎都是近零信心的雜訊，
不是本文件要處理的問題。

**排序：定位失敗（21.2%）> 分類錯誤合計（16.2%）> 同群分類錯誤（4.2%）。** 這決定了 §6 的優先順序。

---

## 5. 分類錯誤：實際混淆的對象，跟 `utils/metrics.py: GROUPS` 原本假設的不同

現有分群（`utils/metrics.py`）：

```
container:     plastic_bottle, non_food_plastic_bottle, glass_bottle, metal_can,
               non_pet_food_beverage_container, drink_carton, non_pet_food_container,
               non_food_plastic_container, aluminum_packaging
cup_tableware: takeaway_beverage_cup, cup, disposable_food_container,
               disposable_tableware, foam_container, plastic_lid
net_rope:      fishing_net_rope, net_like_item, fishing_gear, fish_trap_and_bait
float:         foam_buoy_float, fishing_buoy_float, soft_float
fragment:      anthropogenic_fragment, other, textile
```

top confusions（ground truth → 被判成什麼）：

```
133  anthropogenic_fragment          -> foam_buoy_float           跨群
 75  foam_buoy_float                 -> anthropogenic_fragment    跨群
 54  other                          -> anthropogenic_fragment    同群 (fragment)
 30  non_pet_food_beverage_container -> non_food_plastic_bottle   同群 (container)
 29  foam_container                 -> anthropogenic_fragment    跨群
 27  foam_container                 -> foam_buoy_float           跨群
 24  takeaway_beverage_cup           -> cup                       同群 (cup_tableware)
 24  soft_float                     -> anthropogenic_fragment    跨群
```

**只有 26% 的類別錯誤發生在現有 5 個語意群內。** 全資料集最大的一組混淆——
`anthropogenic_fragment`（碎片群）↔ `foam_buoy_float`（浮球群）——現有分群認為它們不相關，
但實際上「風化破碎的保麗龍浮球碎片」「保麗龍餐具碎片」「軟質浮球」「不明碎片」外觀本來就很像
（無固定形狀、白色/塑膠質感的破碎物），分類器分不出來。

這直接解釋了為什麼 `anthropogenic_fragment`（全資料集第二大類，本次驗證集 1,722 個框）
AP 只有 0.113、`other` 只有 0.101——量體最大的類別之一，表現卻墊底，而且問題不在
docs/DATASET.md §7.3 原本假設的那 5 個「幾何相似」群組上。

---

## 6. 每類 AP（依 AP 由低到高排序）

29 類在本地驗證集有 GT；5 類（`rare_threshold=150` 規則下全數留在訓練集，見
docs/DATASET.md §8）在本地完全無法評分，但比賽測試集會實際評到。

| 類別 | GT 數 | AP |
|---|---|---|
| other | 373 | 0.101 |
| anthropogenic_fragment | 1,722 | 0.113 |
| fishing_net_rope | 379 | 0.170 |
| foam_container | 115 | 0.198 |
| soft_float | 423 | 0.201 |
| net_like_item | 108 | 0.249 |
| fishing_gear | 41 | 0.279 |
| disposable_tableware | 79 | 0.295 |
| takeaway_beverage_cup | 103 | 0.311 |
| plastic_bag | 261 | 0.323 |
| straw | 175 | 0.332 |
| non_pet_food_beverage_container | 112 | 0.352 |
| non_food_plastic_bottle | 222 | 0.360 |
| cup | 94 | 0.373 |
| cigarette_butt | 133 | 0.396 |
| textile | 103 | 0.397 |
| toothbrush | 38 | 0.402 |
| foam_buoy_float | 633 | 0.402 |
| food_wrapper | 195 | 0.412 |
| plastic_lid | 68 | 0.413 |
| disposable_food_container | 179 | 0.434 |
| glass_bottle | 257 | 0.450 |
| fish_trap_and_bait | 40 | 0.458 |
| plastic_bottle_cap | 324 | 0.495 |
| metal_can | 110 | 0.497 |
| plastic_bottle | 1,492 | 0.544 |
| fishing_buoy_float | 440 | 0.552 |
| drink_carton | 124 | 0.593 |
| lighter | 111 | 0.628 |
| non_food_plastic_container | 0 | n/a（本地無 GT） |
| cigarette_pack | 0 | n/a（本地無 GT） |
| syringe_needle | 0 | n/a（本地無 GT） |
| aluminum_packaging | 0 | n/a（本地無 GT） |
| non_pet_food_container | 0 | n/a（本地無 GT） |

**mAP@0.5 = 0.3700（29/34 類有 GT）**

---

## 7. 優化優先順序

1. **Stage-1 recall（進行中）。** 21.2% 的物件無論分類器多強都救不回來，是全部誤差裡最大的一塊。
   換 swin_t backbone 重練 proposal detector（已在跑）理論上等比例拉高全部 34 類的 AP 上限，
   槓桿最大。訓完後重跑 §9 的指令，比對 `proposal_recall` 是否明顯提升。

2. **群內競爭損失（docs/DATASET.md §12.3）不要照抄原本的 5 個語意群。** 那個方案只覆蓋
   26% 的類別錯誤，真正的大頭是 fragment/float/cup_tableware 之間的跨群混淆（§5）。要做的話
   應該先把 `anthropogenic_fragment`、`other`、`foam_buoy_float`、`foam_container`、
   `soft_float` 抓出來當一組「破碎/風化雜物」群，而不是用現有幾何分群。

3. **5 個稀有類別本地永遠看不到分數，但平台會評。** `aluminum_packaging` 等 5 類因
   `rare_threshold=150` 全留訓練集（docs/DATASET.md §8），docs §14.5 的天花板估算顯示
   `aluminum_packaging` 就算給完美框也只有 14% 準確率——不是架構問題，是資料量問題。
   Repeat factor sampling（docs §9.4，目前只有 `crop_sample_weights` 在分類器取樣端做
   inverse-sqrt 重加權，偵測器/整體資料量沒動過）是比較實際的下一步。

4. **加 flip-TTA 到兩階段推論，免重訓可立即測。** `predict_stage2.py` /
   `utils/tool.py: infer_stage2_proposals()` 目前完全沒有 TTA，單階段的 `predict.py`
   有做水平翻轉 TTA。stage-1 recall 是目前最大瓶頸，flip-TTA 對 recall 通常最直接見效。

5. **submission 的 confidence 過濾門檻不是重點。** 見 §1，本地未過濾與平台過濾後的分數
   幾乎一致，優先度最低。

---

## 8. 重現方式

```
uv run python evaluate_stage2.py \
  --detector-weights savemodel_two_stage/proposal/model.pth \
  --detector-backbone mobilenet_v3_large \
  --detector-class-agnostic \
  --classifier savemodel_two_stage/classifier/best.pth \
  --manifest savemodel_two_stage/cache/valid_records.json \
  --output eval_stage2_result.json
```

換了 stage-1 backbone 或權重之後，`--detector-weights` / `--detector-backbone` 要跟著換，
且 `--manifest` 快取跟權重的 mtime 綁在一起，backbone 換了要嘛指向新的快取路徑、
嘛加 `--rebuild-manifest` 讓它重新生成 proposal（2,730 張驗證圖，GPU 上約 10 分鐘）。
