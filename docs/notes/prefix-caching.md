# 改造 A · 给 prefix caching 加开关与命中率统计

> 目标：让 nano-vllm 里**硬编码常开**的 prefix caching 变成可配置项，并能观测命中率。
>
> 动机很直接：原版引擎无法回答"prefix cache 到底带来多少收益"，因为没有对照组。只能比"冷 vs 热"，而这会把"首次调用开销"和"缓存收益"混在一起。加上开关后，才有严格的 **关闭 vs 开启**。

改动 3 个文件、净增约 22 行。**整理说明**：这段改动已经合进 `nanovllm/`（`Config.enable_prefix_caching` + 命中率统计），原来的 patch 文件在整理时裁掉了；文中引用的探针脚本（`scripts/cache_miss_cost.py` 等）同样不在仓库里，数据见 `results/bench_cache_ablation.csv`。

```bash
# 在 nano-vllm 仓库根目录应用
patch -p1 < /path/to/prefix-caching-toggle.patch
```

---

## 一、为什么原版做不了这个实验

`BlockManager.__init__` 里没有任何开关，`can_allocate()` 无条件做前缀查找：

```python
# 改前
def can_allocate(self, seq):
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):     # ← 无条件查
        ...
```

想让"关闭"生效，最省事的错误做法是**清空 `hash_to_block_id`**。这能跑，但会留下两处隐患：

1. `hash_blocks()` 每个 step 仍在算链式 xxhash 并把结果写回字典，**CPU 开销没消失**，等于"关了个假的"；
2. `_allocate_block()` 里的 `if block.hash != -1 and ...` 分支仍会触发删除逻辑。

所以正确做法是在**三个层次**都短路：配置层（不传）、查找层（不查）、写入层（不写）。

## 二、三处改动

### 1. `nanovllm/config.py` — 加字段

```python
kvcache_block_size: int = 256
enable_prefix_caching: bool = True     # ← 新增
num_kvcache_blocks: int = -1
```

默认 `True`，保持与原版行为一致（不改变既有基准的可比性）。

### 2. `nanovllm/engine/scheduler.py` — 透传

```python
self.block_manager = BlockManager(
    config.num_kvcache_blocks,
    config.kvcache_block_size,
    config.enable_prefix_caching,
)
```

### 3. `nanovllm/engine/block_manager.py` — 三处短路 + 统计

**（a）构造与统计字段**

```python
def __init__(self, num_blocks, block_size, enable_prefix_caching=True):
    ...
    self.enable_prefix_caching = enable_prefix_caching
    self.cache_lookup_blocks = 0
    self.cache_hit_blocks = 0

@property
def cache_hit_rate(self) -> float:
    if self.cache_lookup_blocks == 0:
        return 0.0
    return self.cache_hit_blocks / self.cache_lookup_blocks
```

**（b）`can_allocate` — 包住查找循环 + 记账**

```python
if self.enable_prefix_caching:
    for i in range(seq.num_blocks - 1):
        ...
if len(self.free_block_ids) < num_new_blocks:
    return -1
# 只在真正可分配时记账
self.cache_lookup_blocks += seq.num_blocks - 1
self.cache_hit_blocks += num_cached_blocks
return num_cached_blocks
```

**（c）`hash_blocks` — 写入侧短路**

```python
def hash_blocks(self, seq):
    if not self.enable_prefix_caching:
        return
    ...
```

---

## 三、三个设计决策（面试会被追问的就是这些）

### 决策 1：为什么记账放在 `return -1` 之后？

`can_allocate` 会因 KV 池不足返回 -1，此时 `Scheduler.schedule()` 会 `break`，**下一个 step 会对同一个 seq 再次调用 `can_allocate`**。如果把累加写在循环里或提前，同一个请求的 lookup 会被重复计数，命中率被系统性稀释——显存越紧，稀释越严重，恰好在你最需要看指标的时候最不准。

放在 `return -1` 之后，保证"一次请求只记一次账"。这是统计口径问题，不是性能问题，但错了会让结论反向。

### 决策 2：为什么 `num_blocks - 1` 而不是 `num_blocks`？

最后一个块可能没写满。它的内容会随着后续 token 生成继续变化，因此不参与复用查找。这是原版就有的设计，我在写统计时**必须跟着这个口径**，否则分母比分子大 1，命中率永远到不了 100%。

实测印证：2048+16 token 的 prompt = 9 个 block，lookup = 8，命中 8/8 = 100%。

### 决策 3：关闭时只短路读取，不重建数据结构

关掉后 `hash_to_block_id` 保持空字典，`Block.hash` 恒为 -1。看起来 `_allocate_block` 里那段清理逻辑成了死代码，但**保留它**是对的：`BlockManager` 可能在运行期被复用（比如将来做多引擎实例），结构一致性比省几行代码重要。

---

## 四、实验设计与结果

### 负载

单请求、2048 token 共享前缀 + 16 token 不同后缀，每 5 轮一组、独立进程，共 5 组（关闭 2 组、开启 3 组）。

### 一个必须修的坑：预热

第一版直接测，冷轮测得 **1.18s**，而热轮只有 45ms。这 1.18s 里有 1.1s 是进程内第一次 CUDA 调用（cuBLAS autotune、allocator 扩容、flash-attn kernel 选择），**和 cache 完全无关**。

修法是在测量前用一段**人造 id 序列**（`[(i % 10000) + 100 for i in range(2048)]`）跑一次同等长度的 prefill：

- 不能用真实语料 —— `build_corpus` 里 `_PARA * 40` 是周期性文本，换起点很可能仍与共享前缀撞块，反而制造假命中；
- 人造序列与语料 token 分布完全不同，能保证 0 块重叠。

预热后冷轮稳定在 86–102ms，与关闭组的 82–88ms 同一量级，这才可比。

### 结果

最终统计用 **12 个独立进程 × 5 轮、开/关交替**（顺序交替，消除"先跑的被惩罚"）：

| 场景 | TTFT 均值 | 标准差 | 样本 | 相对关闭 |
|---|---|---|---|---|
| 关闭 · 稳态 | **83.7 ms** | 1.8 | 24 轮 / 6 进程 | 1.00× |
| 开启 · 命中 · 稳态 | **40.8 ms** | 3.4 | 21 轮 / 7 进程 | **↓51.3%（2.05×）** |
| 关闭 · 进程首轮 | 88.1 ms | 4.0 | 6 进程 | 1.00× |
| 开启 · 未命中 · 进程首轮 | 91.7 ms | 6.0 | 7 进程 | **+4.1%（噪声内）** |
| 开启 · 首次命中（过渡） | 50.0 ms | 9.1 | 7 进程 | —（graph 重捕获） |

块级命中率：稳态 **100%（63/63）**，含未命中轮 83%（70/84）。

### 未命中到底付出什么：微基准拆解

`scripts/cache_miss_cost.py` 把未命中路径逐段计时（block_size=256，2048 token prompt）：

| 环节 | 耗时 | 说明 |
|---|---|---|
| 单块 `compute_hash`（256 token） | 9.9 µs | 其中 **98% 是 `np.array(token_ids).tobytes()`**，xxhash 本体仅 0.2–0.9 µs |
| `can_allocate` 链式 hash，首块失配 | **9.8 µs** | 冷缓存下第 1 块就 `break`，O(1) |
| `can_allocate` 链式 hash，8 块尾部才失配 | 75.7 µs | 最坏情况，但仍在一百微秒内 |
| 命中后 8 块 `token_ids` 逐块比对 | 17.9 µs | 只在命中时才走 |
| `hash_blocks` 写入 8 块 | **71.2 µs** | hash + 切片 + 字典写入 |
| **未命中合计** | **81 µs = 0.08 ms** | 占 83.7ms TTFT 的 **0.1%** |

外加三笔不在这 81µs 里的：

| 成本 | 量级 | 说明 |
|---|---|---|
| decode 阶段 `hash_blocks` 短路 | 1.1 ms / 64 并发 128 step | `start == end` 立即 return，但 Python 函数调用本身 0.2 µs × 8192 次 |
| 命中块被 `ref_count` 钉住 | 8 块 × 28MB = **224MB**（池子 5%） | KV 池有效容量下降，3584 token 前缀要 392MB（8.7%） |
| 显存不足时每 step 重算 hash 链 | 46.6–75.7 µs / step | `can_allocate` 返回 -1 时 seq 留在队首，下 step 重算；长前缀 + 池子吃紧时是 O(steps × blocks) |
| CPU 常驻内存 | 331 KB 指针 + ≤1.1 MB int | 161 块 × 256 token 的 `token_ids` 副本 |

### 三个结论

1. **省下的是 prefill，天花板明确。** 2048 token prefill 在关闭组里占 83.7ms 的绝大部分；命中后只算 16 token，剩下 40.8ms 是纯固定开销（调度 + pinned H2D + 一次 decode + 采样）。想再降就得打 H2D 和调度，不是 attention kernel。
2. **未命中的代价接近于零（0.08ms），这个优化近乎免费。** 关键在于 `can_allocate` 的探测是 O(1) 短路：第一个块不命中就 `break`，所以永远不会"算完 2048 token 才发现没命中"。实测 +3.6ms（88.1 → 91.7）小于两组标准差（4.0 / 6.0 ms），统计上不显著。
3. **第 1 次命中比稳态慢 23%（50.0 vs 40.8ms）。** 7 个进程都是这个模式：首次命中时 CUDA graph 尚未捕获"13 块而非 8 块"这个 shape、allocator 还在扩容。对外报数字取稳态（51.3%），简历上取更保守的全量口径。

### 一次自我修正

上一版这里写的是"冷启动比关闭慢 **13.8%**，优化不是免费的"。那是错的：

- **口径不一致**：95.4ms 是"开启·首轮"，83.8ms 是"关闭·全程均值（含稳态）"。关闭组自己的首轮也有 88.1ms（比稳态高 4.4ms）——首轮的 CUDA 残留污染两组都有，不能只算一边。
- **样本不足**：n=3、标准差 8.3ms，观测差 11.6ms 连一个标准差都不到。

修正后同口径对比是 +3.6ms（仍在噪声内），微基准确认真实成本 0.08ms。**教训：小样本 + 混口径是假阳性的标准配方。**

### 对普通负载无副作用

关闭组顺带跑了完整负载（C=1/8/32/64，prompt 256 / 输出 128，请求间前缀不同）：C=64 吞吐 **2946 tok/s**，与 baseline 的 2994.6 差 1.6%（在单次运行波动范围内）。合理——这类负载本来就几乎不命中，开关只是去掉了查表开销。

---

## 五、当前局限（写进 README 的 Known Issues）

- **统计未覆盖被淘汰的块。** 关掉时 hash 表不清、命中率不重置，长时间运行后指标反映的是全生命周期累计，不是滑动窗口。生产实现（vLLM）会用 LRU 淘汰 + 分桶统计。
- **没有跨引擎实例的共享。** 单进程内有效，多实例要做中央索引。
- **`num_blocks - 1` 的口径让 1 个块的 prompt 永远 0 命中**，这是设计使然，不是 bug。
- **未做 block 粒度实验。** 256/512 两种粒度下命中收益的差异没测——理论上粒度越大，尾部浪费越多（最后不满的块越大），但 8GB 卡上 block table 太小，差异可能测不出来。

---

## 六、这道题怎么答（面试）

**Q：你为什么要加这个开关，原版不是已经有 prefix cache 了吗？**

> 原版有，但没有观测手段。我想量化它的收益，就必须有对照组。原版只能测"冷 vs 热"，而冷的那一轮会被进程内首次 CUDA 调用污染——我第一版测出 1.18s，实际 cache 收益只有 40ms 量级。加上开关之后才能做"关闭 83.7ms vs 命中 40.8ms（↓51.3%）"。更关键的是，我还用微基准把未命中路径拆到了微秒级：81µs，占 TTFT 的 0.1%——所以这个优化的成本几乎为零，收益只在命中时兑现，不必做任何权衡。

**Q：为什么命中率要按块统计，不按请求统计？**

> 请求级的"命中/未命中"是二值，丢失了命中深度。一个 8 块的 prompt 命中 3 块和命中 7 块，收益差一倍。按块统计能算"前缀复用率"，也才能和 KV 池容量、block_size 这些参数联动分析。

**Q：你改的时候有没有想过影响别的路径？**

> 三处都要短路，只改 `can_allocate` 是假的：`hash_blocks` 每个 step 还在算 xxhash 写字典，CPU 开销没消失。我第一版就只改了查找侧，结果"关闭"组的 TTFT 只比开启组慢一点，看着像 cache 没用，实际上是关了但还在付成本。

---

## 七、复现命令

```bash
cd ~/nano-vllm && source .venv/bin/activate
cp /mnt/d/WorkBuddy/nano-vllm/bench_suite.py .

# 关闭组
python bench_suite.py --model models/Qwen3-0.6B \
  --single '{"enable_prefix_caching": false}' --tag cache_off --shared-only

# 开启组
python bench_suite.py --model models/Qwen3-0.6B \
  --single '{"enable_prefix_caching": true}' --tag cache_on --shared-only
```

`--shared-only` 只跑共享前缀场景，约 20 秒出数；不加则跑完整矩阵（TTFT + 吞吐 + shared）。

数据落 `bench_cache_ablation.csv`，图表：`python scripts/charts.py`。
