# 具身数据闭环：Platform / Curation 领域契约

当前实现采用同仓库、同 Flask 进程内的逻辑拆分。Data Platform 是数据供应与执行面，Data Curation 是数据决策面；两者共享 Dataset selector、Viewer、Job & Artifacts 和 Audit Log。

```mermaid
flowchart LR
    A[Source Delivery] --> B[Platform Ingestion]
    B --> C[Raw Version and Reconciliation]
    C --> D[Platform Preprocessing]
    D --> E[Standard Version]
    E --> F[Profile and Requirement]
    F --> G[Recipe and Curation Workspace]
    G --> H[Published Manifest]
    H --> I[Platform Materializer]
    I --> J[Curated Version]
    J --> K[Catalog and Export]
```

## Capability matrix

| Workspace | Capability | 当前入口或执行模块 |
|---|---|---|
| Data Platform | Catalog、Ingestion、checksum/schema 校验、版本登记 | Dataset selector、Versions & Lineage、`lifecycle.py` |
| Data Platform | Source Delivery、Episode 去向与处理数量对账 | Versions & Lineage、`SourceBatchVersion`、`ProcessingReconciliationReport` |
| Data Platform | Cache、标准化、格式/字段/信号转换 | Cache、Standardize、Transform、`precompute/preprocess/` |
| Data Platform | Split/Merge/Subtract 与 Manifest 物化 | Materialize、`dataset_merge.py`、`lifecycle.materialize_manifest` |
| Data Curation | Viewer、Analysis、Embedding、Compare | Explore 共享入口、Embedding、Compare |
| Data Curation | 技术问题证据与语义质量复核 | Quality Review、Viewer flags |
| Data Curation | Stage/Subtask、bbox、prompt、tag | Stage & Subtask、Object Labeling、Auto-tagging |
| Data Curation | include/exclude、repair recipe、cohort 构造 | Cohorts & Dataset Build |
| Data Curation | Dataset Profile、Requirement、版本化 Recipe | Cohorts & Dataset Build |
| Legacy Admin | 原地 prompt/Parquet/video 修复与 episode 删除 | 默认隐藏；在页面通过管理员密码进入 |

技术损坏进入 Infra 检查或 quarantine；轨迹是否有用、语义是否正确、是否进入发布数据集属于 Curation 决策。Flag 是证据，不等同于删除状态。

### 数据阶段与保护级别

数据阶段和可删除性不绑定为一个字段：

| 维度 | 值 | 含义 |
|---|---|---|
| `stage` | `raw / standard / curated` | 数据在闭环中的处理阶段 |
| `retention_class` | `protected_source / managed / disposable` | 物理数据保护和清理策略 |

受保护状态来自两类控制面策略：配置一个或多个 source root，或在 Catalog 中手工标记单个数据集。路径策略优先且覆盖全部子目录；手工取消保护不会绕过路径策略。Standardize、Split、Merge、Subtract 等 sibling-output 操作可以读取受保护源数据，但输出应位于 source root 之外。`unregister` 只删除 Catalog 记录，不删除物理文件。

## 核心对象

| 对象 | 稳定身份与主要字段 | 可变性 |
|---|---|---|
| `DatasetVersion` | `dataset_id`、`version_id`、fingerprint、schema、parent versions、profile/executor digest、identity artifact | 发布后不可变 |
| `SourceBatchVersion` | 来源类型/URI、robot profile、signal schema、format、预期 Episode 和 retention | 发布后不可变；变化生成下一版本 |
| `ProcessingReconciliationReport` | 输入/输出版本、EpisodeDisposition、数量和原因对账 | 不可变、由 Ingest/派生版本自动生成 |
| `DatasetReplica` | DatasetVersion、物理 root、fingerprint、dataset key、可用状态 | 可新增位置，不改变逻辑版本 |
| `EpisodeRef` | `dataset_version_id + episode_uid` | 不可变；`episode_index` 仅为版本内位置 |
| `DatasetIdentityArtifact` | DatasetVersion、完整 fingerprint、episode UID/content fingerprint、source ref | 不可变、可导入导出 |
| `ProfileVersion` | preprocessing/materialization 类型、有序步骤、内容参数与 executor digest | 发布后不可变 |
| `DatasetProfileVersion` | DatasetVersion 的数量、时长、task 与扩展分布快照 | 不可变、可重复生成同一 digest |
| `DatasetRequirementVersion` | 目标数量、覆盖维度、质量与组成约束 | 不可变；同名变化生成下一版本 |
| `DataRecipeVersion` | base、固定 cohort、include/exclude、组成、精确去重和 seed | 不可变、可确定性编译为 Workspace |
| `CurationWorkspace` | base version、revision、decision/patch/repair/cohort/evidence | 发布前可变，使用乐观锁 |
| `CurationManifestVersion` | Workspace 的固定快照、base、选择、patch/repair、规则/模型与 digest | 不可变 |
| `MaterializationRun` | 幂等键、Manifest/Profile、staging/output、状态与校验报告 | 仅按状态机推进 |

本地 lifecycle control plane 默认位于 console registry 相邻的 `lifecycle/`：

```text
lifecycle/
├── lifecycle.db
└── artifacts/
    ├── identities/<artifact_digest>.json
    └── content/<dataset_fingerprint>.json
```

SQLite 启用 WAL、foreign keys 和 busy timeout，支持单机多进程。启动时会幂等导入旧的目录 JSON ledger，但不删除旧文件。身份与 Content Manifest 位于数据集目录之外，不向 Raw/Standard Dataset 回填控制文件。

首次接入生成随机 `episode_uid` 并固化在 Identity Artifact。复制数据时连同 artifact 导入，会登记同一逻辑版本的新 Replica；Split/Merge/Profile executor 使用显式 `EpisodeLineageMap` 继承 UID。没有 lineage 的历史输出标记为 `legacy_inferred`。

## 生命周期接口

| 逻辑接口 | HTTP / 代码入口 | 结果 |
|---|---|---|
| `ingest(source, identity?)` | `POST /api/lifecycle/ingest[/start]` | 完整 fingerprint、Identity Artifact、Raw Version/Replica；失败进入 Infra quarantine |
| `register_source_batch(delivery)` | `POST /api/lifecycle/source-batches` | 固定来源、数据画像、预期数量与保护级别 |
| `reconcile(input, output)` | `GET /api/lifecycle/reconciliations` | Episode 级 received/accepted/excluded/repaired/generated 去向与数量对账 |
| `preprocess(base, profile)` | `POST /api/lifecycle/preprocess/start` | 执行已登记 Profile，生成带显式 lineage 的 Standard Version |
| `profile(version)` | `POST /api/curation/dataset-profiles` | 固定数量、时长和分布快照 |
| `define_requirement(spec)` | `POST /api/curation/requirements` | 版本化目标数量、覆盖与质量要求 |
| `resolve_cohort(query)` | `POST /api/curation/cohorts/resolve` | 将查询固定为 base fingerprint 与 EpisodeRef 集合 |
| `publish_recipe(strategy)` | `POST /api/curation/recipes` | 固定选择、配比、精确去重和随机种子 |
| `compile_recipe(recipe)` | `POST /api/curation/recipes/<id>/compile` | 生成可审核 Curation Workspace |
| `create/update workspace` | `/api/curation/workspaces` | revision/optimistic-lock 控制的 Curation 草稿 |
| `publish_manifest(workspace)` | `POST /api/curation/workspaces/<id>/publish` | 校验并冻结 Published `CurationManifestVersion` |
| `materialize(manifest, profile)` | `POST /api/lifecycle/materialize/start` | Curated DatasetVersion、Replica、MaterializationRun |

Manifest 状态只允许顺序推进：

```text
Draft → In Review → Approved → Published → Materialized
```

`Materialized` 是由 materialization record 派生的展示状态，不会改写 Published Manifest。

旧 `/api/curation/manifests` 只作为兼容入口；新页面先创建 Workspace，再发布 Manifest。

## Profile 与 Materializer 不变量

- 数据格式版本（LeRobot v2.1/v3.0）、机器人画像、信号 schema、生命周期 stage、来源类型和
  retention class 是独立维度。DVT2 Standardize 为 16D 后仍保留 `h10w_dvt2`；v2.1/v3.0
  物理版本通过 `logical_snapshot_id`、`format_variant_of` 与相同 Episode UID 集合关联。
- 所有 sibling preprocessing 输出进入 `standard`（Curated 输入的派生输出仍为 `curated`），不继承
  Raw 的 `protected_source` 保护级别；来源血缘仍通过 `source_batch_ids` 保留。
- Preprocessing Profile 目前支持 `copy`、`split`、`standardize`、`convert_action` 和 `drop_field`；Materialization Profile 强制 full validation。workers 等 `ExecutionOptions` 不参与版本身份。
- 只接受 Published Manifest 和已登记的 Materialization Profile。
- 发布和物化时重新校验 base fingerprint；源数据变化后必须登记新版本。
- include/exclude、cohort snapshot、Recipe、patch 和 repair 均使用稳定 `EpisodeRef`。
- include/exclude 冲突、重复 patch、重复数值 repair、缺失 episode、空选择均拒绝。
- 当前 repair recipe 支持 `trim` 和 `value_edit`；执行发生在临时 sibling 内。
- `MaterializationRun` 使用 `Planned → Running → Validating → Committed/Failed`；幂等键由 base version、Manifest digest 和 Profile digest 生成。
- 临时输出通过结构、row/frame 数、时间戳、metadata 和所有 data/meta/video 文件 SHA256 校验后才原子 rename，并在一个 SQLite 事务中登记 Version、Replica 和 Committed Run。
- 同一 Manifest/Profile 可以在多个输出路径重现；它们共享逻辑 DatasetVersion，并分别登记 Replica。
- 服务重启后可恢复已经 rename 但尚未提交数据库的输出；staging 由持久化 lease 防止多进程重复执行。
- Split/Merge/Profile executor 显式输出 lineage；fingerprint/保序推断只保留给历史兼容数据。
- Recipe 发布时固定 cohort 查询结果和 seed；编译前校验 Requirement 的目标数量、覆盖维度与组成约束。

## Infra Quarantine

技术质量类别固定为 `checksum`、`missing_file`、`schema`、`parquet`、`video`、`timestamp`、`index`、`metadata`、`unsupported_format`。状态按 `Open → Retrying → Resolved` 或 `Open/Retrying → Waived` 推进，并记录 detector、evidence、受影响文件/episode、retry job 和解决后的 DatasetVersion。语义 flag 不进入 Quarantine。

## 兼容迁移

默认启动隐藏并在服务器侧拒绝旧原地写回 API。需要短期对照旧流程时，在页面使用当前数据根目录的管理员密码进入 Admin Mode；首次进入可直接设置密码。Admin Mode 没有倒计时，主动退出、修改密码或浏览器会话结束后重新锁定。物理删除要求影响预览、删除原因和二次确认，且 Source Protection 始终由服务器强制执行。

迁移对照应比较 episode UID/集合、schema、统计和 fingerprint。只有新旧输出一致后，才下线对应的旧写回路径。
