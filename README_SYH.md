一条命令完成预计算 + 网页展示（默认 `prepare-csv=1`、`precomputed-only=1`）：
```shell
python lerobot/scripts/prepare_videos_from_images.py --root <data_path>
```

```shell
python lerobot/scripts/prepare_videos_from_images.py --root /DATA/disk0/huggingface/lerobot/no_action_0_1
```

默认 `--output-dir` 会自动生成为数据目录同级路径：
- 规则：`<root_parent>/local_vis_<root_name>`
- 示例：
  - `--root /DATA/disk0/huggingface/lerobot/no_action_0_1`
  - 输出目录：`/DATA/disk0/huggingface/lerobot/local_vis_no_action_0_1`

如果第一步预计算文件（视频+CSV）已经存在，会自动跳过预计算，直接启动网页展示。
