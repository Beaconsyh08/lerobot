1. 提前生成视频文件
<video_path> 是视频文件的保存路径，需要起名以 `local_` 开头，如 `local_test_haha`
路径建议使用绝对路径
```shell
python lerobot/scripts/prepare_videos_from_images.py --root <data_path> --output-dir <video_path> --prepare-csv 1 --downsample 5
```

2. 运行网页显示
```shell
python lerobot/scripts/visualize_dataset_html.py --root <data_path> --output-dir <video_path> --precomputed-only 1 --downsample 5
```
