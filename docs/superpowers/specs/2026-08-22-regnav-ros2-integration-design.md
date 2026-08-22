# RegNav ROS 2 統合設計

## 目的

学習済みRegNavを、`tuktuk_modules` のROS 2 Humbleコンテナから連続推論できるようにする。まずは予測軌跡の可視化・検証を目的とし、既存の速度指令やNav2の制御器を置き換えない。

## 前提と制約

- ROS 2ワークスペースは `/home/ubuntu/tuktuk_modules/ros2`。
- 実行コンテナは `tuktuk_modules` のdevcontainerで、GPUはRTX 3060。
- コンテナにはROS 2はあるが、RegNav、PyTorch、timmは未導入。
- 現在のシミュレータにはカメラトピックがないため、合成画像または実カメラでsmoke testを行う。
- `tuktuk_modules` には既存の未コミット変更がある。今回の変更は既存ファイルを上書きせず、RegNavパッケージと必要なdevcontainer設定に限定する。

## 採用構成

`regnav_ros` というament-pythonパッケージを `tuktuk_modules/ros2/src` に追加する。ノードはRegNav本体を再実装せず、RegNavリポジトリの `RegNavPredictor` を直接再利用する。

コンテナにはRegNavリポジトリを `/home/ubuntu/RegNav` としてbind mountし、devcontainerイメージにuvを導入する。uvでsystem-site-packagesを有効にしたPython仮想環境を作成し、その環境へRegNavをeditable installする。ROSのPythonサイトパッケージも同じ環境から利用する。colconはこの仮想環境を有効にした状態で実行する。

推論モデルはノード起動時に一度だけcheckpointからロードし、画像callbackごとに再ロードしない。checkpointとdeviceはROSパラメータで指定する。checkpointロード時に学習済みbackboneがある場合は、timmのpretrained重みをネットワークから再取得しない。

## ROSインターフェース

既定値は実機・既存Nav2構成に合わせるが、すべてROSパラメータで変更可能にする。

| 用途 | 型 | 既定値 |
| --- | --- | --- |
| 入力画像 | `sensor_msgs/msg/CompressedImage` | `/camera/rgb/image_raw/compressed` |
| 自己状態 | `nav_msgs/msg/Odometry` | `/odom` |
| 目標姿勢 | `geometry_msgs/msg/PoseStamped` | `/goal_pose` |
| 予測軌跡 | `nav_msgs/msg/Path` | `/regnav/predicted_path` |

`Odometry.twist.twist` からRegNavのego特徴 `[速度, 角速度, 速度加速度, 角加速度]` を作る。加速度は直前の有効なodomとの差分から計算し、最初のサンプルでは0とする。

`PoseStamped` の目標はtf2で現在のロボット基準座標へ変換し、RegNavのroute-goal `[x, y, yaw, turn_one_hot(3)]` に変換する。目標が未受信、tf変換失敗、odom未受信の間は推論せず警告を出す。画像・odom・goalのheader timestampを比較し、`max_state_age_sec`（既定0.5秒）を超えて古い状態は使わない。

予測軌跡はロボット基準座標の `nav_msgs/Path` としてpublishする。各poseの姿勢は予測yawから生成し、frame_idはパラメータで指定する。衝突回避や速度指令publishは今回の範囲外とする。

## 失敗時の動作

- 画像のJPEGデコード失敗、異常な数値、checkpoint不在はcallback単位でログし、プロセスを落とさない。
- checkpointロード失敗やモデル構成不一致は起動時に明示的な例外として停止する。
- 古い画像に対して古いodom/goalを黙って使わない。入力状態の未準備時はpublishしない。

## テストと検証

1. RegNav側: pretrained重みを再取得せずcheckpointをロードできること、既存CLIと既存105テストが変わらないこと。
2. ROSパッケージ側: ego/route-goal変換、yaw変換、異常入力の無視をROSなしの単体テストで確認する。
3. コンテナ内: `colcon build --symlink-install --packages-select regnav_ros`。
4. コンテナ内: 合成`CompressedImage`、odom、goalをpublishし、`/regnav/predicted_path` が1件以上出るROS2 smoke testを行う。
5. 実カメラ接続時: `ros2 topic list -t` と `ros2 topic hz` で入力型・周期を確認し、CUDA deviceと推論周期をログで確認する。

## 非採用案

ホスト側のRegNavをHTTP等の別推論サービスにしてROSノードから呼ぶ案は、プロセス・プロトコル・障害箇所が増え、推論のGPU常駐も複雑になるため採用しない。

