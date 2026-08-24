# V-JEPA 2.1 SCAND比較設計

## 目的

既存のDINOv2版RegNavを保持したまま、SCANDの4フレーム映像を入力するV-JEPA 2.1 ViT-B版を追加し、同一split・同一trajectory decoderで比較できるようにする。

## 評価指標の扱い

既存評価の速度上限0.8 m/sは配備先の制約であり、SCAND収録速度の上限ではない。SCAND odometryでは0.8 m/s超過率がJackal 93.7%、Spot 90.9%だった。このためmodelとbaselineに加えてtarget trajectoryの制約違反率も出力し、模倣精度と配備可能性を区別する。既存のADE/FDE acceptance条件は変更しない。

## 入力

- DINOv2版は従来どおり現在画像1枚を使う。
- V-JEPA 2.1版は現在を含む連続4フレームを `[C,T,H,W]` で返す。
- trajectoryの先頭3フレームは履歴不足のためsample indexから除外する。
- V-JEPA 2.1は384x384、patch 16、tubelet 2、backbone dimension 768を使う。

## モデル

- `backbone`名が`vjepa2_1_`で始まる場合だけV-JEPA adapterを選ぶ。
- adapterは注入されたfake backboneでも動作し、通常は公式PyTorch HubからViT-B encoderをロードする。
- V-JEPA出力のpatch tokensを既存`SceneRegisterPool`へ渡す。
- 既存のattention `qkv`/`proj` LoRA注入、trajectory decoder、scorerは変更しない。
- 公式repositoryのdownload URL不具合を避けるため、V-JEPA 2.1公開時のcommit `45d025f`を既定refとして固定する。

## 今回含めないもの

- V-JEPA内部へのtrajectory token挿入
- Flow Matching
- predictorの使用
- Nav2/Gazebo統合
- SCAND全量の再学習

これらはV-JEPA adapterの実モデルsmoke testと小規模overfitが成立した後に判断する。

## RTX 3060での実測結果

2026-08-24にRTX 3060 12GBで公式V-JEPA 2.1 ViT-B checkpointをロードし、次を確認した。

- encoder FP32推論：出力`[1,1152,768]`、初回forward約252 ms、計測区間の追加VRAM約0.39 GiB
- RegNav全体のBF16 autocast推論：出力軌道`[1,8,3]`、warm-up後約41.3 ms/sample（24.2 FPS）、追加VRAM約0.40 GiB
- SCAND stride-10実データの学習1 step：batch 16、forward/backward/optimizer更新約1.14秒、ピーク約6.30 GiB
- unit test：113件すべて成功

公式Hub実装のcheckpoint URLは公開commitでもlocalhostを指すため、encoder構造だけを固定commitから読み、公開checkpointをHTTPSで直接取得してstrict loadする。checkpointは約1.55GBである。

モデル全体を`bfloat16`へ変換するとRoPE内でQ/KとVのdtypeが不一致になる。一方、重みをFP32のまま訓練コード既存の`torch.autocast(..., dtype=torch.bfloat16)`を使う経路はLoRA込みで正常に動作する。そのため専用dtype処理は追加しない。

## 次の比較

既存DINOv2版とV-JEPA 2.1版を同じSCAND stride-10 splitで学習し、ADE/FDE、target比の制約違反、推論時間を比較する。V-JEPA版が上回らない段階では、encoder途中へのtrajectory token挿入やFlow Matchingを追加しない。
