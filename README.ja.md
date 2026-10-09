<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/examples/">
    <img src="assets/voxcpmtts_logo_horizontal.webp" alt="Hangry Labs VoxCPMTTS ロゴ" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <a href="README.pl.md">Polski</a> ·
  <strong>日本語</strong> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs VoxCPMTTS

Nano-vLLM 推論、ローカルのブラウザー UI、HTTP API を備えた、すぐに実行できる多言語テキスト読み上げ Docker パッケージです。

この Hangry Labs 版は簡単なローカル利用を目的としています。Python、モデル、音声ツールを手動で構成せずに、コンテナーを 1 つ起動して UI または API から音声を生成できます。

## 主な機能

- 音声生成、ストリーミング、音声デザイン、音声クローン、ローカル文字起こし、任意の非破壊音声仕上げに対応するブラウザー UI
- デザイン設定、画像、検索、SSML-H の読み込みと書き出しを備え、セッションをまたいで再利用できる永続音声プロファイルと対話スクリプト
- ナレーション、複数話者の対話、音声劇に対応する SSML と SSML-H
- アプリケーションやローカル連携向けのネイティブ HTTP API
- VoxCPM2 が対応する 30 言語での生成
- WAV、MP3、FLAC、OGG 出力
- CUDA グラフで高速化された Nano-vLLM 推論
- オフライン推論に必要なモデル資産を含む完全版 Docker イメージ

> [!IMPORTANT]
> プロジェクト独自およびアップストリーム由来のソースコードは Apache-2.0 です。ただし Docker イメージは、モデル資産、NVIDIA CUDA ライブラリ、FFmpeg、その他それぞれの条件が適用されるパッケージを含む集合配布物です。導入または再配布の前に[サードパーティ通知](THIRD_PARTY_NOTICES.md)を確認してください。

## クイックスタート

NVIDIA GPU で完全版イメージを起動します。

```bash
docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:latest
```

起動後に次を開きます。

- UI: [http://localhost:8808/ja](http://localhost:8808/ja)
- API ドキュメント: [http://localhost:8808/tts/docs](http://localhost:8808/tts/docs)
- [言語と音声のサンプル](https://hangry-labs.github.io/VoxCPMTTS/examples/)

`voxcpmtts_data` ボリュームには、モデルキャッシュ、音声プロファイル、対話スクリプト、参照音声、今後の設定が保存され、コンテナーを置き換えても維持されます。完全版イメージには固定されたモデル資産が含まれ、取得後は通常の推論をネットワークなしで実行できます。

公開済みの GitHub リリースと対応する Git タグは変更されません。バージョン付き Docker タグも不変ですが、`latest` と `latest_tiny` は意図的に最新スナップショットを指します。

## 詳細情報

[日本語の製品ページとインストールガイド](https://hangrylabs.app/ja/software/voxcpmtts)を参照してください。詳細な技術文書は[英語版 README](README.md)で管理しています。不具合や提案は [GitHub Issues](https://github.com/Hangry-Labs/VoxCPMTTS/issues) に報告できます。
