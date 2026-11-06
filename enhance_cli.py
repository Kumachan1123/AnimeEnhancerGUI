
import argparse  # コマンドライン引数を解析するために使用する
import contextlib  # 既存処理の標準出力を振り分けるために使用する
import json  # C#とのデータ受け渡しに使用する
import sys  # 標準入出力と終了コードの管理に使用する
from pathlib import Path  # ファイルパスを扱うために使用する

import torch  # GPUの使用可否を確認するために使用する
from basicsr.archs.rrdbnet_arch import RRDBNet  # AIモデルの構造を定義する
from realesrgan import RealESRGANer  # 高画質化処理を実行する

from enhance import process_image  # 既存の画像処理関数を再利用する


# 対応する画像の拡張子を定義する
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def emit(event_type: str, **values) -> None:
    """C#側へJSON形式のイベントを送信する。"""

    # イベントの種類と追加情報をまとめる
    event = {
        "type": event_type,
        **values,
    }

    # JSON形式で標準出力へ送信する
    print(json.dumps(event, ensure_ascii=False), flush=True)


def create_output_path(output_dir: Path, input_path: Path, reserved_names: set[str]) -> Path:
    """既存ファイルと名前が重複しない出力パスを作成する。"""

    # 元画像のファイル名を取得する
    stem = input_path.stem

    # 最初の出力ファイル名を作成する
    output_name = f"{stem}.png"

    # 同名ファイルが存在する間は連番を付ける
    index = 2

    while output_name.casefold() in reserved_names:
        output_name = f"{stem} ({index}).png"
        index += 1

    # 出力ファイル名を予約する
    reserved_names.add(output_name.casefold())

    # 出力先のパスを返す
    return output_dir / output_name


def main() -> int:
    """C#から指定された画像をまとめて高画質化する。"""

    # コマンドライン引数を定義する
    parser = argparse.ArgumentParser()

    # 画像一覧を記述したJSONファイルを受け取る
    parser.add_argument("--manifest", required=True)

    # 使用する計算装置を受け取る
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)

    # コマンドライン引数を解析する
    args = parser.parse_args()

    # スクリプトの保存場所を取得する
    base_dir = Path(__file__).resolve().parent

    # AIモデルの保存場所を設定する
    model_path = base_dir / "weights" / "RealESRGAN_x4plus_anime_6B.pth"

    # JSONファイルのパスを取得する
    manifest_path = Path(args.manifest).resolve()

    # JSONファイルの存在を確認する
    if not manifest_path.is_file():
        emit("error", message="画像一覧ファイルが見つかりません。")
        return 1

    # JSONファイルを読み込む
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception as error:
        emit("error", message=f"画像一覧の読み込みに失敗しました: {error}")
        return 1

    # 入力画像の一覧を取得する
    files = manifest.get("files")

    # 入力画像の一覧が正しいか確認する
    if not isinstance(files, list) or not files:
        emit("error", message="処理対象の画像がありません。")
        return 1

    # 出力先のパスを取得する
    output_dir = Path(manifest.get("output_dir", "")).resolve()

    # 出力先のフォルダを作成する
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        emit("error", message=f"出力先を作成できません: {error}")
        return 1

    # AIモデルが存在するか確認する
    if not model_path.is_file():
        emit("error", message=f"AIモデルが見つかりません: {model_path}")
        return 1

    # CUDAが利用できるか確認する
    if args.device == "cuda" and not torch.cuda.is_available():
        emit("error", message="CUDAが利用できません。GPUの設定を確認してください。")
        return 1

    # 使用する計算装置を設定する
    device = torch.device(args.device)

    # タイルサイズを設定する
    tile_size = 1024

    # モデル読み込み開始を通知する
    emit("status", message="AIモデルを読み込んでいます。")

    # AIモデルを初期化する
    try:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4,
        )

        # AIモデルと画像処理の設定を初期化する
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=(device.type == "cuda"),
            device=device,
        )

    except Exception as error:
        emit("error", message=f"AIモデルの読み込みに失敗しました: {error}")
        return 1

    # モデルの準備完了を通知する
    emit(
        "ready",
        total=len(files),
        device=device.type,
        tile=tile_size,
    )

    # 出力済みファイルとの名前の重複を防ぐ
    reserved_names = {
        path.name.casefold()
        for path in output_dir.iterdir()
        if path.is_file()
    }

    # 成功枚数を初期化する
    success_count = 0

    # 失敗枚数を初期化する
    failure_count = 0

    # 画像を順番に処理する
    for index, file_path in enumerate(files, start=1):

        # 入力画像のパスを取得する
        input_path = Path(file_path).resolve()

        # ファイル名を取得する
        file_name = input_path.name

        # 画像の出力先を決定する
        output_path = create_output_path(
            output_dir,
            input_path,
            reserved_names,
        )

        # 現在の処理対象を通知する
        emit(
            "current",
            index=index,
            total=len(files),
            file=file_name,
        )

        # 入力ファイルの存在と拡張子を確認する
        if not input_path.is_file() or input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            emit(
                "error",
                message=f"処理できない画像です: {file_name}",
            )
            failure_count += 1
            emit(
                "progress",
                current=index,
                total=len(files),
                file=file_name,
                success=False,
            )
            continue

        # 既存の画像処理関数を実行する
        try:
            # 既存関数の通常ログを標準エラー出力へ振り分ける
            with contextlib.redirect_stdout(sys.stderr):
                success = process_image(
                    input_path,
                    output_path,
                    upsampler,
                )

        except Exception as error:
            # 想定外のエラーを記録する
            print(f"画像処理中にエラーが発生しました: {error}", file=sys.stderr)
            success = False

        # 処理結果を集計する
        if success:
            success_count += 1
        else:
            failure_count += 1

    # 変換結果をC#側へ通知する。
    emit(
        "progress",
        current=index,
        total=len(files),
        file=file_name,
        input=str(input_path),
        output=str(output_path) if success else None,
        success=success,
    )

    # 全画像の処理結果を通知する
    emit(
        "done",
        success=success_count,
        failure=failure_count,
        output_dir=str(output_dir),
    )

    # すべて成功した場合は正常終了を返す
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    # 標準出力をUTF-8に設定する
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # プログラムを実行して終了コードを返す
    raise SystemExit(main())
