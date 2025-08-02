import torch
import numpy as np
import argparse
import os

def convert_pt_to_npy(input_pt, output_npy):
    # 加载.pt文件
    try:
        data = torch.load(input_pt, map_location='cpu')
    except Exception as e:
        raise ValueError(f"无法加载文件 {input_pt}: {e}")

    # 检查是否是张量
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"文件 {input_pt} 包含的不是一个PyTorch张量，而是 {type(data)} 类型。")

    # 转换为NumPy数组
    numpy_array = data.numpy()

    # 保存为.npy文件
    np.save(output_npy, numpy_array)
    print(f"转换成功！文件已保存到 {output_npy}")

def main():
    parser = argparse.ArgumentParser(description="将PyTorch的.pt文件转换为NumPy的.npy文件")
    parser.add_argument("input_pt", type=str, help="输入的.pt文件路径")
    parser.add_argument("output_npy", type=str, nargs='?', default=None, help="输出的.npy文件路径（可选）")
    args = parser.parse_args()

    # 确定输出路径
    if args.output_npy is None:
        base = os.path.splitext(args.input_pt)[0]
        output_path = base + ".npy"
    else:
        output_path = args.output_npy

    # 执行转换
    try:
        convert_pt_to_npy(args.input_pt, output_path)
    except Exception as e:
        print(f"错误发生: {e}")

if __name__ == "__main__":
    main()