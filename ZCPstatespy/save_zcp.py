import os
import matplotlib.pyplot as plt

def save_zcp(savefile, figname, width=8, height=6):
    """

    input:
    - savefile: str, 保存的文件夹路径（如 'Figure'）
    - figname: str, 图像文件名（不含扩展名）
    - width: float, 图像宽度（英寸）
    - height: float, 图像高度（英寸）
    """
    #创建文件夹
    if not os.path.exists(savefile):
        os.makedirs(savefile)
        print(f"📁 创建目录: {savefile}--zcp")

    #尺寸
    plt.gcf().set_size_inches(width, height)

    #保存
    for ext in ['pdf', 'png', 'svg']:
        full_path = os.path.join(savefile, f"{figname}.{ext}")
        plt.savefig(full_path, format=ext, dpi=300, bbox_inches='tight')
        print(f"✅ 图像已保存: {full_path}")
