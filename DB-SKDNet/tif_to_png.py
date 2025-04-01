###################################################################### TIF图像转PNG



import numpy as np
import os
from PIL import Image
from osgeo import gdal
 
# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')
 
# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
gdal.UseExceptions()
 
 
def readTif(imgPath, bandsOrder=[3, 2, 1]):
    """
    读取GEO tif影像的前三个波段值，并按照R.G.B顺序存储到形状为【原长*原宽*3】的数组中
    :param imgPath: 图像存储全路径
    :param bandsOrder: RGB对应的波段顺序，如高分二号多光谱包含蓝，绿，红，近红外四个波段，RGB对应的波段为3，2，1
    :return: R.G.B三维数组
    """
    dataset = gdal.Open(imgPath, gdal.GA_ReadOnly)  # 返回一个gdal.Dataset类型的对象
    cols = dataset.RasterXSize                      # tif图像的宽度
    rows = dataset.RasterYSize                      # tif图像的高度
    data = np.empty([rows, cols, 3], dtype=float)   # 定义结果数组，将RGB三波段的矩阵存储
    for i in range(3):  
        band = dataset.GetRasterBand(bandsOrder[i]) # 读取波段数值
        oneband_data = band.ReadAsArray()           # 读取波段数值读为numpy数组
        #print(oneband_data)
        data[:, :, i] = oneband_data                # 将读取的结果存放在三维数组的一页三
    return data
 
def stretchImg(imgPath, resultPath, lower_percent=0.5, higher_percent=99.5):
    """
    #将光谱DN值映射至0-255，并保存
    :param imgPath: 需要转换的tif影像路径（***.tif）
    :param resultPath: 转换后的文件存储路径(***.jpg)
    :param lower_percent: 低值拉伸比率
    :param higher_percent: 高值拉伸比率
    :return: 无返回参数，直接输出图片
    """
    print(imgPath)
    RGB_Array=readTif(imgPath)
    print(RGB_Array.shape)
    band_Num = RGB_Array.shape[2]           # 数组第三维度的大小，在这里是图像的通道数
    JPG_Array = np.zeros_like(RGB_Array, dtype=np.uint8)
    for i in range(band_Num):
        minValue = 0
        maxValue = 255
        #获取数组RGB_Array某个百分比分位上的值
        low_value = np.percentile(RGB_Array[:, :,i], lower_percent)
        high_value = np.percentile(RGB_Array[:, :,i], higher_percent)
        temp_value = minValue + (RGB_Array[:, :,i] - low_value) * (maxValue - minValue) / (high_value - low_value)
        temp_value[temp_value < minValue] = minValue
        temp_value[temp_value > maxValue] = maxValue
        JPG_Array[:, :, i] = temp_value
    outputImg = Image.fromarray(np.uint8(JPG_Array))
    outputImg.save(resultPath)
 
def Batch_Convert_tif_to_jpg(imgdir,savedir):
    #获取文件夹下所有tif文件名称，并存入列表
    file_name_list = os.listdir(imgdir)
    for name in file_name_list:
        # 获取图片文件全路径
        img_path = os.path.join(imgdir, name)
        #获取文件名，不包含扩展名
        filename = os.path.splitext(name)[0]
        print(filename)
        savefilename = filename + "" +".jpg"
        #文件存储全路径
        savepath = os.path.join(savedir, savefilename)
        # img_path为tif文件的完全路径
        # savepath为tif文件对应的jpg文件的完全路径
        print(savepath)
        stretchImg(img_path, savepath)
        print("图片:【", filename, "】完成转换")
    print("完成所有图片转换!")
    
 
if __name__ == '__main__':
    imgdir = r"D:\C2F-SemiCD-and-C2FNet-main\C2F-SemiCD-and-C2FNet-main\GoogleGZ\B"        # tif文件所在的【文件夹】
    savedir = r"D:\C2F-SemiCD-and-C2FNet-main\C2F-SemiCD-and-C2FNet-main\GoogleGZ1\B"   # 转为jpg后存储的【文件夹】
    Batch_Convert_tif_to_jpg(imgdir, savedir)
    






# ####################################################################### TIF标签转PNG

# import os
# from PIL import Image
# import numpy as np

# def rgb_to_single_channel(rgb_image_path, output_path, rgb_dict):
#     """
#     将RGB图像转换为单通道图像，使用指定的RGB字典来映射类别标签。

#     :param rgb_image_path: RGB图像的路径
#     :param output_path: 转换后图像的保存路径
#     :param rgb_dict: RGB字典，键是RGB元组，值是对应的类别标签
#     """
#  # 读取图像
#     image = Image.open(rgb_image_path)
#     # 将图像转换为numpy数组
#     image_array = np.array(image)

#     # 创建一个空的单通道图像数组
#     single_channel_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

#     # 根据图像模式进行处理
#     if image.mode == 'L':
#         # 单通道图像
#         for value, label in rgb_dict.items():
#             if isinstance(value, int):
#                 # 如果字典的键是单通道值
#                 single_channel_array[image_array == value] = label
#             elif isinstance(value, tuple) and len(value) == 1:
#                 # 如果字典的键是单元素元组
#                 single_channel_array[image_array == value[0]] = label
#     elif image.mode == 'RGB':
#         # RGB图像
#         for rgb, label in rgb_dict.items():
#             # 将图像中的每个像素的RGB值与rgb_dict中的RGB元组进行比较
#             single_channel_array[np.all(image_array == rgb, axis=-1)] = label
#     else:
#         raise ValueError(f"不支持的图像模式: {image.mode}")

#     # 将单通道数组转换为图像并保存
#     single_channel_image = Image.fromarray(single_channel_array, mode='L')
#     single_channel_image.save(output_path)

# def convert_tif_to_png(image_folder, output_folder, rgb_dict):
#     """
#     将指定文件夹中的所有TIFF图像转换为PNG格式，并保存到另一个文件夹中。
#     如果图像是单通道图像，则直接转换为PNG。如果图像是RGB图像，则使用rgb_to_single_channel函数进行转换。

#     :param image_folder: 包含TIFF图像的文件夹路径。
#     :param output_folder: 保存转换后的PNG图像的文件夹路径。
#     :param rgb_dict: RGB字典，用于rgb_to_single_channel函数。
#     """
#     # 检查输出文件夹是否存在，如果不存在则创建它
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 遍历图像文件夹中的所有文件
#     for filename in os.listdir(image_folder):
#         filepath = os.path.join(image_folder, filename)

#         # 检查文件是否为TIFF格式
#         if filename.endswith('.tif') or filename.endswith('.tiff'):
#             try:
#                 # 打开TIFF图像
#                 image = Image.open(filepath)
#                 original_mode = image.mode

#             # 处理浮点型图像模式'F'
#                 if original_mode == 'F':
#                     data = np.array(image)
#                     print(f"[调试] {filename} 浮点数据范围:", data.min(), data.max())

#                 """ 如果数据是类似[0.0, 1.0, 2.0...255.0]的整数浮点表示，使用方案1
#                     如果数据是标准化后的[0.0~1.0]范围，使用方案2
#                     如果数据范围异常（如负数或超大值），需要先做特殊处理：
#                     异常数据处理示例（裁剪到0~1范围）
#                     data = np.clip(data, 0, 1)
#                     data_normalized = (data * 255).astype(np.uint8)
#                 """
                
#                 # 方案1. 直接转换  转换为uint8类型，假设浮点数值为整数标签
#                     # data_uint8 = data.astype(np.uint8)   
#                     # image = Image.fromarray(data_uint8, mode='L')
#                 # 方案2：先归一化再转换（适用于0.0~1.0范围的数据）
#                     data_normalized = (data * 255).astype(np.uint8)
#                     image = Image.fromarray(data_normalized, mode='L')  # 使用方案2时变量名要对应
  
#                 ################################################
                
                    

#                 # 处理转换后的图像模式
#                 if image.mode == 'L':
#                     output_filename = os.path.splitext(filename)[0] + '.png'
#                     output_filepath = os.path.join(output_folder, output_filename)
#                     image.save(output_filepath, 'PNG')
#                     print(f"Converted {filename} to PNG format.")
#                 elif image.mode == 'RGB':
#                     output_filename = os.path.splitext(filename)[0] + '_single_channel.png'
#                     output_filepath = os.path.join(output_folder, output_filename)
#                     rgb_to_single_channel(filepath, output_filepath, rgb_dict)
#                     print(f"Converted {filename} to single channel PNG.")
#                 else:
#                     raise ValueError(f"不支持的图像模式: {image.mode}")

#             except Exception as e:
#                 print(f"Error converting {filename}: {str(e)}")

# # 指定图像文件夹和输出文件夹
# image_folder = r"D:\C2F-SemiCD-and-C2FNet-main\C2F-SemiCD-and-C2FNet-main\Dsifn-CD\test\label"
# output_folder = r"D:\C2F-SemiCD-and-C2FNet-main\C2F-SemiCD-and-C2FNet-main\Dsifn-CD\test\label1"

# # 定义RGB字典
# rgb_dict = {
#    255: 1,  # 不透水路面 Impervious surfaces (单通道值: 255)
#     0: 2,    # 建筑物 Building (单通道值: 0)
#     128: 3,  # 低植被 Low vegetation (单通道值: 128)
#     64: 4,   # 树木 Tree (单通道值: 64)
#     192: 5,  # 汽车 Car (单通道值: 192)
#     # 255: 255 # 背景 Clutter/background (单通道值: 255)
# }

# # 调用函数进行转换
# convert_tif_to_png(image_folder, output_folder, rgb_dict)
