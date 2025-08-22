import numpy as np
import colorsys

def generate_35_category_colors(num):
    """
    生成35种类别的归一化三色数组(RGB格式)
    
    返回:
        colors (np.ndarray): 形状为(35, 3)的数组，每行是一个归一化的RGB颜色
    """
    # 使用HSV色彩空间生成更均匀分布的颜色
    colors = []
    
    # 生成不同色调的颜色 (0-1范围)
    for i in range(num):
        # 使用黄金角分布，确保颜色尽可能均匀分布
        hue = (i * 0.618033988749895) % 1.0  # 黄金角近似值
        
        # 固定饱和度和亮度，确保颜色清晰可辨
        saturation = 0.8
        value = 0.9
        
        # 转换为RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([r, g, b])
    
    # 转换为numpy数组
    colors = np.array(colors)
    
    # 确保所有值在[0,1]范围内
    colors = np.clip(colors, 0.0, 1.0)
    
    return colors
# print(generate_35_category_colors(35)[0])


def rotate_yaw(yaw):
    if yaw > np.pi:
        yaw -= 2 * np.pi
    elif yaw < -np.pi:
        yaw += 2 * np.pi
    return np.array(
        [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )

if __name__ == '__main__':
    colors = generate_35_category_colors(35)
    import cv2
    import matplotlib.pyplot as plt
    blank = np.ones((100*36, 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        cv2.rectangle(blank, (0, i * 100), (100, i * 100 + 100), (color * 255).astype(np.uint8).tolist(), -1)
        cv2.putText(blank, f'{i}', (10, i * 100 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    blank = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
    cv2.imwrite('colors.png', blank)