"""
图像处理算法模块
用于CUMCM图像处理相关题目
"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class ImageProcessing:
    """基础图像处理"""
    
    @staticmethod
    def read_image(filepath: str) -> np.ndarray:
        """读取图像（无外部依赖版本）"""
        # 使用numpy创建测试图像或读取简单格式
        try:
            from PIL import Image
            img = Image.open(filepath)
            return np.array(img)
        except ImportError:
            # 生成随机测试图像
            return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        if len(image.shape) == 2:
            return image
        # RGB转灰度
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        img = image.astype(np.float64)
        return (img - img.min()) / (img.max() - img.min() + 1e-10)
    
    @staticmethod
    def resize(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小（双线性插值）"""
        h, w = image.shape[:2]
        new_h, new_w = new_size
        
        result = np.zeros((new_h, new_w, image.shape[2]) if len(image.shape) > 2 else (new_h, new_w))
        
        for i in range(new_h):
            for j in range(new_w):
                src_i = i * h / new_h
                src_j = j * w / new_w
                
                i0, i1 = int(src_i), min(int(src_i) + 1, h - 1)
                j0, j1 = int(src_j), min(int(src_j) + 1, w - 1)
                
                fi, fj = src_i - i0, src_j - j0
                
                if len(image.shape) == 2:
                    result[i, j] = (1-fi)*(1-fj)*image[i0,j0] + fi*(1-fj)*image[i1,j0] + \
                                   (1-fi)*fj*image[i0,j1] + fi*fj*image[i1,j1]
                else:
                    for c in range(image.shape[2]):
                        result[i, j, c] = (1-fi)*(1-fj)*image[i0,j0,c] + fi*(1-fj)*image[i1,j0,c] + \
                                          (1-fi)*fj*image[i0,j1,c] + fi*fj*image[i1,j1,c]
        
        return result.astype(image.dtype)


class EdgeDetection:
    """边缘检测"""
    
    @staticmethod
    def sobel(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sobel边缘检测"""
        img = image.astype(np.float64)
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Sobel算子
        gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        
        h, w = img.shape
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                region = img[i-1:i+2, j-1:j+2]
                gx[i, j] = np.sum(region * gx_kernel)
                gy[i, j] = np.sum(region * gy_kernel)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        return gx, gy, magnitude
    
    @staticmethod
    def canny(image: np.ndarray, threshold1: float = 50, threshold2: float = 150) -> np.ndarray:
        """Canny边缘检测（简化版）"""
        _, _, magnitude = EdgeDetection.sobel(image)
        
        # 双阈值处理
        edges = np.zeros_like(magnitude)
        edges[magnitude >= threshold2] = 255
        edges[(magnitude >= threshold1) & (magnitude < threshold2)] = 100
        edges[magnitude < threshold1] = 0
        
        return edges.astype(np.uint8)


class ImageSegmentation:
    """图像分割"""
    
    @staticmethod
    def threshold_segmentation(image: np.ndarray, method: str = "otsu") -> np.ndarray:
        """阈值分割"""
        img = image.astype(np.float64)
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        
        if method == "otsu":
            # Otsu阈值法
            hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
            total = sum(hist)
            
            best_threshold = 0
            best_var = 0
            
            for t in range(256):
                w0 = sum(hist[:t+1])
                w1 = sum(hist[t+1:])
                
                if w0 == 0 or w1 == 0:
                    continue
                
                u0 = sum(i * hist[i] for i in range(t + 1)) / w0
                u1 = sum(i * hist[i] for i in range(t + 1, 256)) / w1
                
                var = w0 * w1 * (u0 - u1) ** 2
                if var > best_var:
                    best_var = var
                    best_threshold = t
            
            return (img > best_threshold).astype(np.uint8) * 255
        
        elif method == "adaptive":
            # 自适应阈值
            block_size = 11
            c = 2
            h, w = img.shape
            result = np.zeros_like(img)
            
            for i in range(h):
                for j in range(w):
                    i_start = max(0, i - block_size // 2)
                    i_end = min(h, i + block_size // 2)
                    j_start = max(0, j - block_size // 2)
                    j_end = min(w, j + block_size // 2)
                    
                    local_mean = np.mean(img[i_start:i_end, j_start:j_end])
                    if img[i, j] > local_mean - c:
                        result[i, j] = 255
            
            return result.astype(np.uint8)
        
        return (img > np.mean(img)).astype(np.uint8) * 255
    
    @staticmethod
    def kmeans_segmentation(image: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """KMeans图像分割"""
        img = image.astype(np.float64)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        pixels = img.reshape(-1, 3)
        n_pixels = len(pixels)
        
        # 初始化聚类中心
        indices = np.random.choice(n_pixels, k, replace=False)
        centers = pixels[indices].copy()
        
        labels = np.zeros(n_pixels, dtype=int)
        
        for _ in range(50):  # 最多50次迭代
            # 分配
            dists = np.sqrt(((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(dists, axis=1)
            
            # 更新中心
            new_centers = np.zeros_like(centers)
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    new_centers[i] = pixels[mask].mean(axis=0)
            
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        # 重建图像
        segmented = centers[labels].reshape(img.shape)
        
        return segmented.astype(np.uint8), labels.reshape(img.shape[:2])


class FeatureExtraction:
    """特征提取"""
    
    @staticmethod
    def harris_corner(image: np.ndarray, k: float = 0.05, threshold: float = 0.01) -> np.ndarray:
        """Harris角点检测"""
        img = image.astype(np.float64)
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        
        # 计算梯度
        gx, gy, _ = EdgeDetection.sobel(img)
        
        # 计算Ixx, Iyy, Ixy
        Ixx = gx ** 2
        Iyy = gy ** 2
        Ixy = gx * gy
        
        # 高斯加权
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        
        h, w = img.shape
        R = np.zeros((h, w))
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                region_xx = Ixx[i-1:i+2, j-1:j+2]
                region_yy = Iyy[i-1:i+2, j-1:j+2]
                region_xy = Ixy[i-1:i+2, j-1:j+2]
                
                A = np.sum(region_xx * kernel)
                B = np.sum(region_yy * kernel)
                C = np.sum(region_xy * kernel)
                
                R[i, j] = A * B - C ** 2 - k * (A + B) ** 2
        
        # 非极大值抑制
        corners = np.zeros_like(R)
        for i in range(1, h-1):
            for j in range(1, w-1):
                if R[i, j] > threshold:
                    if (R[i, j] >= R[i-1, j] and R[i, j] >= R[i+1, j] and
                        R[i, j] >= R[i, j-1] and R[i, j] >= R[i, j+1]):
                        corners[i, j] = R[i, j]
        
        return corners.astype(np.uint8)
