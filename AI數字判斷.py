import pygame
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 讀取模型
model = load_model("my_model.keras")  # 確保已經有訓練好的模型

# 初始化 Pygame
pygame.init()

# 設定畫布大小
WIDTH, HEIGHT = 280, 280
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a digit (0-9)")

# 顏色設定
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 創建畫布
screen.fill(BLACK)
drawing_surface = pygame.Surface((WIDTH, HEIGHT))
drawing_surface.fill(BLACK)

# 狀態變數
drawing = False
last_pos = None

# 影像預處理函數
def preprocess_image(image_surface):
    """將 Pygame 畫布轉換成 28x28 的數字圖像"""
    # 轉換為 NumPy 陣列
    image = pygame.surfarray.array3d(image_surface)  # (280, 280, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 轉換為灰階 (280, 280)

    # 縮放到 28x28 並使用抗鋸齒
    image = cv2.resize(image, (28, 28))

    # **沿 x = y 軸翻轉**
    image = image.T

    # 顯示處理後的圖片（可選）
   
    plt.imshow(image, cmap="gray")
    plt.title("Processed Image (28x28)")
    
   

    # **正規化並展平成 1D 陣列 (1, 784)**
    image = image / 255.0
    image = image.reshape(1, 784)  # 這行修正 shape
    return image

# 主迴圈
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.line(drawing_surface, WHITE, last_pos, event.pos, 15)
                last_pos = event.pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                image = preprocess_image(drawing_surface)  # 處理影像
                prediction = model.predict(image)  # 預測
                predicted_digit = np.argmax(prediction)  # 取得預測結果

                print(f"Predicted digit: {predicted_digit}")
                pygame.display.set_caption(f"Predicted: {predicted_digit}")
                drawing_surface.fill(BLACK)

    # 顯示畫布
    screen.fill(BLACK)
    screen.blit(drawing_surface, (0, 0))
    pygame.display.update()

    # 按鍵控制
    keys = pygame.key.get_pressed()

'''                  
    if keys[pygame.K_SPACE]:  # 按空白鍵開始辨識
        image = preprocess_image(drawing_surface)  # 處理影像
        prediction = model.predict(image)  # 預測
        predicted_digit = np.argmax(prediction)  # 取得預測結果

        print(f"Predicted digit: {predicted_digit}")
        pygame.display.set_caption(f"Predicted: {predicted_digit}")
        drawing_surface.fill(BLACK)
  '''
# 關閉 Pygame
pygame.quit()
