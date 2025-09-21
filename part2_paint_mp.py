import matplotlib.pyplot as plt

# 你的训练日志数据
epochs = list(range(1, 16))
accuracy = [0.3564, 0.5250, 0.6079, 0.6540, 0.6863, 0.7157, 0.7370, 0.7599, 0.7767, 0.7903, 0.8053, 0.8206, 0.8316, 0.8402, 0.8476]
val_accuracy = [0.1380, 0.2342, 0.5700, 0.5508, 0.6028, 0.6894, 0.6250, 0.6410, 0.6628, 0.6998, 0.6632, 0.7198, 0.7790, 0.7000, 0.7878]
loss = [1.8105, 1.3190, 1.0930, 0.9765, 0.8903, 0.8144, 0.7547, 0.6971, 0.6510, 0.6128, 0.5714, 0.5329, 0.5005, 0.4731, 0.4525]
val_loss = [4.2386, 2.9307, 1.1601, 1.2187, 1.1582, 0.8661, 1.2125, 1.1033, 0.9992, 0.9448, 1.0730, 0.8515, 0.6723, 1.0035, 0.6339]

plt.figure(figsize=(12, 5))

# Accuracy 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, label='Train Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Val Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

# Loss 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("charts/part2_mp_train_val_curve.png")
plt.show()