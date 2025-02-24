import sys
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QPushButton
from torchvision.transforms import v2
import torch
from model import model
from PIL import Image, ImageQt
import PIL
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = torch.load('convRegression-model-params.pt')
model.load_state_dict(params)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(self.genBackground())

        button_nn = QPushButton("Find center")
        button_nn.clicked.connect(self.findcenter)

        button_clear = QPushButton("Clear")
        button_clear.clicked.connect(self.reset)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(button_nn)
        self.layout.addWidget(button_clear)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.center = []
        self.rect_spawned = False

        self.last_x, self.last_y = None, None

    def genBackground(self):
        img = np.random.randint(0, 50, [300, 300], dtype=np.uint8)
        background = PIL.ImageQt.ImageQt(Image.fromarray(img))
        pix = QtGui.QPixmap.fromImage(background)
        return pix

    def reset(self):
        self.label.setPixmap(self.genBackground())

    def findcenter(self):
        if not self.rect_spawned:
            return
        new_center = list(map(lambda x: int(x * 64 / 300), self.center))
        y = new_center[1]
        x = new_center[0]
        img = np.random.randint(0, 50, [64, 64], dtype=np.uint8)
        square = np.random.randint(100, 200, [15, 15], dtype=np.uint8)
        img[(y - 7): (y + 8), (x - 7):(x + 8)] = square
        image = Image.fromarray(img)

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5,), std=(0.5,))
        ])
        picture = transform(image)
        picture = torch.unsqueeze(picture, 0).to(device)
        model.eval()
        prediction = model(picture)
        cords = torch.round(prediction).tolist()[0]
        cords = list(map(lambda cord: int(cord * 300 / 64), cords))

        #draw a point
        canvas = self.label.pixmap()
        painter = QtGui.QPainter(canvas)
        pen = QtGui.QPen()
        pen.setWidth(5)
        pen.setColor(QtGui.QColor('red'))
        painter.setPen(pen)
        painter.drawPoint(cords[1], cords[0])
        painter.end()
        self.label.setPixmap(canvas)


    def draw_rectangle(self):
        square = np.random.randint(100, 200, [70, 70], dtype=np.uint8)
        sq= PIL.ImageQt.ImageQt(Image.fromarray(square))
        pixsq = QtGui.QPixmap.fromImage(sq)
        self.label.setPixmap(self.genBackground())
        center = list(map(int, self.center))

        canvas = self.label.pixmap()
        painter = QtGui.QPainter(canvas)
        painter.drawPixmap(center[0]-35, center[1]-35, pixsq)
        painter.end()
        self.label.setPixmap(canvas)
        self.rect_spawned = True

    def mousePressEvent(self, e):
        x = e.position().x()
        y = e.position().y()
        if 106 < x < 202 and 106 < y < 202: # 70 / 2 + 9 + 60 (because of training dataset) and 309 - 70 / 2 - 9 - 60
            self.center = [x, y]
            self.draw_rectangle()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()