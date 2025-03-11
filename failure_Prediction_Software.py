import sys
import time
import psutil
import requests
import numpy as np
import pickle
import json
import os
import shutil
import getpass
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
import win10toast  # Windows notifications
from pystray import Icon as TrayIcon, MenuItem, Menu
from PIL import Image

# Load AI Model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Remote logging API URL
REMOTE_API_URL = "https://your-server.com/api/log_failure"

class FailurePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.startMonitoring()
        self.createSystemTray()
        self.addToStartup()

    def initUI(self):
        self.setWindowTitle("AI Failure Predictor")
        self.setGeometry(100, 100, 400, 250)
        
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Monitoring system health...", self)
        layout.addWidget(self.status_label)
        
        self.check_button = QPushButton("Check Now", self)
        self.check_button.clicked.connect(self.manualCheck)
        layout.addWidget(self.check_button)
        
        self.setLayout(layout)
        self.show()

    def createSystemTray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("app_icon.png"))
        tray_menu = QMenu()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.exitApp)
        tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def exitApp(self):
        self.tray_icon.hide()
        sys.exit()

    def startMonitoring(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.monitorSystem)
        self.timer.start(10000)  # Check every 10 seconds

    def monitorSystem(self):
        system_data = self.collectSystemMetrics()
        failure_prediction = self.predictFailure(system_data)
        
        if failure_prediction == 1:
            self.alertUser("System Failure Predicted! Check your system immediately.")
            self.logFailure(system_data)

    def collectSystemMetrics(self):
        try:
            temps = psutil.sensors_temperatures().get('coretemp', [{}])
            temperature = temps[0].get('current', 50) if temps else 50  # Default to 50°C if unavailable
        except:
            temperature = 50

        return [
            psutil.cpu_percent(),
            psutil.virtual_memory().percent,
            psutil.disk_usage('/').percent,
            temperature,
            psutil.net_io_counters().bytes_sent / (1024 * 1024),
            psutil.net_io_counters().bytes_recv / (1024 * 1024)
        ]

    def predictFailure(self, data):
        features = np.array(data).reshape(1, -1)
        features = scaler.transform(features)
        return model.predict(features)[0]

    def alertUser(self, message):
        toaster = win10toast.ToastNotifier()
        toaster.show_toast("AI Failure Predictor", message, duration=5)
        QMessageBox.warning(self, "Warning", message)

    def logFailure(self, data):
        payload = json.dumps({"features": data})
        try:
            requests.post(REMOTE_API_URL, data=payload, headers={"Content-Type": "application/json"}, timeout=5)
        except requests.exceptions.RequestException:
            print("⚠️ Failed to send log: No internet connection.")

    def manualCheck(self):
        self.monitorSystem()

    def addToStartup(self):
        username = getpass.getuser()
        startup_path = f"C:\\Users\\{username}\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\failure_prediction.lnk"
        
        if not os.path.exists(startup_path):
            exe_path = os.path.abspath("dist/failure_prediction.exe")
            shutil.copy(exe_path, startup_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FailurePredictionApp()
    sys.exit(app.exec_())
