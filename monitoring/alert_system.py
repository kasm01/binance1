from core.notifier import send_notification
from core.logger import system_logger

class AlertSystem:
    def __init__(self, threshold_cpu=90, threshold_memory=90):
        self.threshold_cpu = threshold_cpu
        self.threshold_memory = threshold_memory

    def check_alerts(self, cpu, memory):
        if cpu > self.threshold_cpu:
            system_logger.warning(f"High CPU usage: {cpu}%")
            send_notification(f"High CPU usage detected: {cpu}%")
        if memory > self.threshold_memory:
            system_logger.warning(f"High Memory usage: {memory}%")
            send_notification(f"High Memory usage detected: {memory}%")
