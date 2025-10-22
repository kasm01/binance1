import psutil
from core.logger import system_logger

class SystemHealth:
    def __init__(self):
        pass

    def check_cpu_memory(self):
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        system_logger.info(f"CPU usage: {cpu}%, Memory usage: {memory}%")
        return cpu, memory

    def check_api_connection(self, api_client):
        try:
            status = api_client.ping()
            return status
        except Exception as e:
            system_logger.error(f"API connection failed: {e}")
            return False
