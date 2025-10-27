#!/usr/bin/env python

import socket
import threading
import select
import sys
import time
import subprocess
import os
from datetime import datetime

terminateAll = False

# 全局变量用于记录活动时间和连接状态
global_last_activity = time.time()
global_connections = {}  # 存储连接ID和最后活动时间
global_lock = threading.Lock()
global_target_available = False
global_timeout = 30
global_start_script = ""
global_stop_script = ""
global_target_host = ""
global_target_port = 0

class ClientThread(threading.Thread):
    def __init__(self, clientSocket, targetHost, targetPort, connection_id):
        threading.Thread.__init__(self)
        self.__clientSocket = clientSocket
        self.__targetHost = targetHost
        self.__targetPort = targetPort
        self.__connection_id = connection_id
        self.__last_activity = time.time()
        self.__targetSocket = None
        self.__active = True
        
    def update_activity(self):
        global global_last_activity, global_connections
        with global_lock:
            self.__last_activity = time.time()
            global_last_activity = self.__last_activity
            global_connections[self.__connection_id] = self.__last_activity

    def run(self):
        global global_target_available, global_start_script, global_stop_script
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Client Thread started for connection {self.__connection_id}")
        
        self.__clientSocket.setblocking(0)
        
        # 尝试连接目标
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries and not terminateAll:
            try:
                self.__targetSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.__targetSocket.settimeout(5)  # 连接超时
                self.__targetSocket.connect((self.__targetHost, self.__targetPort))
                self.__targetSocket.setblocking(0)
                global_target_available = True
                self.update_activity()
                break
            except Exception as e:
                print(f"Failed to connect to target: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    print("Max retries reached, executing start script")
                    if global_start_script:
                        try:
                            start = time.time()
                            subprocess.run(global_start_script, shell=True, check=True)
                            if time.time() - start > 5:
                                print(f"Warning: start script shold run target app in the background")
                            time.sleep(2)  # 等待启动完成
                            retry_count = 0  # 重置重试计数
                        except Exception as script_error:
                            print(f"Start script failed: {script_error}")
                            break
                    else:
                        break
                else:
                    time.sleep(1)  # 等待后重试
        
        if not self.__targetSocket or terminateAll:
            self.__clientSocket.close()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ClientThread {self.__connection_id} terminating due to connection failure")
            return
        
        clientData = b""
        targetData = b""
        terminate = False
        
        while not terminate and not terminateAll and self.__active:
            inputs = [self.__clientSocket, self.__targetSocket]
            outputs = []
            
            if clientData:
                outputs.append(self.__clientSocket)
                
            if targetData:
                outputs.append(self.__targetSocket)
            
            try:
                inputsReady, outputsReady, errorsReady = select.select(inputs, outputs, [], 1.0)
            except Exception as e:
                print(f"Select error: {e}")
                break
                
            for inp in inputsReady:
                if inp == self.__clientSocket:
                    try:
                        data = self.__clientSocket.recv(4096)
                    except Exception as e:
                        print(f"Client recv error: {e}")
                        data = None
                    
                    if data:
                        if len(data) > 0:
                            targetData += data
                            self.update_activity()
                        else:
                            terminate = True
                    else:
                        terminate = True
                elif inp == self.__targetSocket:
                    try:
                        data = self.__targetSocket.recv(4096)
                    except Exception as e:
                        print(f"Target recv error: {e}")
                        data = None
                        
                    if data:
                        if len(data) > 0:
                            clientData += data
                            self.update_activity()
                        else:
                            terminate = True
                    else:
                        terminate = True
                        
            for out in outputsReady:
                if out == self.__clientSocket and clientData:
                    try:
                        bytesWritten = self.__clientSocket.send(clientData)
                        if bytesWritten > 0:
                            clientData = clientData[bytesWritten:]
                            self.update_activity()
                    except Exception as e:
                        print(f"Client send error: {e}")
                        terminate = True
                elif out == self.__targetSocket and targetData:
                    try:
                        bytesWritten = self.__targetSocket.send(targetData)
                        if bytesWritten > 0:
                            targetData = targetData[bytesWritten:]
                            self.update_activity()
                    except Exception as e:
                        print(f"Target send error: {e}")
                        terminate = True

        self.__clientSocket.close()
        if self.__targetSocket:
            self.__targetSocket.close()
        
        # 从全局连接记录中移除
        with global_lock:
            if self.__connection_id in global_connections:
                del global_connections[self.__connection_id]
                
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ClientThread {self.__connection_id} terminating")

def timeout_checker():
    global global_last_activity, global_target_available
    while not terminateAll:
        time.sleep(5)  # 每5秒检查一次
        
        with global_lock:
            current_time = time.time()
            time_since_last_activity = current_time - global_last_activity
            
            # 如果没有活动连接
            if not global_connections:

                # 如果超时且目标可用，执行停止脚本
                if time_since_last_activity > global_timeout and global_target_available:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Timeout reached ({time_since_last_activity:.1f}s > {global_timeout}s), executing stop script")
                    if global_stop_script:
                        try:
                            subprocess.run(global_stop_script, shell=True, check=True)
                            global_target_available = False
                        except Exception as e:
                            print(f"Stop script failed: {e}")
                    else:
                        print("No stop script specified")

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('Usage:\n\tpython proxy.py <listen port> <target host> <target port> <timeout> <start script> <stop script>')
        print('Example:\n\tpython proxy.py 8080 localhost 80 30 "start_service.sh" "stop_service.sh"')
        sys.exit(0)

    listen_port = int(sys.argv[1])
    global_target_host = sys.argv[2]
    global_target_port = int(sys.argv[3])
    global_timeout = int(sys.argv[4])
    global_start_script = sys.argv[5]
    global_stop_script = sys.argv[6]
    
    # 启动超时检查线程
    timeout_thread = threading.Thread(target=timeout_checker)
    timeout_thread.daemon = True
    timeout_thread.start()
    
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind(('0.0.0.0', listen_port))
    serverSocket.listen(10)
    
    print(f"Listening on port {listen_port}, forwarding to {global_target_host}:{global_target_port}")
    print(f"Timeout: {global_timeout}s, Start script: {global_start_script}, Stop script: {global_stop_script}")
    
    connection_counter = 0
    
    while True:
        try:
            clientSocket, address = serverSocket.accept()
            connection_counter += 1
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} New connection from {address[0]}:{address[1]}, ID: {connection_counter}")
            ClientThread(clientSocket, global_target_host, global_target_port, connection_counter).start()
        except KeyboardInterrupt:
            print("\nTerminating...")
            terminateAll = True
            break
        except Exception as e:
            print(f"Accept error: {e}")
            continue

    serverSocket.close()