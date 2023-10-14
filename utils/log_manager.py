import os
from utils.log_writer import LogWriter

class LogManager:
    def __init__(self) -> None:
        self.writers = {}
    
    def setLogDirectory(self, dir_path: str) -> None:
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(self.dir_path)
            
        self.dir_path = "./" + dir_path + "/"
        
    def makeDir(self, path_dir: str) -> str:
        path_dir = self.dir_path + path_dir
        if not os.path.exists(path_dir):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(path_dir)
            
        return path_dir
    
    def createLogWriter(self, key: str) -> LogWriter:
        if key in self.writers:
            return self.writers[key]
        
        writer = LogWriter(self.dir_path + key + ".csv")
        self.writers[key] = writer
        return writer
    
    #テスト用
    def printWriters(self) -> None:
        print(self.writers)