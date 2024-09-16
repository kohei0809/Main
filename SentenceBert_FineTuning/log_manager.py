import os
from log_writer import LogWriter

class LogManager:
    def __init__(self) -> None:
        self.writers = {}
        self.dir_path = "./"
    
    def setLogDirectory(self, dir_path: str) -> None:
        self.dir_path = dir_path
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(self.dir_path, exist_ok=True)
            
        self.dir_path = "./" + dir_path + "/"
        
    def makeDir(self, path_dir: str) -> str:
        path_dir = self.dir_path + path_dir
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(path_dir, exist_ok=True)
            
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