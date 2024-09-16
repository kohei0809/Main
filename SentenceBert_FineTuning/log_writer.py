class LogWriter:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        
        #ファイルを新規作成 or 上書き
        with open(self.file_path, "w") as f:
            pass
            
    #改行なし
    def write(self, log: str) -> None:
        #ファイルへ追記
        with open(self.file_path, 'a', newline='') as f:
            f.write(log + ",")
            
    #改行あり
    def writeLine(self, log: str="") -> None:
        #ファイルへ追記
        with open(self.file_path, "a") as f:
            f.write(log + "\n")