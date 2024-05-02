from log_writer import LogWriter
from log_manager import LogManager

def LogWriterTest():
    logwriter = LogWriter("./test.csv")
    logwriter.write(str(100) + "," + str(200))
    logwriter.write(str(300))
    logwriter.writeLine(str(4.00))
    logwriter.writeLine(str(5.19) + "," + str(600))

def LogManagerTest():
    logmanager = LogManager()
    logmanager.setLogDirectory("test1")
    print(logmanager.makeDir("test2"))
    logmanager.createLogWriter("test3").writeLine("OK")
    a = logmanager.createLogWriter("test4")
    a.write("Yes")
    a.writeLine("")
    a.writeLine("No")
    logmanager.printWriters()


if __name__ == "__main__":
    #LogWriterTest()
    LogManagerTest()