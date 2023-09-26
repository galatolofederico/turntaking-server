import multiprocessing
from AudioAnalyzer import analyze
from InformationSender import send_information

if __name__ == "__main__":
    uri = "ws://localhost:8765/"
    queue = multiprocessing.Queue()
    informationSender = multiprocessing.Process(target=send_information, args=(uri, queue, ))
    analyzer = multiprocessing.Process(target=analyze, args=(queue, ))

    informationSender.start()
    analyzer.start()

    informationSender.join()
    analyzer.join()
