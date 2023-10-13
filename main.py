import multiprocessing
from AnalyzerMultiprocessQueue import analyze
from InformationSender import send_information
from serverMultiprocessQueue import main

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    audio_queue = multiprocessing.Queue()
    latency_queue = multiprocessing.Queue()
    server = multiprocessing.Process(target=main, args=(audio_queue, latency_queue, ))
    analyzer = multiprocessing.Process(target=analyze, args=(audio_queue, latency_queue, ))

    server.start()
    analyzer.start()

    server.join()
    analyzer.join()
