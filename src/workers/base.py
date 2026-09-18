from threading import Event

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class WorkerCancelled(Exception):
    pass


class Worker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._cancel_requested = Event()

    def cancel(self):
        self._cancel_requested.set()

    def check_cancelled(self):
        if self._cancel_requested.is_set():
            raise WorkerCancelled

    @pyqtSlot()
    def run(self):
        try:
            self.check_cancelled()
            result = self.execute()
            self.check_cancelled()
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))

    def execute(self):
        raise NotImplementedError
