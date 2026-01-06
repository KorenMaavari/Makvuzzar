#
#   @date:
#       28/12/25
#   @author:
#       Tal Ben Ami, 212525257
#       Koren Maavari, 207987314 
# 
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
from multiprocessing import Lock, Pipe


class MyQueue(object):
    def __init__(self):
        """Initialize MyQueue and its members."""
        self.put_lock = Lock()
        self.recv_conn, self.send_conn = Pipe(duplex=False)

    def put(self, msg):
        """Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        """
        with self.put_lock:
            self.send_conn.send(msg)

    def get(self):
        """Get the next message from queue (FIFO)

        Return
        ------
        An object
        """
        return self.recv_conn.recv()

    def empty(self):
        """Get whether the queue is currently empty

        Return
        ------
        A boolean value
        """
        return not self.recv_conn.poll()
