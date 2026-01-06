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
import multiprocessing
import os

from my_queue import MyQueue
from network import *
from preprocessor import Worker


class IPNeuralNetwork(NeuralNetwork):
    def fit(self, training_data, validation_data=None):
        """
        Override this function to create and destroy workers
        """
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        self.jobs = multiprocessing.JoinableQueue()
        self.results = MyQueue()
        self.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

        # Start workers
        print(f"Creating {self.num_workers} workers")
        workers = [
            Worker(self.jobs, self.results, training_data, self.mini_batch_size)
            for i in range(self.num_workers)
        ]
        for w in workers:
            w.start()

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        # 3. Stop Workers
        for _ in range(self.num_workers):
            self.jobs.put(None)  # Send Poison Pill
        self.jobs.join()  # Wait for jobs to finish
        for w in workers:
            w.join()

    def create_batches(self, data, labels, batch_size):
        """
        Override this function to return self.number_of_batches batches created by workers
                Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        """
        num_jobs = self.number_of_batches
        for i in range(num_jobs):
            self.jobs.put(i)

        # Get all batches first (drains the pipe so workers don't block)
        ret = []
        for _ in range(num_jobs):
            ret.append(self.results.get())

        return ret
