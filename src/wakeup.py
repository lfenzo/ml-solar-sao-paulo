"""
ATTENTION!!

This script containg depracted code that SHOULD NOT BE USED AS PART OF THE PROJECT.
It was kept only as a record of the previous versions of the project during
development
"""

import os
from dask.distributed import Client, SSHCluster


def get_slaves(slavelist: list):
    return ' '.join(slavelist)


def initialize_on_demand_cluster(slavelist: list):
    """
    Initializes a cluster and returns a connected client reedy to use
    """

    cluster = SSHCluster(hosts = slavelist,
                          worker_options = {
                              'worker_port': '32000',
                              'nanny_port': '32001',
                              'nthreads': 4,
                              'memory_limit': 0.80,
                          })

    return Client('tcp://200.18.102.117:8786')


def initialize_dask_cluster(master: str, slaves: list):
    os.system('dask-ssh ' +
              f'--scheduler {master} ' +
              '--nthreads 4 ' +
              '--nprocs 1 ' +
              '--ssh-port 22 ' +
              '--nanny-port 32000 ' +
              '--worker-port 32001 ' +
              f'{get_slaves(slavelist = slaves)}')


if __name__ == '__main__':

    # insert master IP address here
    master = '200.18.102.117'

    # insert all IP addresses here
    slaves = ['200.18.102.117',
              '200.18.102.118',
              '200.18.102.119',
              '200.18.102.120',
              '200.18.102.121',
              '200.18.102.122',
              '200.18.102.123']

    initialize_on_demand_cluster(slaves)

    client = Client(f'tcp://{master}:8786')
